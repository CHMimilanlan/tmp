"""Vanilla LTX trainer with audio-full tuning and a frozen Motion Track IC-LoRA."""
from __future__ import annotations

import os
import re
from dataclasses import replace
from pathlib import Path

import torch
from accelerate import DistributedType
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor
from ltx_core.text_encoders.gemma import convert_to_additive_mask

from ltx_trainer.trainer_vanilla_base import *  # noqa: F403
from ltx_trainer.trainer_vanilla_base import LtxvTrainer as _BaseTrainer
from ltx_trainer.trainer_vanilla_base import TrainingStepOutput, logger


class LtxvTrainer(_BaseTrainer):
    AUDIO_FULL_MODES = {"audio_full", "full_audio"}

    def _frozen_lora_path(self) -> Path | None:
        model = self._config.model
        lora = getattr(self._config, "lora", None)
        value = (
            getattr(model, "frozen_lora_path", None)
            or getattr(model, "frozen_ic_lora_path", None)
            or getattr(lora, "init_checkpoint", None)
            or os.environ.get("LTX_FROZEN_IC_LORA")
        )
        return Path(value).expanduser().resolve() if value else None

    def _is_audio_full(self) -> bool:
        mode = str(self._config.model.training_mode)
        return mode in self.AUDIO_FULL_MODES or (
            mode == "full" and self._frozen_lora_path() is not None
        )

    def _validate_audio_full(self) -> None:
        if not self._is_audio_full():
            return
        lora = getattr(self._config, "lora", None)
        path = self._frozen_lora_path()
        if lora is None:
            raise ValueError("audio_full requires a lora section with rank and alpha")
        if float(getattr(lora, "dropout", 0.0)) != 0.0:
            raise ValueError("Frozen IC-LoRA requires lora.dropout=0.0")
        if path is None or not path.is_file():
            raise FileNotFoundError(
                "Set model.frozen_lora_path, lora.init_checkpoint, or LTX_FROZEN_IC_LORA"
            )
        strategy = self._config.training_strategy
        video, audio = getattr(strategy, "video", None), getattr(strategy, "audio", None)
        if video is None or not getattr(video, "is_generated", False):
            raise ValueError("audio_full requires video.is_generated=true")
        if audio is None or not getattr(audio, "is_generated", False):
            raise ValueError("audio_full requires audio.is_generated=true")
        if not any(getattr(c, "type", None) == "reference" for c in getattr(video, "conditions", [])):
            raise ValueError("audio_full requires a video reference condition")
        if self._config.acceleration.quantization is not None:
            raise ValueError("Quantization is not supported in audio_full mode")

    def _load_models(self) -> None:
        self._validate_audio_full()
        super()._load_models()

    @staticmethod
    def _key(key: str) -> str:
        key = key.removeprefix("diffusion_model.")
        prefixes = ("_fsdp_wrapped_module.", "_orig_mod.", "module.", "base_model.model.")
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key, changed = key[len(prefix):], True
        return key.replace(".base_layer.", ".")

    @staticmethod
    def _is_lora_key(key: str) -> bool:
        return key.endswith((".lora_A.weight", ".lora_B.weight"))

    def _motion_targets(self) -> list[str]:
        path = self._frozen_lora_path()
        assert path is not None
        modules: set[str] = set()
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for raw in f.keys():
                key = self._key(raw)
                for suffix in (".lora_A.weight", ".lora_B.weight"):
                    if key.endswith(suffix):
                        modules.add(key[:-len(suffix)])
        if not modules:
            raise RuntimeError(f"No LoRA tensors found in {path}")
        return sorted(modules)

    def _setup_frozen_lora(self) -> None:
        cfg = self._config.lora
        self._transformer = get_peft_model(
            self._transformer,
            LoraConfig(
                r=cfg.rank, lora_alpha=cfg.alpha,
                target_modules=self._motion_targets(), lora_dropout=0.0,
                init_lora_weights=True,
            ),
        )
        path = self._frozen_lora_path()
        assert path is not None
        source = {
            self._key(k): v for k, v in load_file(str(path), device="cpu").items()
            if self._is_lora_key(self._key(k))
        }
        current = get_peft_model_state_dict(self._transformer, adapter_name="default")
        mapping = {self._key(k): k for k in current}
        expected = {k for k in mapping if self._is_lora_key(k)}
        if set(source) != expected:
            raise RuntimeError(
                f"IC-LoRA key mismatch; missing={sorted(expected-set(source))[:20]}, "
                f"unexpected={sorted(set(source)-expected)[:20]}"
            )
        native: dict[str, Tensor] = {}
        for key in expected:
            src, dst = source[key], current[mapping[key]]
            if src.shape != dst.shape:
                raise RuntimeError(f"IC-LoRA shape mismatch for {key}: {src.shape} != {dst.shape}")
            native[mapping[key]] = src.to(device=dst.device, dtype=dst.dtype)
        set_peft_model_state_dict(self._transformer, native, adapter_name="default")
        self._transformer.requires_grad_(False)
        logger.info(f"Loaded and froze Motion Track IC-LoRA: {path}")

    @classmethod
    def _is_audio_param(cls, raw: str) -> bool:
        name = cls._key(raw)
        if ".lora_" in name:
            return False
        prefixes = (
            "audio_patchify_proj.", "audio_caption_projection.", "audio_adaln_single.",
            "audio_prompt_adaln_single.", "audio_scale_shift_table", "audio_norm_out.",
            "audio_proj_out.", "av_ca_audio_scale_shift_adaln_single.",
            "av_ca_v2a_gate_adaln_single.",
        )
        markers = (
            ".audio_attn1.", ".audio_attn2.", ".audio_ff.",
            ".audio_scale_shift_table", ".audio_prompt_scale_shift_table",
            ".video_to_audio_attn.", ".scale_shift_table_a2v_ca_audio",
        )
        return name.startswith(prefixes) or any(m in name for m in markers)

    def _enable_audio_params(self) -> None:
        self._transformer.requires_grad_(False)
        for name, parameter in self._transformer.named_parameters():
            parameter.requires_grad_(self._is_audio_param(name))
        names = {self._key(n) for n, p in self._transformer.named_parameters() if p.requires_grad}
        if not names:
            raise RuntimeError("No audio parameters selected")
        bad = [n for n in names if ".audio_to_video_attn." in n]
        lora_bad = [n for n, p in self._transformer.named_parameters() if ".lora_" in n and p.requires_grad]
        if bad or lora_bad:
            raise RuntimeError(f"Frozen branch was unfrozen: {(bad+lora_bad)[:30]}")
        self._audio_param_names = names
        logger.info(f"Audio-full trainable params: {sum(p.numel() for p in self._transformer.parameters() if p.requires_grad):,}")

    def _collect_trainable_params(self) -> None:
        if not self._is_audio_full():
            return super()._collect_trainable_params()
        self._setup_frozen_lora()
        self._enable_audio_params()
        self._trainable_params = [p for p in self._transformer.parameters() if p.requires_grad]

    def _training_step(self, batch: dict[str, dict[str, Tensor]]) -> TrainingStepOutput:
        if not self._is_audio_full():
            return super()._training_step(batch)
        conditions = batch["conditions"]
        if "video_prompt_embeds" in conditions:
            video_features = conditions["video_prompt_embeds"]
            audio_features = conditions.get("audio_prompt_embeds")
        else:
            video_features = audio_features = conditions["prompt_embeds"]
        additive = convert_to_additive_mask(
            conditions["prompt_attention_mask"], video_features.dtype
        )
        video_embeds, audio_embeds, mask = self._embeddings_processor.create_embeddings(
            video_features, audio_features, additive
        )
        conditions.update(
            video_prompt_embeds=video_embeds,
            audio_prompt_embeds=audio_embeds,
            prompt_attention_mask=mask,
        )
        inputs = self._training_strategy.prepare_training_inputs(batch, self._timestep_sampler)
        video_pred, audio_pred = self._transformer(
            video=inputs.video,
            audio=inputs.audio,
            perturbations=None,
        )
        if audio_pred is None or inputs.audio_targets is None:
            raise RuntimeError("Audio prediction/targets missing")
        inputs = replace(inputs, video_targets=None, video_loss_mask=None)
        loss = self._training_strategy.compute_loss(video_pred, audio_pred, inputs)
        return TrainingStepOutput(loss=loss, sigma=inputs.audio.sigma.detach())

    def _find_checkpoint(self, checkpoint_path: str | Path) -> Path | None:
        if not self._is_audio_full():
            return super()._find_checkpoint(checkpoint_path)
        path = Path(checkpoint_path)
        if path.is_file():
            return path if path.suffix == ".safetensors" else None
        if not path.is_dir():
            raise ValueError(f"Invalid checkpoint path: {path}")
        files = list(path.rglob("audio_full_weights_step_*.safetensors"))
        return max(files, key=lambda p: int(re.search(r"step_(\d+)", p.name).group(1))) if files else None

    def _state_map(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for native in self._transformer.state_dict():
            key = self._key(native)
            if key in self._audio_param_names:
                if key in result:
                    raise RuntimeError(f"Duplicate state key: {key}")
                result[key] = native
        return result

    def _load_audio_checkpoint(self, path: Path) -> None:
        source = {self._key(k): v for k, v in load_file(str(path), device="cpu").items()}
        mapping = self._state_map()
        if set(source) != set(mapping):
            raise RuntimeError(
                f"Audio checkpoint mismatch; missing={sorted(set(mapping)-set(source))[:20]}, "
                f"unexpected={sorted(set(source)-set(mapping))[:20]}"
            )
        current, native = self._transformer.state_dict(), {}
        for key, src in source.items():
            dst = current[mapping[key]]
            if src.shape != dst.shape:
                raise RuntimeError(f"Audio checkpoint shape mismatch: {key}")
            native[mapping[key]] = src.to(device=dst.device, dtype=dst.dtype)
        incompatible = self._transformer.load_state_dict(native, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(f"Unexpected checkpoint keys: {incompatible.unexpected_keys[:20]}")

    def _load_checkpoint(self) -> None:
        if not self._is_audio_full():
            return super()._load_checkpoint()
        value = self._config.model.load_checkpoint
        if not value:
            self._resume_state = (0, None)
            return
        path = self._find_checkpoint(value)
        if path is None:
            self._resume_state = (0, None)
            return
        self._loaded_checkpoint_path = path
        self._load_audio_checkpoint(path)
        self._resume_state = self._resolve_resume_state()

    def _prepare_models_for_training(self) -> None:
        if not self._is_audio_full():
            return super()._prepare_models_for_training()
        if self._accelerator.distributed_type == DistributedType.FSDP:
            self._transformer = self._transformer.to(dtype=torch.float32)
            from peft.utils.other import fsdp_auto_wrap_policy
            plugin = self._accelerator.state.fsdp_plugin
            if plugin is None or not getattr(plugin, "use_orig_params", False):
                raise RuntimeError("audio_full FSDP requires fsdp_use_orig_params=true")
            
            plugin.auto_wrap_policy = fsdp_auto_wrap_policy(self._transformer)
        base = self._transformer.get_base_model()
        base.set_gradient_checkpointing(self._config.optimization.enable_gradient_checkpointing)
        self._transformer = self._accelerator.prepare(self._transformer)

    def _save_checkpoint(self) -> Path | None:
        if not self._is_audio_full():
            return super()._save_checkpoint()
        save_dir = Path(self._config.output_dir) / "checkpoints"
        path = save_dir / f"audio_full_weights_step_{self._global_step:05d}.safetensors"
        self._accelerator.wait_for_everyone()
        full = self._accelerator.get_state_dict(self._transformer)
        if not self._accelerator.is_main_process:
            return None
        save_dir.mkdir(exist_ok=True, parents=True)
        dtype = torch.bfloat16 if self._config.checkpoints.precision == "bfloat16" else torch.float32
        state: dict[str, Tensor] = {}
        for raw, value in full.items():
            key = self._key(raw)
            if key in self._audio_param_names:
                if key in state:
                    raise RuntimeError(f"Duplicate checkpoint key: {key}")
                state[key] = value.detach().to(device="cpu", dtype=dtype).contiguous()
        missing = self._audio_param_names - set(state)
        if missing:
            raise RuntimeError(f"Missing audio weights: {sorted(missing)[:50]}")
        metadata = self._build_checkpoint_metadata()
        metadata.update(
            format="ltx-audio-full",
            global_step=str(self._global_step),
            frozen_ic_lora=self._frozen_lora_path().name,
            contains_frozen_ic_lora="false",
            audio_to_video_cross_attention="enabled_during_training",
        )
        save_file({f"diffusion_model.{k}": v for k, v in state.items()}, path, metadata=metadata)
        self._checkpoint_paths.append(path)
        self._cleanup_checkpoints()
        self._save_training_state(save_dir)
        logger.info(f"Saved audio-full weights to {path}")
        return path
