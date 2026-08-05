"""LTX trainer for full fine-tuning of the original audio branch only."""
from __future__ import annotations

import re
from dataclasses import replace
from functools import partial
from pathlib import Path

import torch
from accelerate import DistributedType
from ltx_core.model.transformer.transformer import BasicAVTransformerBlock
from ltx_core.text_encoders.gemma import convert_to_additive_mask
from safetensors.torch import load_file, save_file
from torch import Tensor
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from ltx_trainer.trainer_vanilla_base import *  # noqa: F403
from ltx_trainer.trainer_vanilla_base import LtxvTrainer as _BaseTrainer
from ltx_trainer.trainer_vanilla_base import TrainingStepOutput, logger


class LtxvTrainer(_BaseTrainer):
    """Train original audio-branch weights while keeping video and all LoRA weights frozen."""

    AUDIO_FULL_MODES = {"audio_full", "full_audio"}

    def _is_audio_full(self) -> bool:
        return str(self._config.model.training_mode) in self.AUDIO_FULL_MODES

    def _validate_audio_full(self) -> None:
        if not self._is_audio_full():
            return

        strategy = self._config.training_strategy
        video = getattr(strategy, "video", None)
        audio = getattr(strategy, "audio", None)

        if video is None or not getattr(video, "is_generated", False):
            raise ValueError("audio_full requires video.is_generated=true")
        if audio is None or not getattr(audio, "is_generated", False):
            raise ValueError("audio_full requires audio.is_generated=true")
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

    @classmethod
    def _is_audio_param(cls, raw: str) -> bool:
        name = cls._key(raw)
        if ".lora_" in name:
            return False

        prefixes = (
            "audio_patchify_proj.",
            "audio_caption_projection.",
            "audio_adaln_single.",
            "audio_prompt_adaln_single.",
            "audio_scale_shift_table",
            "audio_norm_out.",
            "audio_proj_out.",
            "av_ca_audio_scale_shift_adaln_single.",
            "av_ca_v2a_gate_adaln_single.",
        )
        markers = (
            ".audio_attn1.",
            ".audio_attn2.",
            ".audio_ff.",
            ".audio_scale_shift_table",
            ".audio_prompt_scale_shift_table",
            ".video_to_audio_attn.",
            ".scale_shift_table_a2v_ca_audio",
        )
        return name.startswith(prefixes) or any(marker in name for marker in markers)

    def _assert_no_lora_modules(self) -> None:
        lora_parameters = [
            name
            for name, _ in self._transformer.named_parameters()
            if ".lora_" in self._key(name)
        ]
        lora_modules = [
            name
            for name, module in self._transformer.named_modules()
            if hasattr(module, "lora_A") or hasattr(module, "lora_B")
        ]
        if lora_parameters or lora_modules:
            raise RuntimeError(
                "audio_full is configured for original audio-branch tuning only, "
                "but LoRA modules were found in the model: "
                f"parameters={lora_parameters[:20]}, modules={lora_modules[:20]}"
            )

    def _enable_audio_params(self) -> None:
        self._transformer.requires_grad_(False)
        for name, parameter in self._transformer.named_parameters():
            parameter.requires_grad_(self._is_audio_param(name))

        names = {
            self._key(name)
            for name, parameter in self._transformer.named_parameters()
            if parameter.requires_grad
        }
        if not names:
            raise RuntimeError("No audio parameters selected")

        frozen_branch_violations = [
            name
            for name in names
            if ".audio_to_video_attn." in name or ".lora_" in name
        ]
        if frozen_branch_violations:
            raise RuntimeError(
                f"Frozen branch was unfrozen: {frozen_branch_violations[:30]}"
            )

        self._audio_param_names = names
        trainable_count = sum(
            parameter.numel()
            for parameter in self._transformer.parameters()
            if parameter.requires_grad
        )
        logger.info(
            "Audio-only full tuning enabled: %s trainable parameters, "
            "video branch frozen, no LoRA modules.",
            f"{trainable_count:,}",
        )

    def _collect_trainable_params(self) -> None:
        if not self._is_audio_full():
            return super()._collect_trainable_params()

        # Do not call get_peft_model() or the base trainer's _setup_lora().
        # The loaded transformer must remain the original LTX model.
        self._assert_no_lora_modules()
        self._enable_audio_params()
        self._trainable_params = [
            parameter
            for parameter in self._transformer.parameters()
            if parameter.requires_grad
        ]

    def _training_step(
        self,
        batch: dict[str, dict[str, Tensor]],
    ) -> TrainingStepOutput:
        if not self._is_audio_full():
            return super()._training_step(batch)

        conditions = batch["conditions"]
        if "video_prompt_embeds" in conditions:
            video_features = conditions["video_prompt_embeds"]
            audio_features = conditions.get("audio_prompt_embeds")
        else:
            video_features = audio_features = conditions["prompt_embeds"]

        additive = convert_to_additive_mask(
            conditions["prompt_attention_mask"],
            video_features.dtype,
        )
        video_embeds, audio_embeds, mask = self._embeddings_processor.create_embeddings(
            video_features,
            audio_features,
            additive,
        )
        conditions.update(
            video_prompt_embeds=video_embeds,
            audio_prompt_embeds=audio_embeds,
            prompt_attention_mask=mask,
        )

        inputs = self._training_strategy.prepare_training_inputs(
            batch,
            self._timestep_sampler,
        )
        video_pred, audio_pred = self._transformer(
            video=inputs.video,
            audio=inputs.audio,
            perturbations=None,
        )
        if audio_pred is None or inputs.audio_targets is None:
            raise RuntimeError("Audio prediction/targets missing")

        # Keep video in the forward pass for AV context, but remove video
        # supervision so the backward pass is driven only by audio loss.
        inputs = replace(inputs, video_targets=None, video_loss_mask=None)
        loss = self._training_strategy.compute_loss(
            video_pred,
            audio_pred,
            inputs,
        )
        return TrainingStepOutput(
            loss=loss,
            sigma=inputs.audio.sigma.detach(),
        )

    def _find_checkpoint(self, checkpoint_path: str | Path) -> Path | None:
        if not self._is_audio_full():
            return super()._find_checkpoint(checkpoint_path)

        path = Path(checkpoint_path)
        if path.is_file():
            return path if path.suffix == ".safetensors" else None
        if not path.is_dir():
            raise ValueError(f"Invalid checkpoint path: {path}")

        files = list(path.rglob("audio_full_weights_step_*.safetensors"))
        if not files:
            return None
        return max(
            files,
            key=lambda candidate: int(
                re.search(r"step_(\d+)", candidate.name).group(1)
            ),
        )

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
        source = {
            self._key(key): value
            for key, value in load_file(str(path), device="cpu").items()
        }
        mapping = self._state_map()
        if set(source) != set(mapping):
            raise RuntimeError(
                "Audio checkpoint mismatch; "
                f"missing={sorted(set(mapping) - set(source))[:20]}, "
                f"unexpected={sorted(set(source) - set(mapping))[:20]}"
            )

        current = self._transformer.state_dict()
        native: dict[str, Tensor] = {}
        for key, source_value in source.items():
            destination = current[mapping[key]]
            if source_value.shape != destination.shape:
                raise RuntimeError(
                    f"Audio checkpoint shape mismatch for {key}: "
                    f"{source_value.shape} != {destination.shape}"
                )
            native[mapping[key]] = source_value.to(
                device=destination.device,
                dtype=destination.dtype,
            )

        incompatible = self._transformer.load_state_dict(native, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(
                f"Unexpected checkpoint keys: {incompatible.unexpected_keys[:20]}"
            )

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
            plugin = self._accelerator.state.fsdp_plugin
            if plugin is None or not getattr(plugin, "use_orig_params", False):
                raise RuntimeError(
                    "audio_full FSDP requires fsdp_use_orig_params=true"
                )

            # Wrap complete AV blocks only. Top-level audio preprocessors keep
            # direct references to patchify/AdaLN modules, so wrapping trainable
            # leaf Linear modules separately would expose one-dimensional FSDP
            # shards to those references.
            plugin.auto_wrap_policy = partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={BasicAVTransformerBlock},
            )

        self._transformer.set_gradient_checkpointing(
            self._config.optimization.enable_gradient_checkpointing
        )
        self._transformer = self._accelerator.prepare(self._transformer)

    def _save_checkpoint(self) -> Path | None:
        if not self._is_audio_full():
            return super()._save_checkpoint()

        save_dir = Path(self._config.output_dir) / "checkpoints"
        path = save_dir / (
            f"audio_full_weights_step_{self._global_step:05d}.safetensors"
        )

        self._accelerator.wait_for_everyone()
        full = self._accelerator.get_state_dict(self._transformer)
        if not self._accelerator.is_main_process:
            return None

        save_dir.mkdir(exist_ok=True, parents=True)
        dtype = (
            torch.bfloat16
            if self._config.checkpoints.precision == "bfloat16"
            else torch.float32
        )

        state: dict[str, Tensor] = {}
        for raw, value in full.items():
            key = self._key(raw)
            if key in self._audio_param_names:
                if key in state:
                    raise RuntimeError(f"Duplicate checkpoint key: {key}")
                state[key] = (
                    value.detach()
                    .to(device="cpu", dtype=dtype)
                    .contiguous()
                )

        missing = self._audio_param_names - set(state)
        if missing:
            raise RuntimeError(
                f"Missing audio weights: {sorted(missing)[:50]}"
            )

        metadata = self._build_checkpoint_metadata()
        metadata.update(
            format="ltx-audio-full",
            global_step=str(self._global_step),
            training_scope="original_audio_branch_only",
            uses_lora="false",
            contains_lora="false",
            video_branch_trainable="false",
        )
        save_file(
            {
                f"diffusion_model.{key}": value
                for key, value in state.items()
            },
            path,
            metadata=metadata,
        )
        self._checkpoint_paths.append(path)
        self._cleanup_checkpoints()
        self._save_training_state(save_dir)
        logger.info(f"Saved audio-only full weights to {path}")
        return path
