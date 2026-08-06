"""Evaluate audio-branch full fine-tuning checkpoints with or without Motion Track IC-LoRA.

The audio-full trainer saves only the original audio-side transformer weights. This
pipeline overlays those weights onto the base LTX checkpoint before optional LoRA
fusion, preserving the exact training-time composition:

1. ``audio_full_ic_lora``: base model -> trained audio weights -> Motion Track IC-LoRA.
2. ``audio_full``: base model -> trained audio weights, with no LoRA or reference video.

The overlay is applied inside the state-dict loader, before the model is materialized.
This is important when the frozen IC-LoRA contains audio-side targets: loading the
Audio checkpoint after LoRA fusion would overwrite the LoRA delta.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path

import torch
from safetensors import safe_open

from ltx_core.components.noisers import GaussianNoiser
from ltx_core.conditioning import ConditioningItem
from ltx_core.loader import LoraPathStrengthAndSDOps, SDOps
from ltx_core.loader.primitives import StateDict
from ltx_core.loader.registry import DummyRegistry, Registry
from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader
from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
from ltx_core.model.transformer import LTXModelConfigurator, LTXV_MODEL_COMFY_RENAMING_MAP
from ltx_core.model.transformer.compiling import CompilationConfig
from ltx_core.model.video_vae import TilingConfig, VideoEncoder, get_video_chunks_number
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import Audio, VideoPixelShape
from ltx_pipelines.iclora_utils import (
    append_ic_lora_reference_video_conditionings,
    read_lora_reference_downscale_factor,
    read_lora_reference_temporal_scale_factor,
)
from ltx_pipelines.utils import args as pipeline_args
from ltx_pipelines.utils.allocator_trim_strategy import AllocatorTrimStrategy
from ltx_pipelines.utils.blocks import (
    AudioDecoder,
    DiffusionStage,
    ImageConditioner,
    PromptEncoder,
    VideoDecoder,
    VideoUpsampler,
)
from ltx_pipelines.utils.constants import DISTILLED_SIGMAS, STAGE_2_DISTILLED_SIGMAS, detect_params
from ltx_pipelines.utils.denoisers import SimpleDenoiser
from ltx_pipelines.utils.helpers import assert_resolution, combined_image_conditionings, get_device
from ltx_pipelines.utils.media_io import decode_video_by_frame, encode_video, video_preprocess
from ltx_pipelines.utils.types import ModalitySpec, OffloadMode

logger = logging.getLogger(__name__)

ImageConditioningInput = pipeline_args.ImageConditioningInput
VideoConditioningAction = pipeline_args.VideoConditioningAction
VideoMaskConditioningAction = pipeline_args.VideoMaskConditioningAction


class EvaluationMode(str, Enum):
    """Supported audio-full evaluation compositions."""

    AUDIO_FULL_IC_LORA = "audio_full_ic_lora"
    AUDIO_FULL = "audio_full"


@dataclass(frozen=True)
class AudioFullCheckpointInfo:
    """Resolved audio-full checkpoint and its safetensors metadata."""

    path: str
    metadata: dict[str, str]
    inferred_mode: EvaluationMode | None


_AUDIO_PREFIXES = (
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
_AUDIO_MARKERS = (
    ".audio_attn1.",
    ".audio_attn2.",
    ".audio_ff.",
    ".audio_scale_shift_table",
    ".audio_prompt_scale_shift_table",
    ".video_to_audio_attn.",
    ".scale_shift_table_a2v_ca_audio",
)


def _canonical_key(raw_key: str) -> str:
    """Normalize trainer, PEFT, FSDP and compiled-model key wrappers."""

    key = raw_key
    leading_prefixes = (
        "model.diffusion_model.",
        "diffusion_model.",
        "velocity_model.",
        "_fsdp_wrapped_module.",
        "_orig_mod.",
        "module.",
        "base_model.model.",
    )
    changed = True
    while changed:
        changed = False
        for prefix in leading_prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True

    # torch.compile may wrap each transformer block rather than the whole model.
    key = key.replace("._orig_mod.", ".")
    key = key.replace("._fsdp_wrapped_module.", ".")
    key = key.replace(".base_layer.", ".")
    return key


def _is_audio_full_key(raw_key: str) -> bool:
    """Match exactly the parameter scope selected by trainer_audio_full.py."""

    key = _canonical_key(raw_key)
    if ".lora_" in key:
        return False
    return key.startswith(_AUDIO_PREFIXES) or any(marker in key for marker in _AUDIO_MARKERS)


def _normalise_paths(paths: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(paths, str):
        values = (paths,)
    else:
        values = tuple(paths)
    return tuple(str(Path(path).expanduser().resolve()) for path in values)


def _read_safetensors_metadata(path: str) -> dict[str, str]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    return {str(key): str(value) for key, value in metadata.items()}


def _infer_checkpoint_mode(metadata: dict[str, str]) -> EvaluationMode | None:
    uses_lora = metadata.get("uses_lora", "").strip().lower()
    training_scope = metadata.get("training_scope", "").strip().lower()
    frozen_lora = metadata.get("frozen_ic_lora", "").strip()

    if uses_lora == "false" or training_scope == "original_audio_branch_only":
        return EvaluationMode.AUDIO_FULL
    if frozen_lora and frozen_lora.lower() not in {"none", "false", "null"}:
        return EvaluationMode.AUDIO_FULL_IC_LORA
    return None


def resolve_audio_full_checkpoint(path_or_directory: str) -> AudioFullCheckpointInfo:
    """Resolve a checkpoint file or the latest checkpoint inside a directory."""

    path = Path(path_or_directory).expanduser().resolve()
    if path.is_dir():
        candidates = list(path.rglob("audio_full_weights_step_*.safetensors"))
        if not candidates:
            raise FileNotFoundError(f"No audio_full_weights_step_*.safetensors found under {path}")

        def step_number(candidate: Path) -> int:
            match = re.search(r"step_(\d+)", candidate.name)
            return int(match.group(1)) if match else -1

        path = max(candidates, key=step_number)
    elif not path.is_file():
        raise FileNotFoundError(f"Audio-full checkpoint does not exist: {path}")

    if path.suffix != ".safetensors":
        raise ValueError(f"Audio-full checkpoint must be a .safetensors file: {path}")

    metadata = _read_safetensors_metadata(str(path))
    checkpoint_format = metadata.get("format")
    if checkpoint_format not in (None, "ltx-audio-full"):
        raise ValueError(
            f"Unsupported checkpoint format {checkpoint_format!r} in {path}; expected 'ltx-audio-full'"
        )

    return AudioFullCheckpointInfo(
        path=str(path),
        metadata=metadata,
        inferred_mode=_infer_checkpoint_mode(metadata),
    )


class AudioFullOverlayStateDictLoader:
    """Load the base model and overlay trained audio weights before LoRA fusion.

    ``SingleGPUModelBuilder`` invokes its state-dict loader for both the base model
    and LoRA files. This wrapper overlays only when the requested path is the base
    checkpoint. The builder then applies Motion Track IC-LoRA to the already-updated
    state dict, reproducing the training-time order exactly.
    """

    def __init__(
        self,
        base_checkpoint_path: str | Sequence[str],
        audio_full_checkpoint_path: str,
    ) -> None:
        self._delegate = SafetensorsModelStateDictLoader()
        self._base_paths = _normalise_paths(base_checkpoint_path)
        self._audio_full_checkpoint_path = str(Path(audio_full_checkpoint_path).expanduser().resolve())

    def metadata(self, path: str) -> dict:
        return self._delegate.metadata(path)

    def load(
        self,
        path: str | list[str],
        sd_ops: SDOps | None = None,
        device: torch.device | None = None,
    ) -> StateDict:
        state = self._delegate.load(path, sd_ops=sd_ops, device=device)
        if _normalise_paths(path) != self._base_paths:
            return state
        return self._overlay_audio_weights(state)

    def _overlay_audio_weights(self, base_state: StateDict) -> StateDict:
        actual_by_canonical: dict[str, str] = {}
        for actual_key in base_state.sd:
            canonical = _canonical_key(actual_key)
            if canonical in actual_by_canonical:
                raise RuntimeError(
                    "Duplicate canonical base-model key while preparing audio checkpoint overlay: "
                    f"{canonical!r} maps to both {actual_by_canonical[canonical]!r} and {actual_key!r}"
                )
            actual_by_canonical[canonical] = actual_key

        expected_audio_keys = {
            canonical for canonical in actual_by_canonical if _is_audio_full_key(canonical)
        }
        if not expected_audio_keys:
            raise RuntimeError("Base checkpoint exposes no audio-branch weights after state-dict mapping")

        source_by_canonical: dict[str, str] = {}
        with safe_open(self._audio_full_checkpoint_path, framework="pt", device="cpu") as handle:
            for raw_key in handle.keys():
                canonical = _canonical_key(raw_key)
                if not _is_audio_full_key(canonical):
                    raise RuntimeError(
                        "Audio-full checkpoint contains a non-audio or LoRA tensor: "
                        f"{raw_key!r} -> {canonical!r}"
                    )
                if canonical in source_by_canonical:
                    raise RuntimeError(f"Duplicate canonical audio checkpoint key: {canonical}")
                source_by_canonical[canonical] = raw_key

            source_keys = set(source_by_canonical)
            missing = expected_audio_keys - source_keys
            unexpected = source_keys - expected_audio_keys
            if missing or unexpected:
                raise RuntimeError(
                    "Audio-full checkpoint is incompatible with the selected base model; "
                    f"missing={sorted(missing)[:30]}, unexpected={sorted(unexpected)[:30]}"
                )

            merged = dict(base_state.sd)
            loaded_bytes = 0
            for canonical in sorted(expected_audio_keys):
                actual_key = actual_by_canonical[canonical]
                target = base_state.sd[actual_key]
                source = handle.get_tensor(source_by_canonical[canonical])
                if source.shape != target.shape:
                    raise RuntimeError(
                        f"Audio-full checkpoint shape mismatch for {canonical}: "
                        f"{tuple(source.shape)} != {tuple(target.shape)}"
                    )
                source = source.to(device=target.device, dtype=target.dtype, non_blocking=True)
                merged[actual_key] = source
                loaded_bytes += source.nbytes

        logger.info(
            "Overlaid %d audio-full tensors (%.2f GiB) from %s",
            len(expected_audio_keys),
            loaded_bytes / (1024**3),
            self._audio_full_checkpoint_path,
        )
        return replace(
            base_state,
            sd=merged,
            size=sum(tensor.nbytes for tensor in merged.values()),
            dtype={tensor.dtype for tensor in merged.values()},
        )


def _build_audio_full_stage(
    *,
    checkpoint_path: str,
    audio_full_checkpoint_path: str,
    dtype: torch.dtype,
    device: torch.device,
    loras: tuple[LoraPathStrengthAndSDOps, ...],
    registry: Registry | None,
    compilation_config: CompilationConfig | None,
    alloc_trim_strategy: AllocatorTrimStrategy,
) -> DiffusionStage:
    """Build a standard DiffusionStage with the audio overlay loader."""

    stage_registry = registry or DummyRegistry()
    loader = AudioFullOverlayStateDictLoader(
        base_checkpoint_path=checkpoint_path,
        audio_full_checkpoint_path=audio_full_checkpoint_path,
    )
    builder = SingleGPUModelBuilder(
        model_class_configurator=LTXModelConfigurator,
        model_path=checkpoint_path,
        model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
        loras=loras,
        model_loader=loader,
        registry=stage_registry,
    )
    return DiffusionStage(
        builder,
        dtype,
        device,
        quantization=None,
        compilation_config=compilation_config,
        alloc_trim_strategy=alloc_trim_strategy,
    )


def _validate_mode_configuration(
    *,
    mode: EvaluationMode,
    checkpoint_info: AudioFullCheckpointInfo,
    loras: tuple[LoraPathStrengthAndSDOps, ...],
    video_conditioning: Sequence[tuple[str, float]],
    conditioning_attention_mask: object | None,
    quantization: QuantizationPolicy | None,
    offload_mode: OffloadMode,
) -> None:
    """Reject silently mismatched evaluation compositions."""

    if quantization is not None:
        raise ValueError(
            "Audio-full checkpoint overlays currently require unquantized inference. "
            "Quantization changes the state-dict layout and would invalidate strict weight checks."
        )
    if offload_mode != OffloadMode.NONE:
        raise ValueError(
            "Audio-full checkpoint overlays currently require --offload-mode none. "
            "CPU/disk block streaming needs a streaming weights-provider overlay implementation."
        )

    inferred = checkpoint_info.inferred_mode
    if inferred is not None and inferred != mode:
        raise ValueError(
            f"Checkpoint metadata indicates mode {inferred.value!r}, but --evaluation-mode is {mode.value!r}"
        )

    if mode == EvaluationMode.AUDIO_FULL_IC_LORA:
        if len(loras) != 1:
            raise ValueError(
                "audio_full_ic_lora requires exactly one --lora: the frozen Sparse Motion Track IC-LoRA"
            )
        lora = loras[0]
        if abs(float(lora.strength) - 1.0) > 1e-8:
            raise ValueError(
                "The frozen Motion Track IC-LoRA must be evaluated at strength 1.0 to match training; "
                f"got {lora.strength}"
            )
        if not video_conditioning:
            raise ValueError("audio_full_ic_lora requires at least one --video-conditioning reference")

        expected_lora_name = checkpoint_info.metadata.get("frozen_ic_lora")
        if expected_lora_name and expected_lora_name.lower() not in {"none", "false", "null"}:
            actual_lora_name = Path(lora.path).name
            if actual_lora_name != Path(expected_lora_name).name:
                raise ValueError(
                    "The supplied IC-LoRA does not match the checkpoint training metadata: "
                    f"expected {expected_lora_name!r}, got {actual_lora_name!r}"
                )
    else:
        if loras:
            raise ValueError("audio_full mode must not receive --lora; this evaluation contains no LoRA")
        if video_conditioning:
            raise ValueError(
                "audio_full mode must not receive --video-conditioning; reference tokens belong to IC-LoRA evaluation"
            )
        if conditioning_attention_mask is not None:
            raise ValueError("--conditioning-attention-mask is only valid in audio_full_ic_lora mode")


class AudioFullEvaluationPipeline:
    """Two-stage evaluation pipeline for both audio-full training schemes."""

    def __init__(
        self,
        distilled_checkpoint_path: str,
        spatial_upsampler_path: str,
        gemma_root: str,
        audio_full_checkpoint_path: str,
        evaluation_mode: EvaluationMode | str,
        loras: Sequence[LoraPathStrengthAndSDOps] = (),
        device: torch.device | None = None,
        quantization: QuantizationPolicy | None = None,
        registry: Registry | None = None,
        compilation_config: CompilationConfig | None = None,
        offload_mode: OffloadMode = OffloadMode.NONE,
        alloc_trim_strategy: AllocatorTrimStrategy = AllocatorTrimStrategy.TRIM,
    ) -> None:
        self.device = device or get_device()
        self.dtype = torch.bfloat16
        self.mode = EvaluationMode(evaluation_mode)
        self.checkpoint_info = resolve_audio_full_checkpoint(audio_full_checkpoint_path)
        self._loras = tuple(loras)
        self._quantization = quantization
        self._offload_mode = offload_mode

        # Full mode validation also happens in __call__, once video conditioning is known.
        if quantization is not None:
            raise ValueError("Audio-full evaluation does not support quantization")
        if offload_mode != OffloadMode.NONE:
            raise ValueError("Audio-full evaluation currently supports only offload_mode=none")

        self.prompt_encoder = PromptEncoder(
            distilled_checkpoint_path,
            gemma_root,
            self.dtype,
            self.device,
            registry=registry,
            offload_mode=offload_mode,
            alloc_trim_strategy=alloc_trim_strategy,
        )
        self.image_conditioner = ImageConditioner(
            distilled_checkpoint_path,
            self.dtype,
            self.device,
            registry=registry,
            alloc_trim_strategy=alloc_trim_strategy,
        )

        stage_1_loras = self._loras if self.mode == EvaluationMode.AUDIO_FULL_IC_LORA else ()
        self.stage_1 = _build_audio_full_stage(
            checkpoint_path=distilled_checkpoint_path,
            audio_full_checkpoint_path=self.checkpoint_info.path,
            dtype=self.dtype,
            device=self.device,
            loras=stage_1_loras,
            registry=registry,
            compilation_config=compilation_config,
            alloc_trim_strategy=alloc_trim_strategy,
        )
        # Stage 2 refines without IC-LoRA/reference tokens, but it must still use
        # the trained audio branch for consistent audio denoising.
        self.stage_2 = _build_audio_full_stage(
            checkpoint_path=distilled_checkpoint_path,
            audio_full_checkpoint_path=self.checkpoint_info.path,
            dtype=self.dtype,
            device=self.device,
            loras=(),
            registry=registry,
            compilation_config=compilation_config,
            alloc_trim_strategy=alloc_trim_strategy,
        )

        self.upsampler = VideoUpsampler(
            distilled_checkpoint_path,
            spatial_upsampler_path,
            self.dtype,
            self.device,
            registry=registry,
            alloc_trim_strategy=alloc_trim_strategy,
        )
        self.video_decoder = VideoDecoder(
            distilled_checkpoint_path,
            self.dtype,
            self.device,
            registry=registry,
            alloc_trim_strategy=alloc_trim_strategy,
        )
        self.audio_decoder = AudioDecoder(
            distilled_checkpoint_path,
            self.dtype,
            self.device,
            registry=registry,
            alloc_trim_strategy=alloc_trim_strategy,
        )

        self.reference_downscale_factor = 1
        self.reference_temporal_scale_factor = 1
        if self.mode == EvaluationMode.AUDIO_FULL_IC_LORA:
            for lora in self._loras:
                self._update_reference_scales(lora)

    def _update_reference_scales(self, lora: LoraPathStrengthAndSDOps) -> None:
        scale = read_lora_reference_downscale_factor(lora.path)
        if scale != 1:
            if self.reference_downscale_factor not in (1, scale):
                raise ValueError(
                    "Conflicting reference_downscale_factor values: "
                    f"{self.reference_downscale_factor} vs {scale} from {lora.path}"
                )
            self.reference_downscale_factor = scale

        temporal = read_lora_reference_temporal_scale_factor(lora.path)
        if temporal != 1:
            if self.reference_temporal_scale_factor not in (1, temporal):
                raise ValueError(
                    "Conflicting reference_temporal_scale_factor values: "
                    f"{self.reference_temporal_scale_factor} vs {temporal} from {lora.path}"
                )
            self.reference_temporal_scale_factor = temporal

    def __call__(  # noqa: PLR0913
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        video_conditioning: Sequence[tuple[str, float]] = (),
        enhance_prompt: bool = False,
        tiling_config: TilingConfig | None = None,
        conditioning_attention_strength: float = 1.0,
        skip_stage_2: bool = False,
        conditioning_attention_mask: torch.Tensor | None = None,
        stage_1_sigmas: torch.Tensor = DISTILLED_SIGMAS,
        stage_2_sigmas: torch.Tensor = STAGE_2_DISTILLED_SIGMAS,
    ) -> tuple[Iterator[torch.Tensor], Audio]:
        assert_resolution(height=height, width=width, is_two_stage=True)
        if not 0.0 <= conditioning_attention_strength <= 1.0:
            raise ValueError(
                f"conditioning_attention_strength must be in [0, 1], got {conditioning_attention_strength}"
            )

        video_conditioning = tuple(video_conditioning)
        _validate_mode_configuration(
            mode=self.mode,
            checkpoint_info=self.checkpoint_info,
            loras=self._loras,
            video_conditioning=video_conditioning,
            conditioning_attention_mask=conditioning_attention_mask,
            quantization=self._quantization,
            offload_mode=self._offload_mode,
        )

        logger.info(
            "Evaluation composition: mode=%s, audio_checkpoint=%s, motion_track_ic_lora=%s",
            self.mode.value,
            self.checkpoint_info.path,
            self._loras[0].path if self._loras else "none",
        )

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        (context,) = self.prompt_encoder(
            [prompt],
            enhance_first_prompt=enhance_prompt,
            enhance_prompt_image=images[0][0] if images else None,
            enhance_prompt_seed=seed,
        )
        video_context, audio_context = context.video_encoding, context.audio_encoding

        stage_1_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width // 2,
            height=height // 2,
            fps=frame_rate,
        )
        stage_1_conditionings = self.image_conditioner(
            lambda encoder: self._create_conditionings(
                images=images,
                video_conditioning=video_conditioning,
                height=stage_1_shape.height,
                width=stage_1_shape.width,
                video_encoder=encoder,
                num_frames=num_frames,
                conditioning_attention_strength=conditioning_attention_strength,
                conditioning_attention_mask=conditioning_attention_mask,
            )
        )

        video_state, audio_state = self.stage_1(
            denoiser=SimpleDenoiser(video_context, audio_context),
            sigmas=stage_1_sigmas.to(dtype=torch.float32, device=self.device),
            noiser=noiser,
            width=stage_1_shape.width,
            height=stage_1_shape.height,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(context=video_context, conditionings=stage_1_conditionings),
            audio=ModalitySpec(context=audio_context),
        )
        if video_state is None or audio_state is None:
            raise RuntimeError("Stage 1 did not return both video and audio states")

        if skip_stage_2:
            logger.info("Skipping Stage 2 (--skip-stage-2 enabled)")
            return (
                self.video_decoder(video_state.latent, tiling_config, generator),
                self.audio_decoder(audio_state.latent),
            )

        upscaled_video_latent = self.upsampler(video_state.latent[:1])
        stage_2_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width,
            height=height,
            fps=frame_rate,
        )
        stage_2_conditionings = self.image_conditioner(
            lambda encoder: combined_image_conditionings(
                images=images,
                height=stage_2_shape.height,
                width=stage_2_shape.width,
                video_encoder=encoder,
                dtype=self.dtype,
                device=self.device,
            )
        )
        stage_2_sigmas = stage_2_sigmas.to(dtype=torch.float32, device=self.device)
        video_state, audio_state = self.stage_2(
            denoiser=SimpleDenoiser(video_context, audio_context),
            sigmas=stage_2_sigmas,
            noiser=noiser,
            width=width,
            height=height,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(
                context=video_context,
                conditionings=stage_2_conditionings,
                noise_scale=stage_2_sigmas[0].item(),
                initial_latent=upscaled_video_latent,
            ),
            audio=ModalitySpec(
                context=audio_context,
                noise_scale=stage_2_sigmas[0].item(),
                initial_latent=audio_state.latent,
            ),
        )
        if video_state is None or audio_state is None:
            raise RuntimeError("Stage 2 did not return both video and audio states")

        return (
            self.video_decoder(video_state.latent, tiling_config, generator),
            self.audio_decoder(audio_state.latent),
        )

    def _create_conditionings(
        self,
        images: list[ImageConditioningInput],
        video_conditioning: Sequence[tuple[str, float]],
        height: int,
        width: int,
        num_frames: int,
        video_encoder: VideoEncoder,
        conditioning_attention_strength: float,
        conditioning_attention_mask: torch.Tensor | None,
    ) -> list[ConditioningItem]:
        conditionings = combined_image_conditionings(
            images=images,
            height=height,
            width=width,
            video_encoder=video_encoder,
            dtype=self.dtype,
            device=self.device,
        )
        if self.mode == EvaluationMode.AUDIO_FULL_IC_LORA:
            append_ic_lora_reference_video_conditionings(
                conditionings,
                list(video_conditioning),
                height=height,
                width=width,
                num_frames=num_frames,
                video_encoder=video_encoder,
                dtype=self.dtype,
                device=self.device,
                reference_downscale_factor=self.reference_downscale_factor,
                reference_temporal_scale_factor=self.reference_temporal_scale_factor,
                conditioning_attention_strength=conditioning_attention_strength,
                conditioning_attention_mask=conditioning_attention_mask,
                tiling_config=None,
            )
            logger.info("Added %d Motion Track IC-LoRA reference video(s)", len(video_conditioning))
        return conditionings


# Backward-compatible import name for callers that used the original file.
ICLoraPipeline = AudioFullEvaluationPipeline


def _resolve_cli_params() -> object:
    """Support both current and slightly older LTX pipeline argument helpers."""

    if hasattr(pipeline_args, "resolve_cli_params"):
        return pipeline_args.resolve_cli_params(distilled=True)
    checkpoint_path = pipeline_args.detect_checkpoint_path(distilled=True)
    return detect_params(checkpoint_path)


@torch.inference_mode()
def main() -> None:
    logging.basicConfig(level=logging.INFO)
    params = _resolve_cli_params()
    parser = pipeline_args.default_2_stage_distilled_arg_parser(params=params)
    parser.add_argument(
        "--evaluation-mode",
        choices=[mode.value for mode in EvaluationMode],
        required=True,
        help=(
            "audio_full_ic_lora: trained audio branch plus the exact frozen Motion Track IC-LoRA; "
            "audio_full: trained audio branch with no LoRA and no reference-video conditioning."
        ),
    )
    parser.add_argument(
        "--audio-full-checkpoint",
        required=True,
        help=(
            "Path to audio_full_weights_step_*.safetensors, or a directory containing checkpoints "
            "from which the highest step is selected."
        ),
    )
    parser.add_argument(
        "--video-conditioning",
        action=VideoConditioningAction,
        nargs=2,
        metavar=("PATH", "STRENGTH"),
        default=None,
        help="Motion Track reference video and conditioning strength; required only for audio_full_ic_lora.",
    )
    parser.add_argument(
        "--conditioning-attention-mask",
        action=VideoMaskConditioningAction,
        nargs=2,
        metavar=("MASK_PATH", "STRENGTH"),
        default=None,
        help="Optional IC-LoRA spatial mask; valid only in audio_full_ic_lora mode.",
    )
    parser.add_argument(
        "--skip-stage-2",
        action="store_true",
        help="Skip Stage 2 upsampling/refinement and decode the half-resolution Stage 1 output.",
    )
    args = parser.parse_args()

    checkpoint_info = resolve_audio_full_checkpoint(args.audio_full_checkpoint)
    loras = tuple(args.lora or ())
    video_conditioning = tuple(args.video_conditioning or ())

    conditioning_attention_mask = None
    conditioning_attention_strength = 1.0
    if args.conditioning_attention_mask is not None:
        mask_path, mask_strength = args.conditioning_attention_mask
        conditioning_attention_strength = float(mask_strength)
        conditioning_attention_mask = _load_mask_video(
            mask_path=mask_path,
            height=args.height // 2,
            width=args.width // 2,
            num_frames=args.num_frames,
        )

    mode = EvaluationMode(args.evaluation_mode)
    _validate_mode_configuration(
        mode=mode,
        checkpoint_info=checkpoint_info,
        loras=loras,
        video_conditioning=video_conditioning,
        conditioning_attention_mask=conditioning_attention_mask,
        quantization=args.quantization,
        offload_mode=args.offload_mode,
    )

    pipeline = AudioFullEvaluationPipeline(
        distilled_checkpoint_path=args.distilled_checkpoint_path,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=args.gemma_root,
        audio_full_checkpoint_path=checkpoint_info.path,
        evaluation_mode=mode,
        loras=loras,
        quantization=args.quantization,
        compilation_config=args.compile,
        offload_mode=args.offload_mode,
    )

    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)
    video, audio = pipeline(
        prompt=args.prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        images=args.images,
        video_conditioning=video_conditioning,
        tiling_config=tiling_config,
        conditioning_attention_strength=conditioning_attention_strength,
        skip_stage_2=args.skip_stage_2,
        conditioning_attention_mask=conditioning_attention_mask,
    )
    encode_video(
        video=video,
        fps=args.frame_rate,
        audio=audio,
        output_path=args.output_path,
        video_chunks_number=video_chunks_number,
    )


def _load_mask_video(
    mask_path: str,
    height: int,
    width: int,
    num_frames: int,
) -> torch.Tensor:
    """Load, resize and normalize a grayscale attention-mask video."""

    device = get_device()
    frame_iterator = decode_video_by_frame(path=mask_path, frame_cap=num_frames, device=device)
    mask_video = video_preprocess(frame_iterator, height, width, torch.bfloat16, device)
    mask = mask_video.mean(dim=1, keepdim=True)
    return ((mask + 1.0) / 2.0).clamp(0.0, 1.0)


if __name__ == "__main__":
    main()
