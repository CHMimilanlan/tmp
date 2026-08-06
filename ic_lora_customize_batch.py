import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch
from einops import rearrange

from ltx_core.components.noisers import GaussianNoiser
from ltx_core.conditioning import ConditioningItem
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.loader.registry import Registry
from ltx_core.model.transformer.compiling import CompilationConfig
from ltx_core.model.video_vae import TilingConfig, VideoEncoder, get_video_chunks_number
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import Audio, AudioLatentShape, VideoPixelShape
from ltx_pipelines.iclora_utils import (
    append_ic_lora_reference_video_conditionings,
    read_lora_reference_downscale_factor,
    read_lora_reference_temporal_scale_factor,
)
from ltx_pipelines.utils.args import (
    ImageConditioningInput,
    VideoConditioningAction,
    VideoMaskConditioningAction,
    default_2_stage_distilled_arg_parser,
    detect_checkpoint_path,
)
from ltx_pipelines.utils.blocks import (
    AudioDecoder,
    DiffusionStage,
    ImageConditioner,
    PromptEncoder,
    VideoDecoder,
    VideoUpsampler,
)
from ltx_pipelines.utils.constants import (
    DISTILLED_SIGMAS,
    STAGE_2_DISTILLED_SIGMAS,
    detect_params,
)
from ltx_pipelines.utils.denoisers import SimpleDenoiser
from ltx_pipelines.utils.helpers import assert_resolution, combined_image_conditionings, get_device
from ltx_pipelines.utils.media_io import decode_video_by_frame, encode_video, resize_and_center_crop, video_preprocess
from ltx_pipelines.utils.types import ModalitySpec, OffloadMode


class ICLoraPipeline:
    """
    Two-stage video generation pipeline with In-Context (IC) LoRA support.
    Allows conditioning the generated video on control signals such as depth maps,
    human pose, or image edges via the video_conditioning parameter.
    The specific IC-LoRA model should be provided via the loras parameter.
    Stage 1 generates video at half of the target resolution, then Stage 2 upsamples
    by 2x and refines with additional denoising steps for higher quality output.
    Both stages use distilled models for efficiency.
    """

    def __init__(
        self,
        distilled_checkpoint_path: str,
        spatial_upsampler_path: str,
        gemma_root: str,
        loras: list[LoraPathStrengthAndSDOps],
        device: torch.device | None = None,
        quantization: QuantizationPolicy | None = None,
        registry: Registry | None = None,
        compilation_config: CompilationConfig | None = None,
        offload_mode: OffloadMode = OffloadMode.NONE,
        spatial_track_encoder_path: str | Path | None = None,
        track_prope_path: str | Path | None = None,
        track_summary_path: str | Path | None = None,
        simple_summary_path: str | Path | None = None,
        summary_source_mode: str = "reference_guided_target",
    ):
        self.device = device or get_device()
        self.dtype = torch.bfloat16

        self.prompt_encoder = PromptEncoder(
            distilled_checkpoint_path,
            gemma_root,
            self.dtype,
            self.device,
            registry=registry,
            offload_mode=offload_mode,
        )
        
        self.image_conditioner = ImageConditioner(distilled_checkpoint_path, self.dtype, self.device, registry=registry)
        self.stage_1 = DiffusionStage(
            distilled_checkpoint_path,
            self.dtype,
            self.device,
            loras=tuple(loras),
            quantization=quantization,
            registry=registry,
            compilation_config=compilation_config,
            offload_mode=offload_mode,
        )
        self.stage_2 = DiffusionStage(
            distilled_checkpoint_path,
            self.dtype,
            self.device,
            loras=(),
            quantization=quantization,
            registry=registry,
            compilation_config=compilation_config,
            offload_mode=offload_mode,
        )
        self.upsampler = VideoUpsampler(
            distilled_checkpoint_path, spatial_upsampler_path, self.dtype, self.device, registry=registry
        )
        self.video_decoder = VideoDecoder(distilled_checkpoint_path, self.dtype, self.device, registry=registry)
        self.audio_decoder = AudioDecoder(distilled_checkpoint_path, self.dtype, self.device, registry=registry)

        self.use_spatial_track_encoder = spatial_track_encoder_path is not None
        self.use_track_prope = track_prope_path is not None
        self.use_track_summary = track_summary_path is not None
        self.use_simple_summary = simple_summary_path is not None
        self.summary_source_mode = summary_source_mode
        self._configure_track_modules(
            spatial_track_encoder_path=spatial_track_encoder_path,
            track_prope_path=track_prope_path,
            track_summary_path=track_summary_path,
            simple_summary_path=simple_summary_path,
            summary_source_mode=summary_source_mode,
        )

        # Read reference scale factors from LoRA metadata.
        # IC-LoRAs trained with scaled reference videos store these factors
        # so inference can resize/subsample reference videos to match training conditions.
        self.reference_downscale_factor = 2
        self.reference_temporal_scale_factor = 1
        for lora in loras:
            scale = read_lora_reference_downscale_factor(lora.path)
            if scale != 1:
                if self.reference_downscale_factor not in (1, scale):
                    raise ValueError(
                        f"Conflicting reference_downscale_factor values in LoRAs: "
                        f"already have {self.reference_downscale_factor}, but {lora.path} "
                        f"specifies {scale}. Cannot combine LoRAs with different reference scales."
                    )
                self.reference_downscale_factor = scale
            temporal = read_lora_reference_temporal_scale_factor(lora.path)
            if temporal != 1:
                if self.reference_temporal_scale_factor not in (1, temporal):
                    raise ValueError(
                        f"Conflicting reference_temporal_scale_factor values in LoRAs: "
                        f"already have {self.reference_temporal_scale_factor}, but {lora.path} "
                        f"specifies {temporal}. Cannot combine LoRAs with different temporal scales."
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
        video_conditioning: list[tuple[str, float]],
        enhance_prompt: bool = False,
        tiling_config: TilingConfig | None = None,
        conditioning_attention_strength: float = 1.0,
        skip_stage_2: bool = False,
        conditioning_attention_mask: torch.Tensor | None = None,
        stage_1_sigmas: torch.Tensor = DISTILLED_SIGMAS,
        stage_2_sigmas: torch.Tensor = STAGE_2_DISTILLED_SIGMAS,
        track: str | Path | None = None,
    ) -> tuple[Iterator[torch.Tensor], Audio]:
        """
        Generate video with IC-LoRA conditioning.
        Args:
            prompt: Text prompt for video generation.
            seed: Random seed for reproducibility.
            height: Output video height in pixels (must be divisible by 64).
            width: Output video width in pixels (must be divisible by 64).
            num_frames: Number of frames to generate.
            frame_rate: Output video frame rate.
            images: List of (path, frame_idx, strength) tuples for image conditioning.
            video_conditioning: List of (path, strength) tuples for IC-LoRA video conditioning.
            enhance_prompt: Whether to enhance the prompt using the text encoder.
            tiling_config: Optional tiling configuration for VAE decoding.
            conditioning_attention_strength: Scale factor for IC-LoRA conditioning attention.
                Controls how strongly the conditioning video influences the output.
                0.0 = ignore conditioning, 1.0 = full conditioning influence. Default 1.0.
                When conditioning_attention_mask is provided, the mask is multiplied by
                this strength before being passed to the conditioning items.
            skip_stage_2: If True, skip Stage 2 upsampling and refinement. Output will be
                at half resolution (height//2, width//2). Default is False.
            conditioning_attention_mask: Optional pixel-space attention mask with the same
                spatial-temporal dimensions as the input reference video. Shape should be
                (B, 1, F, H, W) or (1, 1, F, H, W) where F, H, W match the reference
                video's pixel dimensions. Values in [0, 1].
                The mask is downsampled to latent space using VAE scale factors (with
                causal temporal handling for the first frame), then multiplied by
                conditioning_attention_strength.
                When None (default): scalar conditioning_attention_strength is used
                directly.
        Returns:
            Tuple of (video_iterator, audio_tensor).
        """
        assert_resolution(height=height, width=width, is_two_stage=True)
        if not (0.0 <= conditioning_attention_strength <= 1.0):
            raise ValueError(
                f"conditioning_attention_strength must be in [0.0, 1.0], got {conditioning_attention_strength}"
            )

        if (
            self.use_track_prope
            or (self.use_track_summary and self.summary_source_mode in {"track_xy", "hybrid"})
        ) and track is None:
            raise ValueError("The selected track module configuration requires a track for every sample")
        track_xy, track_valid = self._prepare_track(track, num_frames, frame_rate)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)

        (ctx_p,) = self.prompt_encoder(
            [prompt],
            enhance_first_prompt=enhance_prompt,
            enhance_prompt_image=images[0][0] if len(images) > 0 else None,
            enhance_prompt_seed=seed,
        )
        video_context, audio_context = ctx_p.video_encoding, ctx_p.audio_encoding

        # Stage 1: Initial low resolution video generation.
        stage_1_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width // 2,
            height=height // 2,
            fps=frame_rate,
        )

        # Encode conditionings using the video encoder block
        stage_1_conditionings = self.image_conditioner(
            lambda enc: self._create_conditionings(
                images=images,
                video_conditioning=video_conditioning,
                height=stage_1_output_shape.height,
                width=stage_1_output_shape.width,
                video_encoder=enc,
                num_frames=num_frames,
                conditioning_attention_strength=conditioning_attention_strength,
                conditioning_attention_mask=conditioning_attention_mask,
            )
        )

        stage_1_sigmas = stage_1_sigmas.to(dtype=torch.float32, device=self.device)
        video_state, audio_state = self.stage_1(
            denoiser=SimpleDenoiser(
                video_context, audio_context, track_xy=track_xy, track_valid=track_valid
            ),
            sigmas=stage_1_sigmas,

            noiser=noiser,
            width=stage_1_output_shape.width,
            height=stage_1_output_shape.height,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(
                context=video_context,
                conditionings=stage_1_conditionings,
            ),
            audio=self._audio_spec(audio_context, track_xy, track_valid),
        )

        if skip_stage_2:
            # Skip Stage 2: Decode directly from Stage 1 output at half resolution
            logging.info("[IC-LoRA] Skipping Stage 2 (--skip-stage-2 enabled)")
            decoded_video = self.video_decoder(video_state.latent, tiling_config, generator)
            decoded_audio = self.audio_decoder(audio_state.latent)
            return decoded_video, decoded_audio

        # Stage 2: Upsample and refine the video at higher resolution with distilled LORA.
        upscaled_video_latent = self.upsampler(video_state.latent[:1])

        stage_2_sigmas = stage_2_sigmas.to(dtype=torch.float32, device=self.device)
        stage_2_output_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)
        stage_2_conditionings = self.image_conditioner(
            lambda enc: combined_image_conditionings(
                images=images,
                height=stage_2_output_shape.height,
                width=stage_2_output_shape.width,
                video_encoder=enc,
                dtype=self.dtype,
                device=self.device,
            )
        )

        video_state, audio_state = self.stage_2(
            denoiser=SimpleDenoiser(
                video_context, audio_context, track_xy=track_xy, track_valid=track_valid
            ),
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
            audio=self._audio_spec(
                context=audio_context,
                noise_scale=stage_2_sigmas[0].item(),
                initial_latent=audio_state.latent,
                track_xy=track_xy,
                track_valid=track_valid,
            ),
        )

        decoded_video = self.video_decoder(video_state.latent, tiling_config, generator)
        decoded_audio = self.audio_decoder(audio_state.latent)
        return decoded_video, decoded_audio

    def _configure_track_modules(
        self,
        *,
        spatial_track_encoder_path: str | Path | None,
        track_prope_path: str | Path | None,
        track_summary_path: str | Path | None,
        simple_summary_path: str | Path | None,
        summary_source_mode: str,
    ) -> None:
        """Install hooks that configure transformers when a stage creates them.

        ``DiffusionStage`` owns a lazy transformer context.  There is no model
        instance during pipeline construction; it only exists inside
        ``with stage._transformer_ctx(...)``.  Wrapping that context therefore
        avoids trying to discover a transformer before it has been created and
        also works with offloading modes that create a new instance later.
        """
        if not (
            spatial_track_encoder_path
            or track_prope_path
            or track_summary_path
            or simple_summary_path
        ):
            return

        spatial_path = self._validate_weights_path(spatial_track_encoder_path)
        prope_path = self._validate_weights_path(track_prope_path)
        summary_path = self._validate_weights_path(track_summary_path)
        simple_summary_path = self._validate_weights_path(simple_summary_path)
        self._wrap_transformer_context(
            self.stage_1,
            spatial_track_encoder_path=spatial_path,
            track_prope_path=prope_path,
            track_summary_path=summary_path,
            simple_summary_path=simple_summary_path,
            summary_source_mode=summary_source_mode,
        )
        # Stage 2 has no reference-video Spatial Track Encoder conditioning,
        # but Track-PRoPE must be present for its audio denoising blocks.
        if prope_path is not None:
            self._wrap_transformer_context(
                self.stage_2,
                spatial_track_encoder_path=None,
                track_prope_path=prope_path,
                track_summary_path=None,
                simple_summary_path=None,
                summary_source_mode=summary_source_mode,
            )

    @staticmethod
    def _validate_weights_path(path: str | Path | None) -> Path | None:
        if path is None:
            return None
        resolved = Path(path).expanduser()
        if not resolved.is_file():
            raise FileNotFoundError(f"Track module weights do not exist: {resolved}")
        return resolved

    def _wrap_transformer_context(
        self,
        stage: DiffusionStage,
        *,
        spatial_track_encoder_path: Path | None,
        track_prope_path: Path | None,
        track_summary_path: Path | None,
        simple_summary_path: Path | None,
        summary_source_mode: str,
    ) -> None:
        original_transformer_ctx = stage._transformer_ctx

        transformer_args = {
            "use_track_prope": track_prope_path is not None,
            "use_ic_lora_summary":track_summary_path is not None,
            "use_simple_summary": simple_summary_path is not None,
            "use_audio_summary":False,
        }
        
        @contextmanager
        def track_aware_transformer_ctx(*args: Any, **kwargs: Any):
            kwargs.update(transformer_args)
            with original_transformer_ctx(*args, **kwargs) as transformer:
                self._configure_transformer_instance(
                    transformer,
                    spatial_track_encoder_path=spatial_track_encoder_path,
                    track_prope_path=track_prope_path,
                    track_summary_path=track_summary_path,
                    simple_summary_path=simple_summary_path,
                    summary_source_mode=summary_source_mode,
                )
                yield transformer

        stage._transformer_ctx = track_aware_transformer_ctx

    def _configure_transformer_instance(
        self,
        transformer: torch.nn.Module,
        *,
        spatial_track_encoder_path: Path | None,
        track_prope_path: Path | None,
        track_summary_path: Path | None,
        simple_summary_path: Path | None,
        summary_source_mode: str,
    ) -> None:
        signature = (
            spatial_track_encoder_path, track_prope_path,
            track_summary_path, simple_summary_path, summary_source_mode,
        )
        if getattr(transformer, "_ic_track_modules_signature", None) == signature:
            return

        # ``_transformer_ctx`` yields an ``X0Model`` wrapper whose real LTX
        # transformer (holding ``transformer_blocks`` / track init methods) lives
        # under ``.velocity_model``. Unwrap so we configure the actual model.
        core = getattr(transformer, "velocity_model", transformer)

        if track_prope_path is not None:
            blocks = getattr(core, "transformer_blocks", None)
            if blocks is None:
                raise RuntimeError("The lazily loaded transformer has no transformer_blocks")
            for block in blocks:
                block.use_track_prope = True
                block.initialize_track_prope(device=self.device, dtype=self.dtype)
            self._load_module_weights(transformer, track_prope_path, "track_prope")

        if track_summary_path is not None:
            blocks = getattr(core, "transformer_blocks", None)
            if blocks is None:
                raise RuntimeError("The lazily loaded transformer has no transformer_blocks")
            use_audio_summary = self._summary_uses_audio_attention(track_summary_path)
            for block in blocks:
                if not hasattr(block, "initialize_track_summary"):
                    raise RuntimeError(
                        "The loaded transformer block does not implement initialize_track_summary()"
                    )
                block.use_ic_lora_summary = True
                block.use_audio_to_summary_attention = use_audio_summary
                block.initialize_track_summary(device=self.device, dtype=self.dtype)
                block.set_summary_source_mode(summary_source_mode)
            self._load_module_weights(transformer, track_summary_path, "track_summary")

        if simple_summary_path is not None:
            blocks = getattr(core, "transformer_blocks", None)
            if blocks is None:
                raise RuntimeError("The lazily loaded transformer has no transformer_blocks")
            for block in blocks:
                if not hasattr(block, "initialize_simple_summary"):
                    raise RuntimeError(
                        "The loaded transformer block does not implement initialize_simple_summary()"
                    )
                block.use_simple_summary = True
                block.initialize_simple_summary(device=self.device, dtype=self.dtype)
            self._load_module_weights(transformer, simple_summary_path, "simple_summary")

        if spatial_track_encoder_path is not None:
            from ltx_core.model.transformer.spatial_track_encoder import SpatialTrackEncoderConfig

            core.initialize_spatial_track_modules(
                encoder_type="simple",
                cfg=SpatialTrackEncoderConfig(
                    dim=128, video_t=16, video_h=8, video_w=12, audio_t=122,
                    num_heads=8, dropout=0.0, encoder_depth=4, decoder_depth=2,
                ),
                device=self.device,
                dtype=self.dtype,
            )
            self._load_module_weights(transformer, spatial_track_encoder_path, "spatial_track_encoder")
        transformer._ic_track_modules_signature = signature

    @staticmethod
    def _summary_uses_audio_attention(path: Path) -> bool:
        """Infer which mutually exclusive Summary audio adapter was trained."""
        if path.suffix == ".safetensors":
            from safetensors import safe_open

            with safe_open(str(path), framework="pt", device="cpu") as handle:
                keys = list(handle.keys())
        else:
            state = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            if not isinstance(state, dict):
                raise TypeError(f"Expected a state dict in {path}")
            keys = list(state)
        has_attention = any(".audio_to_summary_attn." in key for key in keys)
        has_projection = any(".summary_to_audio." in key for key in keys)
        if has_attention == has_projection:
            raise ValueError(
                f"Track-Summary checkpoint {path} must contain exactly one of "
                "audio_to_summary_attn or summary_to_audio weights"
            )
        return has_attention

    @staticmethod
    def _load_module_weights(model: torch.nn.Module, path: str | Path, marker: str) -> None:
        """Load a track-module checkpoint into ``model`` (an ``X0Model``).

        The trainer saves each module with a stable, wrapper-free key layout:

        * ``track_prope``           -> ``transformer_blocks.{i}.audio_track_prope.<param>``
          (the block index is kept so every block gets its own tensors).
        * ``spatial_track_encoder`` -> a *relative* key (e.g. ``video_pos``,
          ``track_encoder.layers.0...``) with the ``spatial_track_encoder.``
          prefix already stripped.
        * ``track_summary``         -> ``transformer_blocks.{i}.<summary module>.<param>``.
        * ``simple_summary``        -> ``transformer_blocks.{i}.simple_summary.<param>``
          plus the block's ``simple_summary_scale`` parameter.

        The live model, however, is an ``X0Model`` whose real LTX transformer
        lives under ``velocity_model``, so its state-dict keys look like
        ``velocity_model.transformer_blocks.{i}.audio_track_prope.<param>`` and
        ``velocity_model.spatial_track_encoder.<param>``.

        We therefore turn each checkpoint key into a stable *anchor suffix* and
        match it against the model keys with ``endswith`` (uniquely).
        """
        path = Path(path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"{marker} weights do not exist: {path}")
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file

            state = load_file(str(path), device="cpu")
        else:
            state = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise TypeError(f"Expected a state dict in {path}")

        if marker == "track_prope":
            block_marker = "transformer_blocks."

            def to_anchor(key: str) -> str | None:
                position = key.find(block_marker)
                if position < 0 or ".audio_track_prope." not in key:
                    return None
                # e.g. "transformer_blocks.0.audio_track_prope.q_proj.weight"
                return key[position:]
        elif marker == "track_summary":
            block_marker = "transformer_blocks."
            summary_markers = (
                ".ic_lora_summarizer.", ".summary_to_audio.",
                ".audio_to_summary_attn.", ".summary_scale",
            )

            def to_anchor(key: str) -> str | None:
                position = key.find(block_marker)
                if position < 0 or not any(item in key for item in summary_markers):
                    return None
                return key[position:]
        elif marker == "simple_summary":
            block_marker = "transformer_blocks."
            simple_summary_markers = (
                ".simple_summary.", ".simple_summary_scale",
            )

            def to_anchor(key: str) -> str | None:
                position = key.find(block_marker)
                if position < 0 or not any(
                    item in key for item in simple_summary_markers
                ):
                    return None
                return key[position:]
        elif marker == "spatial_track_encoder":
            module_prefix = "spatial_track_encoder."

            def to_anchor(key: str) -> str | None:
                position = key.find(module_prefix)
                if position >= 0:
                    # Checkpoint key already carries the module prefix.
                    return key[position:]
                # Relative checkpoint key: prepend the prefix to build the anchor.
                return module_prefix + key
        else:
            raise ValueError(f"Unknown module marker: {marker}")

        model_keys = set(model.state_dict())
        selected: dict[str, torch.Tensor] = {}
        unmatched: list[str] = []
        for key, value in state.items():
            if not isinstance(value, torch.Tensor):
                continue
            anchor = to_anchor(key)
            if anchor is None:
                continue
            matches = [candidate for candidate in model_keys if candidate.endswith(anchor)]
            if len(matches) == 1:
                selected[matches[0]] = value
            else:
                unmatched.append(f"{key} -> {anchor} ({len(matches)} candidate matches)")

        if not selected:
            raise ValueError(
                f"No {marker} parameters from {path} match the loaded transformer. "
                f"Unmatched examples: {unmatched[:5]}"
            )
        if unmatched:
            logging.warning(
                "%d %s tensor(s) from %s could not be uniquely matched (e.g. %s)",
                len(unmatched), marker, path, unmatched[:3],
            )

        incompatible = model.load_state_dict(selected, strict=False)
        logging.info("Loaded %d %s tensors from %s", len(selected), marker, path)
        if incompatible.unexpected_keys:
            logging.warning(
                "%d unexpected %s key(s) while loading %s (e.g. %s)",
                len(incompatible.unexpected_keys), marker, path,
                incompatible.unexpected_keys[:3],
            )

    def _prepare_track(
        self, path: str | Path | None, num_frames: int, frame_rate: float
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if path is None:
            return None, None
        raw = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)
        xy = raw if isinstance(raw, torch.Tensor) else raw.get("track_xy")
        valid = None if isinstance(raw, torch.Tensor) else raw.get("track_valid")
        if not isinstance(xy, torch.Tensor):
            raise ValueError(f"Track {path} must be a tensor or contain track_xy")
        if xy.ndim == 3 and xy.shape[0] == 1:
            xy = xy[0]
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError(f"Track {path} must have shape [T, 2], got {tuple(xy.shape)}")
        valid = torch.ones(xy.shape[0], dtype=torch.bool) if valid is None else valid.squeeze().bool()
        if valid.shape != (xy.shape[0],):
            raise ValueError(
                f"Track {path} track_valid must have shape {(xy.shape[0],)}, got {tuple(valid.shape)}"
            )
        valid = valid & torch.isfinite(xy).all(-1)
        token_count = AudioLatentShape.from_duration(1, num_frames / frame_rate).token_count()
        xy = torch.nn.functional.interpolate(
            torch.nan_to_num(xy.float()).T[None], size=token_count, mode="linear", align_corners=True
        ).transpose(1, 2)
        valid = torch.nn.functional.interpolate(
            valid.float()[None, None], size=token_count, mode="nearest"
        ).squeeze(1).bool()
        return xy.to(self.device, self.dtype), valid.to(self.device)

    @staticmethod
    def _audio_spec(context: torch.Tensor, track_xy=None, track_valid=None, **kwargs: Any) -> ModalitySpec:
        values = {"context": context, **kwargs}
        supported = {field.name for field in fields(ModalitySpec)}

        if track_xy is not None:
            if not {"track_xy", "track_valid"}.issubset(supported):
                raise RuntimeError(
                    "This LTX installation's ModalitySpec does not expose track_xy/track_valid; "
                    "install the Track-PRoPE-enabled LTX core used by pipeline/transformer.py"
                )
            values.update(track_xy=track_xy, track_valid=track_valid)
        return ModalitySpec(**values)

    def _create_conditionings(
        self,
        images: list[ImageConditioningInput],
        video_conditioning: list[tuple[str, float]],
        height: int,
        width: int,
        num_frames: int,
        video_encoder: VideoEncoder,
        conditioning_attention_strength: float = 1.0,
        conditioning_attention_mask: torch.Tensor | None = None,
    ) -> list[ConditioningItem]:
        """
        Create conditioning items for video generation.
        Args:
            conditioning_attention_strength: Scalar attention weight in [0, 1].
                If conditioning_attention_mask is also provided, the downsampled mask
                is multiplied by this strength. Otherwise this scalar is passed
                directly as the attention mask.
            conditioning_attention_mask: Optional pixel-space attention mask with shape
                (B, 1, F_pixel, H_pixel, W_pixel) matching the reference video's
                pixel dimensions. Downsampled to latent space with causal temporal
                handling, then multiplied by conditioning_attention_strength.
        Returns:
            List of conditioning items. IC-LoRA conditionings are appended last.
        """
        conditionings = combined_image_conditionings(
            images=images,
            height=height,
            width=width,
            video_encoder=video_encoder,
            dtype=self.dtype,
            device=self.device,
        )

        append_ic_lora_reference_video_conditionings(
            conditionings,
            video_conditioning,
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

        if video_conditioning:
            logging.info("[IC-LoRA] Added %d video conditioning(s)", len(video_conditioning))

        return conditionings


@torch.inference_mode()
def main() -> None:
    logging.basicConfig(level=logging.INFO)
    checkpoint_path = detect_checkpoint_path(distilled=True)
    params = detect_params(checkpoint_path)
    parser = default_2_stage_distilled_arg_parser(params=params)
    # The upstream single-sample parser marks --prompt as required. Batch
    # manifests provide a prompt per sample, so argparse must not reject the
    # command before we have a chance to inspect --batch-json. Single-sample
    # mode is validated explicitly after parsing below.
    prompt_action = next(
        (action for action in parser._actions if action.dest == "prompt"),
        None,
    )
    if prompt_action is None:
        raise RuntimeError("The LTX argument parser does not define --prompt")
    prompt_action.required = False
    parser.add_argument(
        "--video-conditioning",
        action=VideoConditioningAction,
        nargs=2,
        metavar=("PATH", "STRENGTH"),
        required=False,
    )
    parser.add_argument(
        "--conditioning-attention-mask",
        action=VideoMaskConditioningAction,
        nargs=2,
        metavar=("MASK_PATH", "STRENGTH"),
        default=None,
        help=(
            "Optional spatial attention mask: path to a grayscale mask video and "
            "attention strength. The mask video pixel values in [0,1] control "
            "per-region conditioning attention strength. The strength scalar is "
            "multiplied with the spatial mask. "
            "0.0 = ignore IC-LoRA conditioning, 1.0 = full conditioning influence. "
            "When not provided, full conditioning strength (1.0) is used. "
            "Example: --conditioning-attention-mask path/to/mask.mp4 0.5"
        ),
    )
    parser.add_argument(
        "--batch-json",
        type=Path,
        help=("JSON array (or an object with a 'samples' array). Each sample must contain "
              "prompt, reference_video and track; output_path and per-sample overrides are optional."),
    )
    parser.add_argument("--spatial-track-encoder-weights", type=Path)
    parser.add_argument("--track-prope-weights", type=Path)
    parser.add_argument(
        "--track-summary-weights",
        type=Path,
        help="Enable Track-Summary and load its trained module checkpoint.",
    )
    parser.add_argument(
        "--simple-summary-weights",
        type=Path,
        help="Enable Simple-Summary and load its trained module checkpoint.",
    )
    parser.add_argument(
        "--summary-source-mode",
        choices=("track_xy", "reference_tokens", "reference_guided_target", "hybrid"),
        default="reference_guided_target",
        help="Summary token source used by a loaded Track-Summary module.",
    )
    parser.add_argument(
        "--skip-stage-2",
        action="store_true",
        help=(
            "Skip Stage 2 upsampling and refinement. Output will be at half resolution "
            "(height//2, width//2). Useful for faster iteration or when GPU memory is limited."
        ),
    )
    parser.add_argument(
        "--decode-mode",
        action="store_true",
        help=(
            "After saving each generated video, decode its Mid/Side (mono diff) audio "
            "back into Left/Right stereo and save the result under a 'generate_decode' "
            "folder inside --output-path."
        ),
    )
    args = parser.parse_args()


    if args.batch_json is None:
        if not args.prompt:
            parser.error("--prompt is required when --batch-json is not used")
        if not args.video_conditioning:
            parser.error("--video-conditioning is required when --batch-json is not used")

    # Load mask video if provided via --conditioning-attention-mask
    conditioning_attention_mask = None
    conditioning_attention_strength = 1.0
    if args.conditioning_attention_mask is not None:
        mask_path, mask_strength = args.conditioning_attention_mask
        conditioning_attention_strength = mask_strength
        conditioning_attention_mask = _load_mask_video(
            mask_path=mask_path,
            height=args.height // 2,  # Stage 1 operates at half resolution
            width=args.width // 2,
            num_frames=args.num_frames,
        )

    pipeline = ICLoraPipeline(
        distilled_checkpoint_path=args.distilled_checkpoint_path,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=args.gemma_root,
        loras=tuple(args.lora) if args.lora else (),
        quantization=args.quantization,
        compilation_config=args.compile,
        offload_mode=args.offload_mode,
        spatial_track_encoder_path=args.spatial_track_encoder_weights,
        track_prope_path=args.track_prope_weights,
        track_summary_path=args.track_summary_weights,
        simple_summary_path=args.simple_summary_weights,
        summary_source_mode=args.summary_source_mode,
    )
    tiling_config = TilingConfig.default()
    result_directory = _result_directory(
        output_root=args.output_path,
        track_prope_weights=args.track_prope_weights,
        spatial_track_encoder_weights=args.spatial_track_encoder_weights,
        track_summary_weights=args.track_summary_weights,
        simple_summary_weights=args.simple_summary_weights,
        summary_source_mode=args.summary_source_mode if args.track_summary_weights is not None else None,
    )
    result_directory.mkdir(parents=True, exist_ok=True)
    # Separate the plain generated videos from the side-by-side concat videos
    # into dedicated sub-directories under the result directory.
    generated_directory = result_directory / "generated"
    concat_directory = result_directory / "concat"
    generated_directory.mkdir(parents=True, exist_ok=True)
    concat_directory.mkdir(parents=True, exist_ok=True)
    logging.info("Writing results to %s", result_directory)
    samples = _read_batch_samples(args.batch_json) if args.batch_json else [{
        "prompt": args.prompt,
        "reference_video": args.video_conditioning[0][0],
        "reference_strength": args.video_conditioning[0][1],
        "track": None,
        "output_path": None,
    }]
    if args.track_prope_weights and not args.batch_json:
        parser.error("Track-PRoPE needs per-sample tracks; use --batch-json")

    for index, sample in enumerate(samples):
        sample_frames = int(sample.get("num_frames", args.num_frames))
        sample_fps = float(sample.get("frame_rate", args.frame_rate))
        output_path = _sample_output_path(generated_directory, sample.get("output_path"), index)
        logging.info("Generating batch sample %d/%d -> %s", index + 1, len(samples), output_path)
        video, audio = pipeline(
            prompt=sample["prompt"], seed=int(sample.get("seed", args.seed)),
            height=int(sample.get("height", args.height)), width=int(sample.get("width", args.width)),
            num_frames=sample_frames, frame_rate=sample_fps, images=args.images,
            video_conditioning=[(sample["reference_video"], float(sample.get("reference_strength", 1.0)))],
            track=sample.get("track"), tiling_config=tiling_config,
            conditioning_attention_strength=conditioning_attention_strength,
            skip_stage_2=args.skip_stage_2,
            conditioning_attention_mask=conditioning_attention_mask,
        )
        # Materialise the generated frames once so we can both save the plain
        # video and build a side-by-side comparison against the reference video.
        # ``video`` is a single-consumption iterator, so it must be drained here.
        gen_frames = torch.cat([chunk.to("cpu") for chunk in video], dim=0)  # [F, H, W, C]

        encode_video(video=gen_frames, fps=sample_fps, audio=audio, output_path=output_path,
                     video_chunks_number=1)

        if args.decode_mode:
            decode_directory = result_directory / "generate_decode"
            _decode_mono_diff_to_stereo(
                video_path=output_path,
                output_dir=decode_directory,
            )


        reference_video = sample.get("reference_video")

        if reference_video:
            ref_frames = _load_reference_frames_for_concat(
                reference_video=reference_video,
                target_height=int(gen_frames.shape[1]),
                frame_cap=int(gen_frames.shape[0]),
                device=get_device(),
            )
            if ref_frames is not None:
                n = min(int(gen_frames.shape[0]), int(ref_frames.shape[0]))
                # Side-by-side: reference on the left, generated on the right (width dim=2).
                concat_frames = torch.cat([ref_frames[:n], gen_frames[:n]], dim=2)
                concat_path = _concat_output_path(output_path, concat_directory)
                encode_video(video=concat_frames, fps=sample_fps, audio=audio, output_path=concat_path,
                             video_chunks_number=1)


def _read_batch_samples(path: Path) -> list[dict[str, Any]]:
    """Validate and resolve paths in the batch manifest relative to the JSON file."""
    with path.expanduser().open(encoding="utf-8") as handle:
        document = json.load(handle)
    samples = document.get("samples") if isinstance(document, dict) else document
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"{path} must contain a non-empty samples array")
    base = path.expanduser().resolve().parent
    result = []
    for index, value in enumerate(samples):
        if not isinstance(value, dict):
            raise TypeError(f"Sample {index} in {path} must be an object")
        missing = {"prompt", "reference_video", "track"} - value.keys()
        if missing:
            raise ValueError(f"Sample {index} in {path} is missing: {', '.join(sorted(missing))}")
        sample = dict(value)
        for key in ("reference_video", "track"):
            if sample.get(key) and not Path(sample[key]).expanduser().is_absolute():
                sample[key] = str(base / sample[key])
        result.append(sample)
    return result


def _result_directory(
    *,
    output_root: str | Path,
    track_prope_weights: str | Path | None,
    spatial_track_encoder_weights: str | Path | None,
    track_summary_weights: str | Path | None,
    simple_summary_weights: str | Path | None,
    summary_source_mode: str | None = None,
) -> Path:
    """Build the module-specific result directory below ``output_root``.

    Checkpoint filenames are used verbatim, including their extension.
    When ``summary_source_mode`` is provided, it is appended to the child
    directory name so different summary source modes write to separate folders.
    If the resulting directory already exists, a ``_v1`` / ``_v2`` / ... suffix
    is appended so previous results are never overwritten.
    """
    names = []
    if track_prope_weights is not None:
        names.append(Path(track_prope_weights).expanduser().name)
    if spatial_track_encoder_weights is not None:
        names.append(Path(spatial_track_encoder_weights).expanduser().name)
    if track_summary_weights is not None:
        names.append(Path(track_summary_weights).expanduser().name)
    if simple_summary_weights is not None:
        names.append(Path(simple_summary_weights).expanduser().name)
    if summary_source_mode:
        names.append(summary_source_mode)
    child_name = "__".join(names) if names else "vanilla_results"
    base = Path(output_root).expanduser()
    candidate = base / child_name
    version = 1
    while candidate.exists():
        candidate = base / f"{child_name}_v{version}"
        version += 1
    return candidate


def _sample_output_path(result_directory: Path, requested_path: str | Path | None, index: int) -> str:
    """Return a unique output file inside the selected result directory."""
    if requested_path:
        filename = Path(requested_path).name
        if not filename:
            raise ValueError(f"Sample {index} output_path must include a file name")
    else:
        filename = f"sample_{index:04d}.mp4"
    return str(result_directory / filename)


def _concat_output_path(output_path: str, concat_directory: Path | None = None) -> str:
    """Derive the side-by-side comparison path by appending ``_concat`` to the stem.

    When ``concat_directory`` is provided, the resulting file is placed inside
    that directory instead of alongside the generated video.
    """
    path = Path(output_path)
    filename = f"{path.stem}_concat{path.suffix}"
    if concat_directory is not None:
        return str(concat_directory / filename)
    return str(path.with_name(filename))


def _load_reference_frames_for_concat(
    reference_video: str,
    target_height: int,
    frame_cap: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Load and format reference video frames for side-by-side concatenation.

    Decodes up to ``frame_cap`` frames, resizes each to ``target_height`` while
    preserving aspect ratio, and returns a tensor of shape ``[F, H, W, C]`` with
    float values in ``[0, 1]`` on CPU (matching the generated-video chunk format).
    Returns ``None`` when no frames could be decoded.
    """
    frames: list[torch.Tensor] = []
    for f in decode_video_by_frame(path=reference_video, frame_cap=frame_cap, device=device):
        # f: [1, H, W, C] uint8 [0, 255]. Compute target width from the source
        # aspect ratio so the reference is not distorted, rounded to an even value.
        src_h, src_w = int(f.shape[1]), int(f.shape[2])
        target_width = max(2, int(round(src_w * target_height / src_h)) // 2 * 2)
        # resize_and_center_crop returns [1, C, F, H, W]; convert to [F, H, W, C].
        resized = resize_and_center_crop(f.to(torch.float32), target_height, target_width)
        frame = rearrange(resized, "b c f h w -> (b f) h w c").div_(255.0).clamp_(0.0, 1.0)
        frames.append(frame.to("cpu"))
    if not frames:
        return None
    return torch.cat(frames, dim=0)


def _load_mask_video(
    mask_path: str,
    height: int,
    width: int,
    num_frames: int,
) -> torch.Tensor:
    """Load a mask video and return a pixel-space tensor of shape (1, 1, F, H, W).
    The mask video is loaded, resized to (height, width), converted to
    grayscale, and normalised to [0, 1].
    Args:
        mask_path: Path to the mask video file.
        height: Target height in pixels.
        width: Target width in pixels.
        num_frames: Maximum number of frames to load.
    Returns:
        Tensor of shape ``(1, 1, F, H, W)`` with values in ``[0, 1]``.
    """
    device = get_device()
    frame_gen = decode_video_by_frame(path=mask_path, frame_cap=num_frames, device=device)
    mask_video = video_preprocess(frame_gen, height, width, torch.bfloat16, device)
    # mask_video shape: (1, C, F, H, W) — take mean over channels for grayscale
    mask = mask_video.mean(dim=1, keepdim=True)  # (1, 1, F, H, W)
    # Normalise to [0, 1] — video_preprocess applies normalize_latent,
    # so undo that: values are in [-1, 1], remap to [0, 1]
    mask = (mask + 1.0) / 2.0
    return mask.clamp(0.0, 1.0)


def _decode_mono_diff_to_stereo(video_path: str | Path, output_dir: str | Path) -> Path:
    """Decode a video's Mid/Side (mono diff) audio track back into Left/Right stereo.

    Mirrors the ``decode`` path of ``wav_mid_side.py``:
        1. ffmpeg extracts the stereo (M/D) PCM audio track.
        2. Apply ``L = M + D`` / ``R = M - D`` to recover the original stereo.
        3. ffmpeg muxes the new audio track with the original video stream
           (``-c:v copy``) and writes the result under ``output_dir``.

    Args:
        video_path: Path to the generated video whose audio should be decoded.
        output_dir: Directory in which to save the decoded video. Created if absent.

    Returns:
        The path to the decoded video file.
    """
    import shutil
    import subprocess
    import tempfile

    import numpy as np
    import soundfile as sf

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for --decode-mode but was not found on PATH.")

    video_path = Path(video_path).expanduser()
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / video_path.name

    def _run_ffmpeg(cmd: list[str]) -> None:
        completed = subprocess.run(
            cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        if completed.returncode != 0:
            stderr = completed.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg failed (returncode={completed.returncode}): {stderr}")

    with tempfile.TemporaryDirectory(prefix="mono_diff_decode_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        extracted_wav = tmpdir_path / "audio_in.wav"
        decoded_wav = tmpdir_path / "audio_out.wav"

        # Step 1: extract the M/D audio as stereo PCM 16-bit WAV.
        _run_ffmpeg([
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(video_path),
            "-vn", "-ac", "2", "-acodec", "pcm_s16le",
            str(extracted_wav),
        ])

        if not extracted_wav.exists() or extracted_wav.stat().st_size == 0:
            raise RuntimeError(f"No usable audio track found in {video_path}")

        # Step 2: Mid/Side -> Left/Right (L = M + D, R = M - D).
        audio, sample_rate = sf.read(str(extracted_wav), dtype="float64", always_2d=True)
        if audio.ndim != 2 or audio.shape[1] != 2:
            raise ValueError(
                f"Expected stereo (M/D) audio with shape [frames, 2], got {audio.shape}"
            )
        mid = audio[:, 0]
        side = audio[:, 1]
        stereo = np.column_stack((mid + side, mid - side))
        if not np.isfinite(stereo).all():
            raise ValueError("Decoded audio contains NaN or Inf values")
        sf.write(str(decoded_wav), stereo, sample_rate, format="WAV", subtype="PCM_16")

        # Step 3: mux decoded audio with the original video stream.
        _run_ffmpeg([
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(video_path),
            "-i", str(decoded_wav),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "320k", "-shortest",
            str(output_path),
        ])

    logging.info("[decode-mode] Saved stereo-decoded video to %s", output_path)
    return output_path


if __name__ == "__main__":
    main()
