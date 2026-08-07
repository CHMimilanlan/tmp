import inspect
import logging
from collections.abc import Iterator, Mapping
from pathlib import Path

import torch
from safetensors.torch import load_file as load_safetensors_file

from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.conditioning import ConditioningItem
from ltx_core.loader import LoraPathStrengthAndSDOps
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
from ltx_pipelines.utils.args import (
    ImageConditioningInput,
    VideoConditioningAction,
    VideoMaskConditioningAction,
    default_2_stage_arg_parser,
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
from ltx_pipelines.utils.constants import STAGE_2_DISTILLED_SIGMAS, detect_params
from ltx_pipelines.utils.denoisers import GuidedDenoiser, SimpleDenoiser
from ltx_pipelines.utils.helpers import assert_resolution, combined_image_conditionings, get_device
from ltx_pipelines.utils.media_io import decode_video_by_frame, encode_video, video_preprocess
from ltx_pipelines.utils.types import ModalitySpec, OffloadMode


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


def _checkpoint_key(raw: str) -> str:
    """Normalize trainer/PEFT wrapper prefixes to the native LTX transformer key."""
    key = raw
    for prefix in ("model.diffusion_model.", "diffusion_model."):
        if key.startswith(prefix):
            key = key[len(prefix) :]
            break

    prefixes = ("_fsdp_wrapped_module.", "_orig_mod.", "module.", "base_model.model.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True

    return key.replace(".base_layer.", ".")


def _is_audio_param(raw: str) -> bool:
    """Match exactly the audio-full parameter set used by trainer_audio_full.py."""
    name = _checkpoint_key(raw)
    if ".lora_" in name:
        return False
    return name.startswith(_AUDIO_PREFIXES) or any(marker in name for marker in _AUDIO_MARKERS)


def _extract_tensor_state_dict(payload: object, path: str) -> dict[str, torch.Tensor]:
    """Extract a tensor state dict from a .pt/.pth payload."""
    if isinstance(payload, Mapping):
        for field in ("state_dict", "model_state_dict", "audio_state_dict", "model"):
            nested = payload.get(field)
            if isinstance(nested, Mapping) and any(isinstance(v, torch.Tensor) for v in nested.values()):
                payload = nested
                break

    if not isinstance(payload, Mapping):
        raise RuntimeError(f"Audio checkpoint {path} does not contain a state dict")

    tensors = {str(k): v for k, v in payload.items() if isinstance(v, torch.Tensor)}
    if not tensors:
        raise RuntimeError(f"Audio checkpoint {path} contains no tensors")
    return tensors


class AudioOverlayModelStateDictLoader:
    """
    Load the normal LTX model checkpoint, then replace only the audio-full tensors
    with a fine-tuned audio checkpoint before LoRA fusion happens.

    The audio checkpoint may be the trainer's native .safetensors file or a .pt/.pth
    state dict. LoRA files are delegated unchanged to the normal safetensors loader.
    """

    def __init__(self, audio_checkpoint_path: str) -> None:
        self.audio_checkpoint_path = str(Path(audio_checkpoint_path).expanduser().resolve())
        self._base_loader = SafetensorsModelStateDictLoader()

    def metadata(self, path: str) -> dict:
        return self._base_loader.metadata(path)

    def _load_audio_checkpoint(self) -> dict[str, torch.Tensor]:
        path = Path(self.audio_checkpoint_path)
        suffix = path.suffix.lower()

        if suffix == ".safetensors":
            raw_state = load_safetensors_file(str(path), device="cpu")
        elif suffix in {".pt", ".pth", ".ckpt"}:
            try:
                payload = torch.load(str(path), map_location="cpu", weights_only=True)
            except TypeError:
                payload = torch.load(str(path), map_location="cpu")
            raw_state = _extract_tensor_state_dict(payload, str(path))
        else:
            raise ValueError(
                f"Unsupported audio checkpoint format: {path}. "
                "Expected .safetensors, .pt, .pth, or .ckpt."
            )

        audio_state: dict[str, torch.Tensor] = {}
        ignored = 0
        for raw_key, value in raw_state.items():
            key = _checkpoint_key(raw_key)
            if not _is_audio_param(key):
                ignored += 1
                continue
            if key in audio_state:
                raise RuntimeError(f"Duplicate audio checkpoint key after normalization: {key}")
            audio_state[key] = value.detach().cpu()

        if not audio_state:
            raise RuntimeError(f"No audio-full weights found in audio checkpoint: {path}")
        if ignored:
            logging.info("[Audio CKPT] Ignored %d non-audio tensor(s) from %s", ignored, path)
        return audio_state

    def load(self, path, sd_ops=None, device: torch.device | None = None) -> StateDict:
        paths = path if isinstance(path, list) else [path]
        resolved_audio = self.audio_checkpoint_path
        audio_matches = [p for p in paths if str(Path(p).expanduser().resolve()) == resolved_audio]

        # LoRA loading and all other ordinary safetensors loads use the stock loader.
        if not audio_matches:
            return self._base_loader.load(path, sd_ops=sd_ops, device=device)
        if len(audio_matches) != 1:
            raise RuntimeError(f"Audio checkpoint appears multiple times in model paths: {resolved_audio}")

        base_paths = [p for p in paths if str(Path(p).expanduser().resolve()) != resolved_audio]
        if not base_paths:
            raise RuntimeError("Audio overlay loader requires the base LTX checkpoint together with --audio-ckpt")

        base_arg = base_paths[0] if len(base_paths) == 1 else base_paths
        base_state = self._base_loader.load(base_arg, sd_ops=sd_ops, device=device)
        audio_state = self._load_audio_checkpoint()

        expected_audio_keys = {key for key in base_state.sd if _is_audio_param(key)}
        source_audio_keys = set(audio_state)
        if source_audio_keys != expected_audio_keys:
            missing = sorted(expected_audio_keys - source_audio_keys)[:30]
            unexpected = sorted(source_audio_keys - expected_audio_keys)[:30]
            raise RuntimeError(
                "Audio checkpoint mismatch; "
                f"missing={missing}, unexpected={unexpected}"
            )

        target_device = device or base_state.device
        merged = dict(base_state.sd)
        for key, src in audio_state.items():
            dst = base_state.sd[key]
            if src.shape != dst.shape:
                raise RuntimeError(
                    f"Audio checkpoint shape mismatch for {key}: "
                    f"checkpoint={tuple(src.shape)}, model={tuple(dst.shape)}"
                )
            merged[key] = src.to(device=target_device, dtype=dst.dtype, non_blocking=True)

        logging.info(
            "[Audio CKPT] Overlaid %d fine-tuned audio tensors from %s before LoRA fusion",
            len(audio_state),
            self.audio_checkpoint_path,
        )
        return StateDict(
            sd=merged,
            device=target_device,
            size=sum(tensor.nbytes for tensor in merged.values()),
            dtype={tensor.dtype for tensor in merged.values()},
        )


def _build_audio_overlay_stage(
    checkpoint_path: str,
    audio_ckpt: str,
    dtype: torch.dtype,
    device: torch.device,
    loras: tuple[LoraPathStrengthAndSDOps, ...],
    registry: Registry,
) -> DiffusionStage:
    """Build a DiffusionStage whose base model is dev + fine-tuned audio weights."""
    builder = SingleGPUModelBuilder(
        model_class_configurator=LTXModelConfigurator,
        model_path=(checkpoint_path, audio_ckpt),
        model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
        loras=loras,
        model_loader=AudioOverlayModelStateDictLoader(audio_ckpt),
        registry=registry,
    )

    # LTX's DiffusionStage constructor changed in newer releases. Support both
    # the older checkpoint_path+transformer_builder API and the newer builder API.
    parameters = inspect.signature(DiffusionStage.__init__).parameters
    if "transformer_builder" in parameters:
        return DiffusionStage(
            checkpoint_path,
            dtype,
            device,
            loras=loras,
            registry=registry,
            offload_mode=OffloadMode.NONE,
            transformer_builder=builder,
        )

    return DiffusionStage(builder, dtype, device)


class ICLoraPipeline:
    """
    Two-stage video generation pipeline with In-Context (IC) LoRA support
    using the non-distilled LTX-2.3 base checkpoint plus a fine-tuned audio branch.

    Stage 1 runs the full/dev model with the regular LTX2 sigma schedule and
    multimodal CFG/STG guidance. IC-LoRA is applied in Stage 1 together with
    the reference-video conditioning.

    The fine-tuned audio checkpoint is overlaid on the dev transformer before
    LoRA fusion. Stage 2 uses the same audio-overlaid dev base and then fuses
    the distilled refinement LoRA.
    """

    def __init__(
        self,
        checkpoint_path: str,
        audio_ckpt: str,
        distilled_lora: list[LoraPathStrengthAndSDOps],
        spatial_upsampler_path: str,
        gemma_root: str,
        loras: list[LoraPathStrengthAndSDOps],
        device: torch.device | None = None,
        quantization: QuantizationPolicy | None = None,
        registry: Registry | None = None,
        compilation_config: CompilationConfig | None = None,
        offload_mode: OffloadMode = OffloadMode.NONE,
    ):
        audio_ckpt_path = Path(audio_ckpt).expanduser().resolve()
        if not audio_ckpt_path.is_file():
            raise FileNotFoundError(f"Audio checkpoint not found: {audio_ckpt_path}")
        if offload_mode != OffloadMode.NONE:
            raise ValueError(
                "--audio-ckpt currently requires --offload none because the fine-tuned "
                "audio weights must be overlaid before LoRA fusion."
            )
        if quantization is not None:
            raise ValueError(
                "--audio-ckpt currently does not support --quantization. "
                "Load the fine-tuned audio branch in bf16/full precision."
            )
        if compilation_config is not None:
            raise ValueError(
                "--audio-ckpt currently does not support --compile because compilation "
                "rewrites transformer state-dict keys before loading."
            )

        self.device = device or get_device()
        self.dtype = torch.bfloat16
        self._scheduler = LTX2Scheduler()
        self.audio_ckpt = str(audio_ckpt_path)

        self.prompt_encoder = PromptEncoder(
            checkpoint_path,
            gemma_root,
            self.dtype,
            self.device,
            registry=registry,
            offload_mode=offload_mode,
        )

        self.image_conditioner = ImageConditioner(checkpoint_path, self.dtype, self.device, registry=registry)
        stage_registry = registry or DummyRegistry()

        # Loading order is important:
        #   dev base -> fine-tuned audio base -> IC-LoRA
        # This reproduces training where the audio base was trainable while IC-LoRA stayed frozen.
        self.stage_1 = _build_audio_overlay_stage(
            checkpoint_path=checkpoint_path,
            audio_ckpt=self.audio_ckpt,
            dtype=self.dtype,
            device=self.device,
            loras=tuple(loras),
            registry=stage_registry,
        )

        # Stage 2 uses the same fine-tuned audio base, then fuses the official
        # distilled refinement LoRA. The final decoded audio still comes from Stage 1.
        self.stage_2 = _build_audio_overlay_stage(
            checkpoint_path=checkpoint_path,
            audio_ckpt=self.audio_ckpt,
            dtype=self.dtype,
            device=self.device,
            loras=tuple(distilled_lora),
            registry=stage_registry,
        )

        self.upsampler = VideoUpsampler(
            checkpoint_path, spatial_upsampler_path, self.dtype, self.device, registry=registry
        )
        self.video_decoder = VideoDecoder(checkpoint_path, self.dtype, self.device, registry=registry)
        self.audio_decoder = AudioDecoder(checkpoint_path, self.dtype, self.device, registry=registry)

        # Read reference scale factors from IC-LoRA metadata.
        self.reference_downscale_factor = 1
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
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        num_inference_steps: int,
        video_guider_params: MultiModalGuiderParams,
        audio_guider_params: MultiModalGuiderParams,
        images: list[ImageConditioningInput],
        video_conditioning: list[tuple[str, float]],
        enhance_prompt: bool = False,
        tiling_config: TilingConfig | None = None,
        conditioning_attention_strength: float = 1.0,
        skip_stage_2: bool = False,
        conditioning_attention_mask: torch.Tensor | None = None,
        max_batch_size: int = 1,
        stage_1_sigmas: torch.Tensor | None = None,
        stage_2_sigmas: torch.Tensor = STAGE_2_DISTILLED_SIGMAS,
    ) -> tuple[Iterator[torch.Tensor], Audio]:
        """
        Generate video with IC-LoRA conditioning using the non-distilled base model.

        Stage 1 uses the regular LTX2 scheduler and guided denoising (CFG/STG/
        cross-modal guidance). Stage 2 uses the distilled refinement schedule
        with the dedicated distilled LoRA.
        """
        assert_resolution(height=height, width=width, is_two_stage=True)
        if not (0.0 <= conditioning_attention_strength <= 1.0):
            raise ValueError(
                f"conditioning_attention_strength must be in [0.0, 1.0], got {conditioning_attention_strength}"
            )

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)

        ctx_p, ctx_n = self.prompt_encoder(
            [prompt, negative_prompt],
            enhance_first_prompt=enhance_prompt,
            enhance_prompt_image=images[0][0] if len(images) > 0 else None,
            enhance_prompt_seed=seed,
        )
        v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
        v_context_n, a_context_n = ctx_n.video_encoding, ctx_n.audio_encoding

        # Stage 1: full/dev model at half resolution with guided denoising.
        stage_1_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width // 2,
            height=height // 2,
            fps=frame_rate,
        )

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

        sigmas = (
            stage_1_sigmas if stage_1_sigmas is not None else self._scheduler.execute(steps=num_inference_steps)
        ).to(dtype=torch.float32, device=self.device)

        video_state, audio_state = self.stage_1(
            denoiser=GuidedDenoiser(
                v_context=v_context_p,
                a_context=a_context_p,
                video_guider=MultiModalGuider(
                    params=video_guider_params,
                    negative_context=v_context_n,
                ),
                audio_guider=MultiModalGuider(
                    params=audio_guider_params,
                    negative_context=a_context_n,
                ),
            ),
            sigmas=sigmas,
            noiser=noiser,
            width=stage_1_output_shape.width,
            height=stage_1_output_shape.height,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(
                context=v_context_p,
                conditionings=stage_1_conditionings,
            ),
            audio=ModalitySpec(
                context=a_context_p,
            ),
            max_batch_size=max_batch_size,
        )

        if skip_stage_2:
            logging.info("[IC-LoRA] Skipping Stage 2 (--skip-stage-2 enabled)")
            decoded_video = self.video_decoder(video_state.latent, tiling_config, generator)
            decoded_audio = self.audio_decoder(audio_state.latent)
            return decoded_video, decoded_audio

        # Stage 2: upsample and refine with the distilled Stage-2 LoRA/schedule.
        upscaled_video_latent = self.upsampler(video_state.latent[:1])

        stage_2_sigmas = stage_2_sigmas.to(dtype=torch.float32, device=self.device)
        stage_2_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width,
            height=height,
            fps=frame_rate,
        )
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

        # Keep Stage-1 audio as the final audio, matching the official non-distilled
        # two-stage pipeline. Stage 2 is used to refine the upscaled video.
        video_state, _ = self.stage_2(
            denoiser=SimpleDenoiser(v_context_p, a_context_p),
            sigmas=stage_2_sigmas,
            noiser=noiser,
            width=width,
            height=height,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(
                context=v_context_p,
                conditionings=stage_2_conditionings,
                noise_scale=stage_2_sigmas[0].item(),
                initial_latent=upscaled_video_latent,
            ),
            audio=ModalitySpec(
                context=a_context_p,
                noise_scale=stage_2_sigmas[0].item(),
                initial_latent=audio_state.latent,
            ),
        )

        decoded_video = self.video_decoder(video_state.latent, tiling_config, generator)
        decoded_audio = self.audio_decoder(audio_state.latent)
        return decoded_video, decoded_audio

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

    # Non-distilled/full LTX-2.3 checkpoint, e.g. ltx-2.3-22b-dev.safetensors.
    checkpoint_path = detect_checkpoint_path(distilled=False)
    params = detect_params(checkpoint_path)
    parser = default_2_stage_arg_parser(params=params)

    parser.add_argument(
        "--audio-ckpt",
        type=str,
        required=True,
        help=(
            "Path to the fine-tuned LTX audio-branch checkpoint. Supports the "
            "audio_full_weights_step_*.safetensors format produced by trainer_audio_full.py "
            "and tensor state dicts saved as .pt/.pth/.ckpt."
        ),
    )
    parser.add_argument(
        "--video-conditioning",
        action=VideoConditioningAction,
        nargs=2,
        metavar=("PATH", "STRENGTH"),
        required=True,
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
        "--skip-stage-2",
        action="store_true",
        help=(
            "Skip Stage 2 upsampling and distilled-LoRA refinement. Output will be "
            "at half resolution (height//2, width//2)."
        ),
    )
    args = parser.parse_args()

    conditioning_attention_mask = None
    conditioning_attention_strength = 1.0
    if args.conditioning_attention_mask is not None:
        mask_path, mask_strength = args.conditioning_attention_mask
        conditioning_attention_strength = mask_strength
        conditioning_attention_mask = _load_mask_video(
            mask_path=mask_path,
            height=args.height // 2,
            width=args.width // 2,
            num_frames=args.num_frames,
        )

    pipeline = ICLoraPipeline(
        checkpoint_path=args.checkpoint_path,
        audio_ckpt=args.audio_ckpt,
        distilled_lora=args.distilled_lora,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=args.gemma_root,
        loras=tuple(args.lora) if args.lora else (),
        quantization=args.quantization,
        compilation_config=args.compile,
        offload_mode=args.offload_mode,
    )
    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)

    video, audio = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        num_inference_steps=args.num_inference_steps,
        video_guider_params=MultiModalGuiderParams(
            cfg_scale=args.video_cfg_guidance_scale,
            stg_scale=args.video_stg_guidance_scale,
            rescale_scale=args.video_rescale_scale,
            modality_scale=args.a2v_guidance_scale,
            skip_step=args.video_skip_step,
            stg_blocks=args.video_stg_blocks,
        ),
        audio_guider_params=MultiModalGuiderParams(
            cfg_scale=args.audio_cfg_guidance_scale,
            stg_scale=args.audio_stg_guidance_scale,
            rescale_scale=args.audio_rescale_scale,
            modality_scale=args.v2a_guidance_scale,
            skip_step=args.audio_skip_step,
            stg_blocks=args.audio_stg_blocks,
        ),
        images=args.images,
        video_conditioning=args.video_conditioning,
        enhance_prompt=args.enhance_prompt,
        tiling_config=tiling_config,
        conditioning_attention_strength=conditioning_attention_strength,
        skip_stage_2=args.skip_stage_2,
        conditioning_attention_mask=conditioning_attention_mask,
        max_batch_size=args.max_batch_size,
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
    """Load a mask video and return a pixel-space tensor of shape (1, 1, F, H, W)."""
    device = get_device()
    frame_gen = decode_video_by_frame(path=mask_path, frame_cap=num_frames, device=device)
    mask_video = video_preprocess(frame_gen, height, width, torch.bfloat16, device)
    mask = mask_video.mean(dim=1, keepdim=True)
    mask = (mask + 1.0) / 2.0
    return mask.clamp(0.0, 1.0)


if __name__ == "__main__":
    main()
