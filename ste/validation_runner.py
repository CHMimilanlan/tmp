"""Track-aware validation runner for LTX-2 / Track-PRoPE.

This module subclasses the existing validation runner instead of copying its
entire implementation. It keeps all existing validation behavior and changes
only the parts required to pass ``track_xy`` and ``track_valid`` into the audio
Modality during positive, CFG and STG forward passes.

Usage in trainer.py:

    from prope.validation import ValidationRunner

Validation YAML:

    validation:
      samples:
        - prompt: "..."
          track: /absolute/path/to/.precomputed/track/sample_000001.pt
          conditions:
            ...

The corresponding ``ValidationSample`` model must contain:

    track: str | Path | None = None
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import Tensor

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.guiders import CFGGuider, STGGuider
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.model import X0Model
from ltx_core.types import AudioLatentShape, LatentState
from ltx_trainer import logger
from ltx_trainer.config import ValidationConfig, ValidationSample
from ltx_trainer.progress import SamplingContext
from ltx_trainer.validation_runner_bakcup import ValidationRunner as _BaseValidationRunner

if TYPE_CHECKING:
    from ltx_core.model.transformer import LTXModel


@dataclass(frozen=True)
class CachedTrack:
    """One normalized validation trajectory stored on CPU."""

    track_xy: Tensor      # [T_track, 2], float32, CPU
    track_valid: Tensor   # [T_track], bool, CPU
    source_path: Path


@dataclass(frozen=True)
class ActiveTrack:
    """Track information active while one validation sample is generated."""

    cached: CachedTrack
    target_audio_tokens: int


class ValidationRunner(_BaseValidationRunner):
    """Existing LTX validation runner extended with Track-PRoPE inputs."""

    def __init__(
        self,
        config: ValidationConfig,
        model_path: str | Path,
        text_encoder_path: str | Path | None,
        load_text_encoder_in_8bit: bool = False,
    ) -> None:
        super().__init__(
            config=config,
            model_path=model_path,
            text_encoder_path=text_encoder_path,
            load_text_encoder_in_8bit=load_text_encoder_in_8bit,
        )

        # Validation does not use the training DataLoader, so tracks must be
        # loaded from each ValidationSample.track path.
        self._cached_tracks = self._cache_validation_tracks()
        self._active_track: ActiveTrack | None = None

    # ------------------------------------------------------------------
    # Track loading and normalization
    # ------------------------------------------------------------------

    def _cache_validation_tracks(self) -> dict[int, CachedTrack]:
        """Load every configured validation track once and keep it on CPU."""
        cached: dict[int, CachedTrack] = {}

        for sample_index, sample in enumerate(self._config.samples):
            track_path_value = getattr(sample, "track", None)
            if track_path_value is None:
                continue

            track_path = Path(track_path_value).expanduser().resolve()
            cached[id(sample)] = self._load_track_file(track_path)

            logger.info(
                "Cached validation track "
                f"{sample_index + 1}: {track_path}"
            )

        return cached

    @staticmethod
    def _load_track_file(track_path: Path) -> CachedTrack:
        """Load the same track formats accepted by the training DataLoader.

        Accepted files:

        1. A tensor:
               Tensor[T, 2]

        2. A dictionary:
               {
                   "track_xy": Tensor[T, 2],
                   "track_valid": Tensor[T],  # optional
               }
        """
        if not track_path.exists():
            raise FileNotFoundError(
                f"Validation track file does not exist: {track_path}"
            )
        if not track_path.is_file():
            raise ValueError(
                f"Validation track path is not a file: {track_path}"
            )

        raw = torch.load(
            track_path,
            map_location="cpu",
            weights_only=True,
        )

        if isinstance(raw, Tensor):
            track_xy = raw
            track_valid = None
        elif isinstance(raw, dict):
            if "track_xy" not in raw:
                raise KeyError(
                    "Validation track dictionary must contain 'track_xy': "
                    f"{track_path}. Available keys: {sorted(raw)}"
                )
            track_xy = raw["track_xy"]
            track_valid = raw.get("track_valid")
        else:
            raise TypeError(
                "Validation track file must contain a Tensor or dictionary, "
                f"got {type(raw).__name__}: {track_path}"
            )

        if not isinstance(track_xy, Tensor):
            raise TypeError(
                "'track_xy' must be a torch.Tensor, "
                f"got {type(track_xy).__name__}: {track_path}"
            )

        # A validation file represents one sample. Permit an old singleton
        # batch dimension but remove it before caching.
        if track_xy.ndim == 3 and track_xy.shape[0] == 1:
            track_xy = track_xy.squeeze(0)

        if track_xy.ndim != 2 or track_xy.shape[-1] != 2:
            raise ValueError(
                "Validation track_xy must have shape [T, 2], "
                f"got {tuple(track_xy.shape)}: {track_path}"
            )

        track_xy = track_xy.to(
            device="cpu",
            dtype=torch.float32,
        ).contiguous()
        track_length = track_xy.shape[0]

        if track_length < 1:
            raise ValueError(
                f"Validation trajectory is empty: {track_path}"
            )

        if track_valid is None:
            track_valid = torch.ones(
                track_length,
                dtype=torch.bool,
            )
        else:
            if not isinstance(track_valid, Tensor):
                raise TypeError(
                    "'track_valid' must be a torch.Tensor, "
                    f"got {type(track_valid).__name__}: {track_path}"
                )

            if (
                track_valid.ndim == 2
                and track_valid.shape[0] == 1
            ):
                track_valid = track_valid.squeeze(0)

            if track_valid.shape != (track_length,):
                raise ValueError(
                    "Validation track_valid must have shape [T], "
                    f"expected {(track_length,)}, "
                    f"got {tuple(track_valid.shape)}: {track_path}"
                )

            track_valid = track_valid.to(
                device="cpu",
                dtype=torch.bool,
            ).contiguous()

        finite_mask = torch.isfinite(track_xy).all(dim=-1)
        track_valid = track_valid & finite_mask

        track_xy = torch.nan_to_num(
            track_xy,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        track_xy = torch.where(
            track_valid.unsqueeze(-1),
            track_xy,
            torch.zeros_like(track_xy),
        )

        return CachedTrack(
            track_xy=track_xy.contiguous(),
            track_valid=track_valid.contiguous(),
            source_path=track_path,
        )

    @staticmethod
    def _align_track_to_audio(
        *,
        cached_track: CachedTrack,
        target_audio_tokens: int,
        final_audio_tokens: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        """Align a frame trajectory with the final validation audio sequence.

        The existing validation runner constructs the sequence as:

            [generated target audio tokens, appended reference tokens]

        Therefore invalid entries are appended at the end. This differs from
        the current training FlexibleStrategy, where reference tokens are
        prepended and the invalid entries must be placed at the beginning.
        """
        if target_audio_tokens < 1:
            raise ValueError(
                f"target_audio_tokens must be positive, got "
                f"{target_audio_tokens}."
            )
        if final_audio_tokens < target_audio_tokens:
            raise ValueError(
                "Final validation audio sequence is shorter than the target "
                "audio sequence: "
                f"final={final_audio_tokens}, "
                f"target={target_audio_tokens}."
            )

        track_xy = cached_track.track_xy.unsqueeze(0)
        track_valid = cached_track.track_valid.unsqueeze(0)

        # Validation normally uses B=1. Expand rather than repeat so this also
        # remains correct if validation batching is introduced later.
        if batch_size != 1:
            track_xy = track_xy.expand(
                batch_size,
                -1,
                -1,
            )
            track_valid = track_valid.expand(
                batch_size,
                -1,
            )

        track_xy = track_xy.to(
            device=device,
            dtype=dtype,
            non_blocking=True,
        )
        track_valid = track_valid.to(
            device=device,
            dtype=torch.bool,
            non_blocking=True,
        )

        source_length = track_xy.shape[1]

        # Match the same temporal alignment used by the training strategy.
        if source_length != target_audio_tokens:
            track_xy = F.interpolate(
                track_xy.transpose(1, 2),
                size=target_audio_tokens,
                mode="linear",
                align_corners=True,
            ).transpose(1, 2)

            track_valid = F.interpolate(
                track_valid.to(torch.float32).unsqueeze(1),
                size=target_audio_tokens,
                mode="nearest",
            ).squeeze(1).to(torch.bool)

        # Current validation appends audio-reference / STE tokens after the
        # generated target audio sequence.
        suffix_length = final_audio_tokens - target_audio_tokens

        if suffix_length > 0:
            suffix_xy = torch.zeros(
                batch_size,
                suffix_length,
                2,
                device=device,
                dtype=dtype,
            )
            suffix_valid = torch.zeros(
                batch_size,
                suffix_length,
                device=device,
                dtype=torch.bool,
            )

            track_xy = torch.cat(
                [track_xy, suffix_xy],
                dim=1,
            )
            track_valid = torch.cat(
                [track_valid, suffix_valid],
                dim=1,
            )

        track_xy = torch.where(
            track_valid.unsqueeze(-1),
            track_xy,
            torch.zeros_like(track_xy),
        )

        expected_xy_shape = (
            batch_size,
            final_audio_tokens,
            2,
        )
        expected_valid_shape = (
            batch_size,
            final_audio_tokens,
        )

        if track_xy.shape != expected_xy_shape:
            raise RuntimeError(
                "Internal validation track_xy shape error: "
                f"expected {expected_xy_shape}, "
                f"got {tuple(track_xy.shape)}."
            )
        if track_valid.shape != expected_valid_shape:
            raise RuntimeError(
                "Internal validation track_valid shape error: "
                f"expected {expected_valid_shape}, "
                f"got {tuple(track_valid.shape)}."
            )

        return (
            track_xy.contiguous(),
            track_valid.contiguous(),
        )

    # ------------------------------------------------------------------
    # Generation entry point
    # ------------------------------------------------------------------

    def _generate_sample(
        self,
        sample: ValidationSample,
        cached_embeddings: Any,
        cached_media: Any,
        transformer: "LTXModel",
        device: torch.device,
        sampling_ctx: SamplingContext,
    ) -> tuple[Tensor | None, Tensor | None]:
        """Activate the sample track and delegate generation to the base runner."""
        cached_track = self._cached_tracks.get(id(sample))
        track_prope_enabled = self._uses_track_prope(
            transformer
        )
        audio_will_exist = (
            self._config.generate_audio
            or any(
                condition.type == "audio_to_video"
                for condition in sample.conditions
            )
        )

        if (
            track_prope_enabled
            and audio_will_exist
            and cached_track is None
        ):
            raise ValueError(
                "Track-PRoPE is enabled in the transformer, but this "
                "validation sample has no track file.\n"
                "Add a track path to the sample:\n"
                "  validation:\n"
                "    samples:\n"
                "      - prompt: ...\n"
                "        track: /path/to/sample_track.pt"
            )

        if cached_track is not None and audio_will_exist:
            dims = (
                sample.video_dims
                or self._config.video_dims
            )
            num_frames = dims[2]

            target_shape = AudioLatentShape.from_duration(
                batch=1,
                duration=(
                    num_frames
                    / self._config.frame_rate
                ),
            )
            target_audio_tokens = (
                target_shape.token_count()
            )

            self._active_track = ActiveTrack(
                cached=cached_track,
                target_audio_tokens=target_audio_tokens,
            )
        else:
            self._active_track = None

        try:
            # The base implementation eventually calls self._run_denoising(),
            # so Python dispatches to the track-aware override below.
            return super()._generate_sample(
                sample=sample,
                cached_embeddings=cached_embeddings,
                cached_media=cached_media,
                transformer=transformer,
                device=device,
                sampling_ctx=sampling_ctx,
            )
        finally:
            self._active_track = None

    @staticmethod
    def _uses_track_prope(
        transformer: torch.nn.Module,
    ) -> bool:
        """Inspect wrapped PEFT/FSDP models for Track-PRoPE blocks."""
        candidates: list[torch.nn.Module] = [
            transformer
        ]
        visited: set[int] = set()

        while candidates:
            module = candidates.pop()

            if id(module) in visited:
                continue
            visited.add(id(module))

            blocks = getattr(
                module,
                "transformer_blocks",
                None,
            )
            if blocks is not None:
                return any(
                    bool(
                        getattr(
                            block,
                            "use_track_prope",
                            False,
                        )
                    )
                    for block in blocks
                )

            for attribute in (
                "module",
                "model",
                "base_model",
            ):
                child = getattr(
                    module,
                    attribute,
                    None,
                )
                if isinstance(
                    child,
                    torch.nn.Module,
                ):
                    candidates.append(child)

            get_base_model = getattr(
                module,
                "get_base_model",
                None,
            )
            if callable(get_base_model):
                try:
                    child = get_base_model()
                except Exception:
                    child = None
                if isinstance(
                    child,
                    torch.nn.Module,
                ):
                    candidates.append(child)

        return False

    # ------------------------------------------------------------------
    # Track-aware denoising
    # ------------------------------------------------------------------

    def _run_denoising(  # noqa: PLR0913
        self,
        transformer: "LTXModel",
        video_state: LatentState | None,
        audio_state: LatentState | None,
        video_clean: LatentState | None,
        audio_clean: LatentState | None,
        *,
        video_frozen: bool,
        audio_frozen: bool,
        v_ctx_pos: Tensor,
        a_ctx_pos: Tensor,
        v_ctx_neg: Tensor | None,
        a_ctx_neg: Tensor | None,
        device: torch.device,
        sampling_ctx: SamplingContext,
    ) -> tuple[
        LatentState | None,
        LatentState | None,
    ]:
        """Run Euler denoising while attaching tracks to audio Modality."""
        config = self._config

        scheduler = LTX2Scheduler()
        sigmas = scheduler.execute(
            steps=config.inference_steps
        ).to(device).float()

        stepper = EulerDiffusionStep()
        cfg_guider = CFGGuider(
            config.guidance_scale
        )
        stg_guider = STGGuider(
            config.stg_scale
        )

        stg_perturbation_config = (
            self._build_stg_perturbation_config(
                config.stg_blocks,
                config.stg_mode,
            )
            if stg_guider.enabled()
            else None
        )

        track_xy: Tensor | None = None
        track_valid: Tensor | None = None

        if (
            audio_state is not None
            and self._active_track is not None
        ):
            batch_size = audio_state.latent.shape[0]
            final_audio_tokens = (
                audio_state.latent.shape[1]
            )

            track_xy, track_valid = (
                self._align_track_to_audio(
                    cached_track=(
                        self._active_track.cached
                    ),
                    target_audio_tokens=(
                        self._active_track
                        .target_audio_tokens
                    ),
                    final_audio_tokens=(
                        final_audio_tokens
                    ),
                    batch_size=batch_size,
                    device=device,
                    dtype=audio_state.latent.dtype,
                )
            )

        x0_model = X0Model(transformer)

        for step_index, sigma in enumerate(
            sigmas[:-1]
        ):
            video_sigma = (
                torch.zeros_like(sigma)
                if video_frozen
                else sigma
            )
            audio_sigma = (
                torch.zeros_like(sigma)
                if audio_frozen
                else sigma
            )

            video = (
                self._modality_from_latent_state(
                    state=video_state,
                    context=v_ctx_pos,
                    sigma=video_sigma.unsqueeze(0),
                )
                if video_state is not None
                else None
            )
            audio = (
                self._modality_from_latent_state(
                    state=audio_state,
                    context=a_ctx_pos,
                    sigma=audio_sigma.unsqueeze(0),
                    track_xy=track_xy,
                    track_valid=track_valid,
                )
                if audio_state is not None
                else None
            )

            positive_video, positive_audio = x0_model(
                video=video,
                audio=audio,
                perturbations=None,
            )
            denoised_video = positive_video
            denoised_audio = positive_audio

            # CFG uses dataclasses.replace, so track_xy and track_valid remain
            # attached to the negative audio Modality automatically.
            if (
                cfg_guider.enabled()
                and v_ctx_neg is not None
            ):
                negative_video_input = (
                    replace(
                        video,
                        context=v_ctx_neg,
                    )
                    if video is not None
                    else None
                )
                negative_audio_input = (
                    replace(
                        audio,
                        context=a_ctx_neg,
                    )
                    if audio is not None
                    else None
                )

                (
                    negative_video,
                    negative_audio,
                ) = x0_model(
                    video=negative_video_input,
                    audio=negative_audio_input,
                    perturbations=None,
                )

                if (
                    not video_frozen
                    and denoised_video is not None
                ):
                    denoised_video = (
                        denoised_video
                        + cfg_guider.delta(
                            positive_video,
                            negative_video,
                        )
                    )

                if (
                    not audio_frozen
                    and denoised_audio is not None
                ):
                    denoised_audio = (
                        denoised_audio
                        + cfg_guider.delta(
                            positive_audio,
                            negative_audio,
                        )
                    )

            # STG receives the same positive audio Modality and therefore the
            # same track tensors.
            if stg_perturbation_config is not None:
                (
                    perturbed_video,
                    perturbed_audio,
                ) = x0_model(
                    video=video,
                    audio=audio,
                    perturbations=(
                        stg_perturbation_config
                    ),
                )

                if (
                    not video_frozen
                    and denoised_video is not None
                ):
                    denoised_video = (
                        denoised_video
                        + stg_guider.delta(
                            positive_video,
                            perturbed_video,
                        )
                    )

                if (
                    not audio_frozen
                    and denoised_audio is not None
                    and perturbed_audio is not None
                ):
                    denoised_audio = (
                        denoised_audio
                        + stg_guider.delta(
                            positive_audio,
                            perturbed_audio,
                        )
                    )

            if (
                denoised_video is not None
                and video_clean is not None
                and video_state is not None
            ):
                denoised_video = (
                    self._post_process_latent(
                        denoised_video,
                        video_state.denoise_mask,
                        video_clean.clean_latent,
                    )
                )

            if (
                denoised_audio is not None
                and audio_clean is not None
                and audio_state is not None
            ):
                denoised_audio = (
                    self._post_process_latent(
                        denoised_audio,
                        audio_state.denoise_mask,
                        audio_clean.clean_latent,
                    )
                )

            if (
                video_state is not None
                and not video_frozen
            ):
                video_state = replace(
                    video_state,
                    latent=stepper.step(
                        video.latent,
                        denoised_video,
                        sigmas,
                        step_index,
                    ),
                )

            if (
                audio_state is not None
                and not audio_frozen
            ):
                audio_state = replace(
                    audio_state,
                    latent=stepper.step(
                        audio.latent,
                        denoised_audio,
                        sigmas,
                        step_index,
                    ),
                )

            sampling_ctx.advance_step()

        return video_state, audio_state

    @staticmethod
    def _modality_from_latent_state(
        state: LatentState,
        context: Tensor,
        sigma: Tensor,
        *,
        track_xy: Tensor | None = None,
        track_valid: Tensor | None = None,
    ) -> Modality:
        """Build a Modality and optionally attach Track-PRoPE tensors."""
        if track_xy is not None:
            expected_xy_shape = (
                state.latent.shape[0],
                state.latent.shape[1],
                2,
            )
            if track_xy.shape != expected_xy_shape:
                raise ValueError(
                    "Validation track_xy does not match the current latent "
                    f"sequence: expected {expected_xy_shape}, "
                    f"got {tuple(track_xy.shape)}."
                )

        if track_valid is not None:
            expected_valid_shape = (
                state.latent.shape[0],
                state.latent.shape[1],
            )
            if track_valid.shape != expected_valid_shape:
                raise ValueError(
                    "Validation track_valid does not match the current latent "
                    f"sequence: expected {expected_valid_shape}, "
                    f"got {tuple(track_valid.shape)}."
                )

        return Modality(
            enabled=True,
            latent=state.latent,
            sigma=sigma,
            timesteps=state.denoise_mask * sigma,
            positions=state.positions,
            context=context,
            context_mask=None,
            attention_mask=state.attention_mask,
            track_xy=track_xy,
            track_valid=track_valid,
        )
