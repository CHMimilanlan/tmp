from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass(frozen=True)
class SimpleTrackSummaryConfig:
    """Configuration for the lightweight IC-LoRA summary-token branch.

    The module implements:

        learned summary queries -> reference video tokens
        reference-aware summary -> target video tokens
        audio tokens -> final summary tokens

    Expected default LTX-2.3 shapes:

        target video:    [B, 6144, 4096]
        reference video: [B, 1536, 4096]
        audio:           [B,  122, 2048]
        summary:         [B,   32, 2048]
    """

    video_dim: int = 4096
    audio_dim: int = 2048
    num_summary_tokens: int = 32
    num_heads: int = 16
    dropout: float = 0.0

    target_token_count: int = 6144
    reference_token_count: int = 1536

    use_temporal_summary_positions: bool = True
    use_summary_ffn: bool = True
    use_residual_fusion: bool = True
    ffn_multiplier: int = 4
    norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.video_dim <= 0 or self.audio_dim <= 0:
            raise ValueError("video_dim and audio_dim must be positive")
        if self.num_summary_tokens <= 0:
            raise ValueError("num_summary_tokens must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.audio_dim % self.num_heads != 0:
            raise ValueError(
                f"audio_dim={self.audio_dim} must be divisible by "
                f"num_heads={self.num_heads}"
            )
        if self.target_token_count <= 0:
            raise ValueError("target_token_count must be positive")
        if self.reference_token_count <= 0:
            raise ValueError("reference_token_count must be positive")


@dataclass
class SimpleTrackSummaryOutput:
    """Outputs useful for both training and optional debugging."""

    audio_delta: torch.Tensor
    summary_tokens: torch.Tensor
    reference_summary: torch.Tensor
    target_summary: torch.Tensor
    target_tokens: torch.Tensor
    reference_tokens: torch.Tensor


class _SummaryFFN(nn.Module):
    def __init__(self, dim: int, multiplier: int, dropout: float) -> None:
        super().__init__()
        hidden_dim = dim * multiplier
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleTrackSummaryTokens(nn.Module):
    """Lightweight summary-token adapter for stereo audio-video generation.

    The three attention stages do NOT share parameters:

    1. ``summary_to_reference`` extracts motion/spatial information from the
       appended IC-LoRA reference tokens.
    2. ``summary_to_target`` grounds the reference summary in the current target
       video state.
    3. ``audio_to_summary`` lets every audio token retrieve the spatial summary
       relevant to its own current hidden state.

    The final projection is zero-initialized. Therefore, immediately after
    initialization this adapter contributes an exact zero residual and preserves
    the pretrained BasicAVTransformerBlock behavior.
    """

    def __init__(self, config: SimpleTrackSummaryConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.audio_dim

        self.summary_tokens = nn.Parameter(
            torch.empty(1, config.num_summary_tokens, dim)
        )
        if config.use_temporal_summary_positions:
            self.summary_temporal_positions = nn.Parameter(
                torch.empty(1, config.num_summary_tokens, dim)
            )
        else:
            self.register_parameter("summary_temporal_positions", None)

        # Attention 1: Q=summary, K/V=reference video.
        self.summary_to_reference = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            kdim=config.video_dim,
            vdim=config.video_dim,
            batch_first=True,
        )

        # Attention 2: Q=reference summary, K/V=target video.
        # This is intentionally a separate parameter set.
        self.summary_to_target = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            kdim=config.video_dim,
            vdim=config.video_dim,
            batch_first=True,
        )

        # Final injection: Q=audio, K/V=summary.
        self.audio_to_summary = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )

        self.summary_query_norm = nn.LayerNorm(dim, eps=config.norm_eps)
        self.reference_kv_norm = nn.LayerNorm(
            config.video_dim, eps=config.norm_eps
        )
        self.reference_summary_norm = nn.LayerNorm(dim, eps=config.norm_eps)
        self.target_kv_norm = nn.LayerNorm(
            config.video_dim, eps=config.norm_eps
        )
        self.audio_query_norm = nn.LayerNorm(dim, eps=config.norm_eps)
        self.final_summary_norm = nn.LayerNorm(dim, eps=config.norm_eps)

        if config.use_summary_ffn:
            self.reference_ffn_norm = nn.LayerNorm(dim, eps=config.norm_eps)
            self.reference_ffn = _SummaryFFN(
                dim, config.ffn_multiplier, config.dropout
            )
            self.target_ffn_norm = nn.LayerNorm(dim, eps=config.norm_eps)
            self.target_ffn = _SummaryFFN(
                dim, config.ffn_multiplier, config.dropout
            )
        else:
            self.reference_ffn_norm = None
            self.reference_ffn = None
            self.target_ffn_norm = None
            self.target_ffn = None

        if config.use_residual_fusion:
            self.summary_fusion = nn.Sequential(
                nn.LayerNorm(dim * 3, eps=config.norm_eps),
                nn.Linear(dim * 3, dim),
                nn.GELU(approximate="tanh"),
                nn.Linear(dim, dim),
            )
        else:
            self.summary_fusion = None

        # A separate zero-initialized projection is preferable to a zero scalar
        # gate: gradients immediately reach this projection, while the base
        # transformer output remains exactly unchanged at initialization.
        self.audio_output = nn.Linear(dim, dim)
        nn.init.zeros_(self.audio_output.weight)
        nn.init.zeros_(self.audio_output.bias)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.summary_tokens, mean=0.0, std=0.02)
        if self.summary_temporal_positions is not None:
            nn.init.normal_(
                self.summary_temporal_positions, mean=0.0, std=0.02
            )

    def _resolve_counts(
        self,
        video_hidden: torch.Tensor,
        target_token_count: Optional[int],
        reference_token_count: Optional[int],
    ) -> tuple[int, int]:
        total_tokens = video_hidden.shape[1]
        target_count = (
            self.config.target_token_count
            if target_token_count is None
            else int(target_token_count)
        )
        reference_count = (
            self.config.reference_token_count
            if reference_token_count is None
            else int(reference_token_count)
        )

        if target_count <= 0 or reference_count <= 0:
            raise ValueError(
                "target_token_count and reference_token_count must be positive"
            )
        if target_count + reference_count > total_tokens:
            raise ValueError(
                "IC-LoRA token split exceeds video sequence length: "
                f"target={target_count}, reference={reference_count}, "
                f"total={total_tokens}"
            )
        return target_count, reference_count

    def split_video_tokens(
        self,
        video_hidden: torch.Tensor,
        *,
        target_token_count: Optional[int] = None,
        reference_token_count: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split official IC-LoRA order: [target tokens, reference tokens]."""

        if video_hidden.ndim != 3:
            raise ValueError(
                "video_hidden must have shape [B, N, C], got "
                f"{tuple(video_hidden.shape)}"
            )
        if video_hidden.shape[-1] != self.config.video_dim:
            raise ValueError(
                f"Expected video_dim={self.config.video_dim}, got "
                f"{video_hidden.shape[-1]}"
            )

        target_count, reference_count = self._resolve_counts(
            video_hidden,
            target_token_count,
            reference_token_count,
        )
        target_tokens = video_hidden[:, :target_count]
        reference_start = target_count
        reference_end = target_count + reference_count
        reference_tokens = video_hidden[:, reference_start:reference_end]
        return target_tokens, reference_tokens

    @staticmethod
    def _to_key_padding_mask(
        valid_mask: Optional[torch.Tensor],
        *,
        expected_length: int,
        name: str,
    ) -> Optional[torch.Tensor]:
        """Convert a valid-token mask to MultiheadAttention padding semantics."""

        if valid_mask is None:
            return None
        if valid_mask.ndim != 2 or valid_mask.shape[1] != expected_length:
            raise ValueError(
                f"{name} must have shape [B, {expected_length}], got "
                f"{tuple(valid_mask.shape)}"
            )
        # MultiheadAttention uses True for positions that must be ignored.
        return ~valid_mask.to(dtype=torch.bool)

    def forward(
        self,
        *,
        video_hidden: torch.Tensor,
        audio_hidden: torch.Tensor,
        target_token_count: Optional[int] = None,
        reference_token_count: Optional[int] = None,
        target_valid: Optional[torch.Tensor] = None,
        reference_valid: Optional[torch.Tensor] = None,
        need_attention_weights: bool = False,
    ) -> SimpleTrackSummaryOutput:
        if audio_hidden.ndim != 3:
            raise ValueError(
                "audio_hidden must have shape [B, N, C], got "
                f"{tuple(audio_hidden.shape)}"
            )
        if audio_hidden.shape[-1] != self.config.audio_dim:
            raise ValueError(
                f"Expected audio_dim={self.config.audio_dim}, got "
                f"{audio_hidden.shape[-1]}"
            )
        if video_hidden.shape[0] != audio_hidden.shape[0]:
            raise ValueError("video_hidden and audio_hidden batch sizes differ")

        target_tokens, reference_tokens = self.split_video_tokens(
            video_hidden,
            target_token_count=target_token_count,
            reference_token_count=reference_token_count,
        )

        batch_size = video_hidden.shape[0]
        summary = self.summary_tokens.expand(batch_size, -1, -1)
        if self.summary_temporal_positions is not None:
            summary = summary + self.summary_temporal_positions

        reference_padding_mask = self._to_key_padding_mask(
            reference_valid,
            expected_length=reference_tokens.shape[1],
            name="reference_valid",
        )
        target_padding_mask = self._to_key_padding_mask(
            target_valid,
            expected_length=target_tokens.shape[1],
            name="target_valid",
        )

        reference_delta, _ = self.summary_to_reference(
            query=self.summary_query_norm(summary),
            key=self.reference_kv_norm(reference_tokens),
            value=self.reference_kv_norm(reference_tokens),
            key_padding_mask=reference_padding_mask,
            need_weights=need_attention_weights,
            average_attn_weights=False,
        )
        reference_summary = summary + reference_delta
        if self.reference_ffn is not None:
            reference_summary = reference_summary + self.reference_ffn(
                self.reference_ffn_norm(reference_summary)
            )

        target_delta, _ = self.summary_to_target(
            query=self.reference_summary_norm(reference_summary),
            key=self.target_kv_norm(target_tokens),
            value=self.target_kv_norm(target_tokens),
            key_padding_mask=target_padding_mask,
            need_weights=need_attention_weights,
            average_attn_weights=False,
        )
        target_summary = reference_summary + target_delta
        if self.target_ffn is not None:
            target_summary = target_summary + self.target_ffn(
                self.target_ffn_norm(target_summary)
            )

        if self.summary_fusion is not None:
            # Explicitly expose the reference-target residual requested by the
            # spatial-audio task.
            final_summary = self.summary_fusion(
                torch.cat(
                    [
                        reference_summary,
                        target_summary,
                        target_summary - reference_summary,
                    ],
                    dim=-1,
                )
            )
        else:
            final_summary = target_summary

        audio_delta, _ = self.audio_to_summary(
            query=self.audio_query_norm(audio_hidden),
            key=self.final_summary_norm(final_summary),
            value=self.final_summary_norm(final_summary),
            need_weights=need_attention_weights,
            average_attn_weights=False,
        )
        audio_delta = self.audio_output(audio_delta)

        return SimpleTrackSummaryOutput(
            audio_delta=audio_delta,
            summary_tokens=final_summary,
            reference_summary=reference_summary,
            target_summary=target_summary,
            target_tokens=target_tokens,
            reference_tokens=reference_tokens,
        )
