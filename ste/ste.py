"""
SpatialTrackEncoder modules for converting LTX motion-track reference video latent
into audio-side spatial condition tokens.

Expected LTX-2.3 shapes from the user's setting:
    target video latent:            [B, 6144, 128] ~= [B, 16, 16, 24, 128]
    motion-track reference latent:  [B, 1536, 128] ~= [B, 16,  8, 12, 128]
    audio reference/control latent: [B,  122, 128]

This file provides multiple architectures from simple to complex:
    1. SimpleSpatialTrackEncoder
    2. AttnPoolTCNSpatialTrackEncoder
    3. AudioQuerySpatialTrackEncoder
    4. HybridSpatialTrackEncoder

Recommended first serious version:
    AudioQuerySpatialTrackEncoder

Important assumption:
    ref_tokens are flattened in [T, H, W] order, i.e.
        ref_tokens.view(B, T, H, W, C)
    If your LTX code uses a different flattening order, modify `_unflatten_ref_tokens`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch.distributed as dist

EncoderType = Literal["simple", "attn_tcn", "audio_query", "hybrid"]


@dataclass
class SpatialTrackEncoderConfig:
    """Configuration for SpatialTrackEncoder variants."""

    dim: int = 128
    video_t: int = 16
    video_h: int = 8
    video_w: int = 12
    audio_t: int = 122
    num_heads: int = 8
    dropout: float = 0.0

    # Used by AttnPoolTCNSpatialTrackEncoder
    num_temporal_blocks: int = 4

    # Used by AudioQuerySpatialTrackEncoder / HybridSpatialTrackEncoder
    encoder_depth: int = 4
    decoder_depth: int = 2

    # Used by optional fusion/helper modules
    use_segment_embedding: bool = True

    @property
    def ref_num_tokens(self) -> int:
        return self.video_t * self.video_h * self.video_w


def _validate_ref_tokens(
    ref_tokens: torch.Tensor,
    *,
    cfg: SpatialTrackEncoderConfig,
) -> Tuple[int, int, int]:
    """Validate shape of reference motion-track latent tokens."""
    if ref_tokens.ndim != 3:
        raise ValueError(
            f"ref_tokens must be [B, N, C], but got shape {tuple(ref_tokens.shape)}"
        )

    b, n, c = ref_tokens.shape

    if c != cfg.dim:
        raise ValueError(f"Expected channel dim={cfg.dim}, but got C={c}")

    if n != cfg.ref_num_tokens:
        raise ValueError(
            f"Expected N={cfg.ref_num_tokens} = "
            f"{cfg.video_t}*{cfg.video_h}*{cfg.video_w}, but got N={n}. "
            "Please check video_t/video_h/video_w or LTX latent flattening."
        )

    return b, n, c


def _unflatten_ref_tokens(
    ref_tokens: torch.Tensor,
    cfg: SpatialTrackEncoderConfig,
) -> torch.Tensor:
    """
    Convert [B, T*H*W, C] to [B, T, H, W, C].

    Assumption:
        tokens are flattened in [T, H, W] order.

    If your LTX pipeline uses another ordering, modify this function only.
    """
    b, _, c = _validate_ref_tokens(ref_tokens, cfg=cfg)
    return ref_tokens.view(b, cfg.video_t, cfg.video_h, cfg.video_w, c)


class SimpleSpatialTrackEncoder(nn.Module):
    """
    Baseline encoder.

    Pipeline:
        [B,1536,128]
        -> [B,16,8,12,128]
        -> spatial average pooling
        -> [B,16,128]
        -> temporal interpolation 16 -> 122
        -> MLP
        -> [B,122,128]

    Pros:
        Very simple and stable.
    Cons:
        Spatial average pooling may wash out left/right position information.
    """

    def __init__(self, cfg: Optional[SpatialTrackEncoderConfig] = None):
        super().__init__()
        self.cfg = cfg or SpatialTrackEncoderConfig()
        dim = self.cfg.dim

        self.proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, ref_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ref_tokens: [B, 1536, 128]

        Returns:
            spatial_audio_tokens: [B, 122, 128]
        """
        x = _unflatten_ref_tokens(ref_tokens, self.cfg)
        # [B, T, H, W, C] -> [B, T, C]
        x = x.mean(dim=(2, 3))

        # [B, T, C] -> [B, C, T] -> interpolate -> [B, C, audio_t]
        x = F.interpolate(
            x.transpose(1, 2),
            size=self.cfg.audio_t,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

        return self.proj(x)


class SpatialAttentionPool(nn.Module):
    """Attention pooling over H*W latent grid for each latent time step."""

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, H, W, C]

        Returns:
            pooled: [B, T, C]
        """
        b, t, h, w, c = x.shape
        score = self.score(x).squeeze(-1)  # [B, T, H, W]
        weight = torch.softmax(score.view(b, t, h * w), dim=-1)  # [B, T, H*W]
        x_flat = x.view(b, t, h * w, c)
        pooled = torch.sum(x_flat * weight.unsqueeze(-1), dim=2)
        return pooled


class TemporalConvBlock(nn.Module):
    """Residual temporal convolution block over [B, T, C]."""

    def __init__(self, dim: int, kernel_size: int = 3, dropout: float = 0.0):
        super().__init__()
        padding = kernel_size // 2

        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, C]
        """
        residual = x
        y = self.mlp(self.norm(x))
        y = self.conv(y.transpose(1, 2)).transpose(1, 2)
        return residual + y


class AttnPoolTCNSpatialTrackEncoder(nn.Module):
    """
    Stronger baseline.

    Pipeline:
        [B,1536,128]
        -> [B,16,8,12,128]
        -> learned spatial attention pooling
        -> [B,16,128]
        -> temporal conv blocks
        -> temporal interpolation 16 -> 122
        -> [B,122,128]

    This version is a good first training candidate because it is lightweight
    but still preserves more spatial information than average pooling.
    """

    def __init__(self, cfg: Optional[SpatialTrackEncoderConfig] = None):
        super().__init__()
        self.cfg = cfg or SpatialTrackEncoderConfig()
        dim = self.cfg.dim

        self.spatial_pool = SpatialAttentionPool(dim)
        self.temporal_blocks = nn.Sequential(
            *[
                TemporalConvBlock(dim, dropout=self.cfg.dropout)
                for _ in range(self.cfg.num_temporal_blocks)
            ]
        )
        self.out_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
        )

    def forward(self, ref_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ref_tokens: [B, 1536, 128]

        Returns:
            spatial_audio_tokens: [B, 122, 128]
        """
        x = _unflatten_ref_tokens(ref_tokens, self.cfg)
        x = self.spatial_pool(x)  # [B, 16, 128]
        x = self.temporal_blocks(x)

        x = F.interpolate(
            x.transpose(1, 2),
            size=self.cfg.audio_t,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

        return self.out_proj(x)


class CrossAttentionBlock(nn.Module):
    """
    Decoder-style block:
        query self-attention
        query-to-memory cross-attention
        FFN
    """

    def __init__(
        self,
        dim: int = 128,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm_q1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_q2 = nn.LayerNorm(dim)
        self.norm_mem = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        hidden_dim = int(dim * mlp_ratio)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, q: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q:      [B, Nq, C]
            memory: [B, Nm, C]
        """
        q_norm = self.norm_q1(q)
        self_out, _ = self.self_attn(
            q_norm,
            q_norm,
            q_norm,
            need_weights=False,
        )
        q = q + self_out

        q_norm = self.norm_q2(q)
        mem_norm = self.norm_mem(memory)
        cross_out, _ = self.cross_attn(
            q_norm,
            mem_norm,
            mem_norm,
            need_weights=False,
        )
        q = q + cross_out
        q = q + self.ffn(self.norm_ffn(q))
        return q


class AudioQuerySpatialTrackEncoder(nn.Module):
    """
    Recommended main version.

    Pipeline:
        ref_tokens [B,1536,128]
        -> unflatten [B,16,8,12,128]
        -> add 3D spatio-temporal positional embedding
        -> transformer encoder over track memory [B,1536,128]
        -> 122 learned audio-time queries cross-attend to track memory
        -> [B,122,128]

    Intuition:
        The reference video latent is the memory.
        The audio tokens are queries.
        Each audio token learns where/when to read spatial motion information.
    """

    def __init__(self, cfg: Optional[SpatialTrackEncoderConfig] = None):
        super().__init__()
        self.cfg = cfg or SpatialTrackEncoderConfig()
        dim = self.cfg.dim

        if dim % self.cfg.num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={self.cfg.num_heads}")

        self.video_pos = nn.Parameter(
            torch.zeros(1, self.cfg.video_t, self.cfg.video_h, self.cfg.video_w, dim)
        )
        self.audio_queries = nn.Parameter(torch.randn(1, self.cfg.audio_t, dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=self.cfg.num_heads,
            dim_feedforward=dim * 4,
            dropout=self.cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.track_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.cfg.encoder_depth,
            enable_nested_tensor=False,
        )

        self.cross_blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    dim=dim,
                    num_heads=self.cfg.num_heads,
                    dropout=self.cfg.dropout,
                )
                for _ in range(self.cfg.decoder_depth)
            ]
        )

        self.out_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.video_pos, std=0.02)
        nn.init.trunc_normal_(self.audio_queries, std=0.02)

    # def forward(self, ref_tokens: torch.Tensor) -> torch.Tensor:
    #     """
    #     Args:
    #         ref_tokens: [B, 1536, 128]

    #     Returns:
    #         spatial_audio_tokens: [B, 122, 128]
    #     """
    #     b, _, c = _validate_ref_tokens(
    #         ref_tokens,
    #         cfg=self.cfg,
    #     )

    #     x = ref_tokens.view(
    #         b,
    #         self.cfg.video_t,
    #         self.cfg.video_h,
    #         self.cfg.video_w,
    #         c,
    #     )

    #     # 在执行位置编码相加前，打印每个 rank 的 shape。
    #     _print_shapes_by_rank(
    #         x=x,
    #         video_pos=self.video_pos,
    #     )

    #     x = x + self.video_pos

    #     memory = x.view(
    #         b,
    #         self.cfg.ref_num_tokens,
    #         c,
    #     )
    #     memory = self.track_encoder(memory)

    #     q = self.audio_queries.expand(b, -1, -1)

    #     for block in self.cross_blocks:
    #         q = block(q, memory)

    #     return self.out_proj(q)

    def forward(self, ref_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ref_tokens: [B, 1536, 128]

        Returns:
            spatial_audio_tokens: [B, 122, 128]
        """
        b, _, c = _validate_ref_tokens(ref_tokens, cfg=self.cfg)

        x = ref_tokens.view(
            b,
            self.cfg.video_t,
            self.cfg.video_h,
            self.cfg.video_w,
            c,
        )
        x = x + self.video_pos

        memory = x.view(b, self.cfg.ref_num_tokens, c)
        memory = self.track_encoder(memory)

        q = self.audio_queries.expand(b, -1, -1)
        for block in self.cross_blocks:
            q = block(q, memory)

        return self.out_proj(q)





class SpatialMomentPool(nn.Module):
    """
    Differentiable spatial moment pooling.

    This module estimates an explicit spatial bottleneck from latent tokens:
        soft_x, soft_y, var_x, var_y, confidence

    It does NOT use bbox/mask as input. The attention map is learned from the
    sparse-track reference latent itself.
    """

    def __init__(self, dim: int, video_h: int, video_w: int):
        super().__init__()
        self.video_h = video_h
        self.video_w = video_w

        self.score = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )

        xs = torch.linspace(-1.0, 1.0, video_w)
        ys = torch.linspace(-1.0, 1.0, video_h)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

        self.register_buffer("grid_x", grid_x.view(1, 1, video_h, video_w))
        self.register_buffer("grid_y", grid_y.view(1, 1, video_h, video_w))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, H, W, C]

        Returns:
            moments: [B, T, 5]
                [soft_x, soft_y, var_x, var_y, confidence]
        """
        b, t, h, w, _ = x.shape
        if h != self.video_h or w != self.video_w:
            raise ValueError(
                f"Expected H,W=({self.video_h},{self.video_w}), got ({h},{w})"
            )

        score = self.score(x).squeeze(-1)  # [B, T, H, W]
        weight = torch.softmax(score.view(b, t, h * w), dim=-1).view(b, t, h, w)

        x_mean = torch.sum(weight * self.grid_x, dim=(2, 3))
        y_mean = torch.sum(weight * self.grid_y, dim=(2, 3))
        x_var = torch.sum(weight * (self.grid_x - x_mean[:, :, None, None]) ** 2, dim=(2, 3))
        y_var = torch.sum(weight * (self.grid_y - y_mean[:, :, None, None]) ** 2, dim=(2, 3))

        entropy = -torch.sum(weight * torch.log(weight.clamp_min(1e-6)), dim=(2, 3))
        max_entropy = torch.log(torch.tensor(float(h * w), device=x.device, dtype=x.dtype))
        confidence = 1.0 - entropy / max_entropy

        return torch.stack([x_mean, y_mean, x_var, y_var, confidence], dim=-1)


class HybridSpatialTrackEncoder(nn.Module):
    """
    Hybrid version:
        implicit AudioQuerySpatialTrackEncoder
        + explicit spatial moment bottleneck
        + fusion MLP

    This version is useful for stronger experiments and ablations.
    """

    def __init__(self, cfg: Optional[SpatialTrackEncoderConfig] = None):
        super().__init__()
        self.cfg = cfg or SpatialTrackEncoderConfig()
        dim = self.cfg.dim

        self.implicit_encoder = AudioQuerySpatialTrackEncoder(self.cfg)
        self.moment_pool = SpatialMomentPool(
            dim=dim,
            video_h=self.cfg.video_h,
            video_w=self.cfg.video_w,
        )
        self.moment_encoder = nn.Sequential(
            nn.Linear(5, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, ref_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ref_tokens: [B, 1536, 128]

        Returns:
            spatial_audio_tokens: [B, 122, 128]
        """
        x = _unflatten_ref_tokens(ref_tokens, self.cfg)

        implicit_tokens = self.implicit_encoder(ref_tokens)  # [B, 122, 128]

        moments = self.moment_pool(x)  # [B, 16, 5]
        moments = F.interpolate(
            moments.transpose(1, 2),
            size=self.cfg.audio_t,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)  # [B, 122, 5]

        explicit_tokens = self.moment_encoder(moments)  # [B, 122, 128]
        fused = torch.cat([implicit_tokens, explicit_tokens], dim=-1)
        return self.fusion(fused)


class AudioSpatialConditionConcat(nn.Module):
    """
    Helper for IC-LoRA-style concatenation.

    Given:
        spatial_tokens:     [B, 122, 128]
        noisy_audio_tokens: [B, 122, 128]

    Returns:
        audio_input_tokens: [B, 244, 128]

    During training, loss should normally be computed only on the target audio
    part, i.e. output[:, 122:, :], not on the condition part.
    """

    def __init__(self, cfg: Optional[SpatialTrackEncoderConfig] = None):
        super().__init__()
        self.cfg = cfg or SpatialTrackEncoderConfig()
        dim = self.cfg.dim

        if self.cfg.use_segment_embedding:
            self.cond_segment = nn.Parameter(torch.zeros(1, 1, dim))
            self.audio_segment = nn.Parameter(torch.zeros(1, 1, dim))
        else:
            self.register_parameter("cond_segment", None)
            self.register_parameter("audio_segment", None)

    def forward(
        self,
        spatial_tokens: torch.Tensor,
        noisy_audio_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if spatial_tokens.shape != noisy_audio_tokens.shape:
            raise ValueError(
                "spatial_tokens and noisy_audio_tokens should have the same shape, "
                f"but got {tuple(spatial_tokens.shape)} and {tuple(noisy_audio_tokens.shape)}"
            )

        if spatial_tokens.shape[1] != self.cfg.audio_t:
            raise ValueError(
                f"Expected audio_t={self.cfg.audio_t}, got {spatial_tokens.shape[1]}"
            )

        if self.cond_segment is not None:
            spatial_tokens = spatial_tokens + self.cond_segment
            noisy_audio_tokens = noisy_audio_tokens + self.audio_segment

        return torch.cat([spatial_tokens, noisy_audio_tokens], dim=1)


def build_spatial_track_encoder(
    encoder_type: EncoderType = "audio_query",
    cfg: Optional[SpatialTrackEncoderConfig] = None,
) -> nn.Module:
    """
    Factory function.

    Args:
        encoder_type:
            "simple"      -> SimpleSpatialTrackEncoder
            "attn_tcn"    -> AttnPoolTCNSpatialTrackEncoder
            "audio_query" -> AudioQuerySpatialTrackEncoder, recommended
            "hybrid"      -> HybridSpatialTrackEncoder
        cfg:
            SpatialTrackEncoderConfig

    Returns:
        nn.Module that maps [B,1536,128] to [B,122,128].
    """
    cfg = cfg or SpatialTrackEncoderConfig()

    if encoder_type == "simple":
        return SimpleSpatialTrackEncoder(cfg)
    if encoder_type == "attn_tcn":
        return AttnPoolTCNSpatialTrackEncoder(cfg)
    if encoder_type == "audio_query":
        return AudioQuerySpatialTrackEncoder(cfg)
    if encoder_type == "hybrid":
        return HybridSpatialTrackEncoder(cfg)

    raise ValueError(
        f"Unknown encoder_type={encoder_type!r}. "
        "Choose from: 'simple', 'attn_tcn', 'audio_query', 'hybrid'."
    )


@torch.no_grad()
def _smoke_test(device: str = "cpu") -> None:
    """Run shape checks for all encoder variants."""
    cfg = SpatialTrackEncoderConfig()
    ref_tokens = torch.randn(1, cfg.ref_num_tokens, cfg.dim, device=device)
    print(ref_tokens.shape)
    noisy_audio_tokens = torch.randn(1, cfg.audio_t, cfg.dim, device=device)

    for encoder_type in ["simple", "attn_tcn", "audio_query", "hybrid"]:
        encoder = build_spatial_track_encoder(encoder_type, cfg).to(device)
        encoder.eval()
        out = encoder(ref_tokens)
        print(f"{encoder_type:>11s}: {tuple(out.shape)}")
        assert out.shape == (1, cfg.audio_t, cfg.dim)

    concat = AudioSpatialConditionConcat(cfg).to(device)
    spatial_tokens = build_spatial_track_encoder("audio_query", cfg).to(device)(ref_tokens)
    audio_input = concat(spatial_tokens, noisy_audio_tokens)
    print(f"concat input: {tuple(audio_input.shape)}")
    assert audio_input.shape == (1, cfg.audio_t * 2, cfg.dim)


# if __name__ == "__main__":
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     _smoke_test(device=device)




import torch
import torch.distributed as dist


def _get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def _print_shapes_by_rank(
    x: torch.Tensor,
    video_pos: torch.Tensor,
) -> None:
    """
    按 rank 顺序打印张量形状，避免多进程输出混乱。

    注意：所有 rank 都必须执行这个函数，否则 barrier 会阻塞。
    """
    if not dist.is_available() or not dist.is_initialized():
        print(
            "[single process] "
            f"x.shape={tuple(x.shape)}, "
            f"video_pos.shape={tuple(video_pos.shape)}",
            flush=True,
        )
        return

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    for current_rank in range(world_size):
        if rank == current_rank:
            print(
                f"[rank {rank}] "
                f"x.shape={tuple(x.shape)}, "
                f"video_pos.shape={tuple(video_pos.shape)}, "
                f"x.dtype={x.dtype}, "
                f"video_pos.dtype={video_pos.dtype}, "
                f"x.device={x.device}, "
                f"video_pos.device={video_pos.device}",
                flush=True,
            )

        # 等待当前 rank 打印完成，再让下一个 rank 打印。
        dist.barrier()
