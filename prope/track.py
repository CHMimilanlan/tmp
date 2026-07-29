from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class TrackPRoPEConfig:
    """
    Track-PRoPE adapter configuration.

    model_dim:
        LTX audio transformer hidden dimension，注意不是 Audio VAE latent 的 128。

    adapter_dim:
        Track attention branch 的内部维度。为了节省显存，可以远小于 model_dim。

    num_heads:
        Track attention 的 head 数量。

    matrix_scale:
        轨迹坐标写入 4x4 矩阵时的缩放系数。
    """

    model_dim: int
    adapter_dim: int = 512
    num_heads: int = 8
    matrix_scale: float = 0.5
    norm_eps: float = 1e-6


def resample_track_to_audio_tokens(
    track_xy: Tensor,
    target_length: int,
    track_valid: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """
    将视频帧轨迹对齐到 Audio token 时间长度。

    Args:
        track_xy:
            [B, T_video, 2]，坐标范围通常为 [0,1]。
        target_length:
            Audio token 数，例如 122。
        track_valid:
            [B, T_video]，1 表示当前帧轨迹有效。

    Returns:
        audio_track_xy:
            [B, target_length, 2]
        audio_track_valid:
            [B, target_length]
    """
    if track_xy.ndim != 3 or track_xy.shape[-1] != 2:
        raise ValueError(
            f"track_xy must have shape [B,T,2], got {tuple(track_xy.shape)}"
        )

    finite = torch.isfinite(track_xy).all(dim=-1)
    track_xy = torch.nan_to_num(
        track_xy,
        nan=0.5,
        posinf=1.0,
        neginf=0.0,
    )

    if track_valid is None:
        track_valid = finite
    else:
        if track_valid.shape != track_xy.shape[:2]:
            raise ValueError(
                "track_valid must have shape [B,T] matching track_xy"
            )
        track_valid = track_valid.bool() & finite

    if track_xy.shape[1] == target_length:
        return track_xy, track_valid

    # [B,T,2] -> [B,2,T] -> interpolate -> [B,T_audio,2]
    audio_track_xy = F.interpolate(
        track_xy.transpose(1, 2).float(),
        size=target_length,
        mode="linear",
        align_corners=True,
    ).transpose(1, 2)

    audio_track_valid = F.interpolate(
        track_valid.float().unsqueeze(1),
        size=target_length,
        mode="nearest",
    ).squeeze(1) > 0.5

    return audio_track_xy.to(track_xy.dtype), audio_track_valid


def build_track_features(
    track_xy: Tensor,
    track_valid: Tensor,
) -> tuple[Tensor, Tensor]:
    """
    构造绝对轨迹特征。

    Returns:
        features:
            [B,T,6] =
            [x, y, dx, dy, speed, valid]
        centered_xy:
            [B,T,2]，范围约为 [-1,1]
    """
    if track_xy.shape[:2] != track_valid.shape:
        raise ValueError("track_xy and track_valid shapes do not match")

    # [0,1] -> [-1,1]
    centered_xy = track_xy.float().clamp(0.0, 1.0) * 2.0 - 1.0

    delta_xy = torch.zeros_like(centered_xy)
    delta_xy[:, 1:] = centered_xy[:, 1:] - centered_xy[:, :-1]

    # 如果相邻任一位置无效，则对应速度也无效
    pair_valid = torch.zeros_like(track_valid)
    pair_valid[:, 0] = track_valid[:, 0]
    pair_valid[:, 1:] = track_valid[:, 1:] & track_valid[:, :-1]

    delta_xy = delta_xy * pair_valid.unsqueeze(-1).float()
    speed = torch.linalg.vector_norm(
        delta_xy,
        dim=-1,
        keepdim=True,
    )

    valid_float = track_valid.unsqueeze(-1).float()

    features = torch.cat(
        [
            centered_xy,
            delta_xy,
            speed,
            valid_float,
        ],
        dim=-1,
    )

    # 无效位置不提供条件
    features = features * valid_float
    centered_xy = centered_xy * valid_float

    return features, centered_xy


def build_track_projective_matrices(
    centered_xy: Tensor,
    track_valid: Tensor,
    matrix_scale: float,
) -> tuple[Tensor, Tensor]:
    """
    构造每个 Audio token 对应的 4x4 轨迹矩阵及其逆矩阵。

    P_t =
        [[1,0,0,beta*x_t],
         [0,1,0,beta*y_t],
         [0,0,1,0],
         [0,0,0,1]]

    Args:
        centered_xy:
            [B,T,2]
        track_valid:
            [B,T]

    Returns:
        matrices:
            [B,T,4,4]
        inverse_matrices:
            [B,T,4,4]
    """
    batch_size, seq_len, _ = centered_xy.shape
    device = centered_xy.device

    identity = torch.eye(
        4,
        device=device,
        dtype=torch.float32,
    ).view(1, 1, 4, 4)

    matrices = identity.expand(
        batch_size,
        seq_len,
        4,
        4,
    ).clone()

    inverse_matrices = matrices.clone()

    tx = centered_xy[..., 0].float() * matrix_scale
    ty = centered_xy[..., 1].float() * matrix_scale

    matrices[..., 0, 3] = tx
    matrices[..., 1, 3] = ty

    # 纯平移矩阵的逆变换可以直接写成负平移
    inverse_matrices[..., 0, 3] = -tx
    inverse_matrices[..., 1, 3] = -ty

    valid = track_valid[..., None, None]
    identity_full = identity.expand_as(matrices)

    # 无效位置使用单位矩阵
    matrices = torch.where(valid, matrices, identity_full)
    inverse_matrices = torch.where(
        valid,
        inverse_matrices,
        identity_full,
    )

    return matrices, inverse_matrices


def apply_per_token_4x4(
    hidden: Tensor,
    matrices: Tensor,
) -> Tensor:
    """
    将每个 token 的 4x4 矩阵重复应用到 attention head_dim。

    Args:
        hidden:
            [B,H,T,D]
        matrices:
            [B,T,4,4]

    要求:
        D % 4 == 0
    """
    batch_size, num_heads, seq_len, head_dim = hidden.shape

    if head_dim % 4 != 0:
        raise ValueError(
            f"Track-PRoPE head_dim must be divisible by 4, got {head_dim}"
        )

    if matrices.shape != (batch_size, seq_len, 4, 4):
        raise ValueError(
            f"Expected matrices {(batch_size, seq_len, 4, 4)}, "
            f"got {tuple(matrices.shape)}"
        )

    original_dtype = hidden.dtype

    # 小矩阵变换在 fp32 下更稳定
    hidden_fp32 = hidden.float().reshape(
        batch_size,
        num_heads,
        seq_len,
        head_dim // 4,
        4,
    )

    matrices_fp32 = matrices.float()

    output = torch.einsum(
        "btij,bhtpj->bhtpi",
        matrices_fp32,
        hidden_fp32,
    )

    return output.reshape(
        batch_size,
        num_heads,
        seq_len,
        head_dim,
    ).to(original_dtype)
