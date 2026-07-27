#!/usr/bin/env python3
"""
mask_vggt_integrated_3d_binaural_rms_matched_luavs.py

Create dynamic spatial stereo audio from a video, per-frame sounding-object
masks, and optional VGGT sequence geometry.

Recommended VGGT path
---------------------
VGGT jointly estimates depth maps, camera intrinsics and camera extrinsics.  Its
geometry is treated here as sequence-consistent *relative* geometry, not metric
metres.  This matters because acoustic RIR simulation needs physical distances.
The pipeline therefore separates two questions:

1) Geometric motion shape:
       sound mask centroid + VGGT depth + VGGT intrinsics/extrinsics
       -> relative 3-D source/listener trajectory.
2) Acoustic metric scale:
       one global scale calibration for the whole sequence
       -> trajectory in metres for gpuRIR rendering.

Never normalize VGGT depth independently per video frame.  Per-frame
normalization changes distance ratios over time and creates artificial Doppler /
level / reflection motion cues.  This script rejects predictions labeled as
per-frame normalized.

Scale policies (--vggt_scale_mode)
----------------------------------
relative_only:
    Exports relative 3-D trajectory only; audio rendering is deliberately
    blocked because there is no physical distance scale.
median_anchor (default):
    Applies one global factor so the median sounding-object distance equals
    --vggt_anchor_distance_m.  This is a controlled synthetic acoustic scale,
    not recovered ground-truth metric distance.
scale_factor:
    Applies one externally calibrated metres-per-VGGT-unit factor.  Use this
    when an external metric cue is available.
hook:
    Leaves a user interface for scene-specific external scale calibration.

Integrated VGGT inference path
------------------------------
- ``--vggt_provider hook`` now executes the official VGGT inference API inside
  this script: it loads sampled video frames, predicts depth/confidence and
  camera intrinsics/extrinsics, and supplies them directly to the 3-D acoustic
  trajectory builder.
- ``--vggt_model_path`` selects a local VGGT checkpoint directory or a
  Hugging Face model identifier.
- ``--vggt_sample_fps``/``--vggt_max_frames`` limit dense-video inference;
  geometry is estimated on sampled views and interpolated on the video time
  axis for dynamic audio rendering.
- ``--vggt_save_npz`` can cache the raw relative geometry for reproducibility.
- ``calibrate_vggt_scale_m_per_unit(...)`` remains an optional interface only
  for externally justified metric-scale calibration.


LU-AVS interval-mask mode
--------------------------
When ``--luavs_mask_npy`` is provided, the script reads a mask path such as::

    .../-vSNKf_nTJE__train_wheels_squealing__st__62__et__215/masks.npy

It parses the source video id and frame interval from the parent directory,
resolves ``<luavs_video_root>/<video_id>.mp4``, extracts only that interval
(including its corresponding audio preserved as lossless PCM before
spatialisation), and spatializes the resulting clip.  ``--luavs_et_policy auto`` infers inclusive/exclusive end convention
when the mask count makes this unambiguous; otherwise it follows the dataset
description and treats ``et`` as inclusive while marking the decision in
metadata.  If the mask count differs from extracted video frames, masks are
interpreted as sparse observations covering the full annotated interval and
are temporally resampled rather than incorrectly attached only to clip start.

RMS-matched stereo rendering
----------------------------
After gpuRIR renders the two-ear signal, the output can sound quieter than the
original mono source.  By default this script matches the rendered stereo
energy-RMS to the source mono RMS using one *shared* gain for both channels.
The left/right ratio, interaural level difference (ILD) and interaural timing
structure (ITD) are therefore not altered by independent channel
normalisation.  When the requested gain would clip, a stereo-linked limiter
uses the same instantaneous attenuation for L and R; this preserves spatial
channel balance while transparently protecting peaks.  All achieved RMS and
peak values are written to metadata for audit.

Coordinate conventions
----------------------
VGGT/OpenCV input convention:
    +x image-right, +y image-down, +z camera-forward.
Acoustic/gpuRIR convention used for exported/rendered trajectories:
    +X forward, +Y listener/image-left, +Z up.
VGGT official camera extrinsics are expected as camera-from-world matrices and
are inverted/converted inside this file.

Examples
--------
# A. Reproducible VGGT outputs; controlled synthetic metric scale for RIR audio
python mask_vggt_integrated_3d_binaural.py \
  --video input.mp4 --mask masks --output vggt_audio.mp4 \
  --use_vggt --vggt_provider npz --vggt_npz vggt_outputs.npz \
  --vggt_scale_mode median_anchor --vggt_anchor_distance_m 2.0 \
  --room_size 10 10 4 --rir_hz 12 --save_overlay

# B. Externally calibrated metric scale; strongest basis for metric claims
python mask_vggt_integrated_3d_binaural.py \
  --video input.mp4 --mask masks --output vggt_metric_audio.mp4 \
  --use_vggt --vggt_provider npz --vggt_npz vggt_outputs.npz \
  --vggt_scale_mode scale_factor --vggt_scale_factor_m_per_unit 1.37 \
  --room_size 10 10 4 --rir_hz 12

# C. Relative-trajectory audit only; no gpuRIR required
python mask_vggt_integrated_3d_binaural.py \
  --video input.mp4 --mask masks --output audit.mp4 \
  --use_vggt --vggt_provider npz --vggt_npz vggt_outputs.npz \
  --vggt_scale_mode relative_only --trajectory_only --save_overlay

Dependencies
------------
  pip install numpy opencv-python soundfile tqdm
  ffmpeg must be on PATH.
  gpuRIR is required for audio rendering, but not with --trajectory_only.
"""

from __future__ import annotations
import os
import argparse
import csv
import json
import math
import shutil
import subprocess
import tempfile
from contextlib import nullcontext
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional, Union

import cv2
import numpy as np
import soundfile as sf
from tqdm import tqdm

import torch
import yaml
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.models.vggt import VGGT

import gpuRIR


# =============================================================================
# Generic IO
# =============================================================================


def run_cmd(cmd: list[str]) -> None:
    # print(" ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


@dataclass
class VideoInfo:
    fps: float
    total_frames: int
    width: int
    height: int
    duration: float


def get_video_info(video_path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if fps <= 0 or total_frames <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"Invalid video metadata for {video_path}")
    return VideoInfo(fps, total_frames, width, height, total_frames / fps)


def _parse_named_integer(tokens: list[str], marker: str, parent_name: str) -> int:
    try:
        marker_index = tokens.index(marker)
        return int(tokens[marker_index + 1])
    except (ValueError, IndexError) as exc:
        raise ValueError(
            f"Cannot parse '__{marker}__<frame>' from LU-AVS mask parent directory: {parent_name}"
        ) from exc



def extract_mono_audio_from_video(video_path: Path, output_wav: Path) -> None:
    """Extract the first audio stream and downmix to one source signal."""
    run_cmd([
        "ffmpeg", "-y", "-loglevel", "quiet", "-i", str(video_path), "-map", "0:a:0", "-vn",
        "-ac", "1", "-c:a", "pcm_f32le", str(output_wav),
    ])


def read_audio_as_mono(audio_path: Path) -> tuple[np.ndarray, int]:
    audio, fs = sf.read(str(audio_path), dtype="float32", always_2d=True)
    mono = np.mean(audio, axis=1).astype(np.float32)
    peak = float(np.max(np.abs(mono)) + 1e-8)
    if peak > 0.99:
        mono = mono / peak * 0.99
    return mono, int(fs)


def mux_video_and_audio(video_path: Path, audio_path: Path, output_path: Path) -> None:
    run_cmd([
        "ffmpeg", "-y", "-loglevel", "quiet", "-i", str(video_path), "-i", str(audio_path),
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
        "-b:a", "192k", "-shortest", str(output_path),
    ])


# =============================================================================
# Mask loading and mask-centroid trajectory
# =============================================================================


@dataclass
class MaskTrack:
    timestamps: np.ndarray
    center_x: np.ndarray
    center_y: np.ndarray
    area_ratio: np.ndarray
    valid: np.ndarray
    binary_masks: Optional[list[np.ndarray]] = None  # resized to video size when kept


def sorted_mask_images(mask_dir: Path) -> list[Path]:
    suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    paths = sorted(p for p in mask_dir.iterdir() if p.suffix.lower() in suffixes)
    if not paths:
        raise FileNotFoundError(f"No mask images found in: {mask_dir}")
    return paths


def iter_mask_frames(mask_source: Path) -> tuple[Iterator[np.ndarray], Optional[float], Optional[int]]:
    if mask_source.is_dir():
        paths = sorted_mask_images(mask_source)
        def image_iter() -> Iterator[np.ndarray]:
            for path in paths:
                image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if image is None:
                    raise RuntimeError(f"Cannot read mask image: {path}")
                yield image

        return image_iter(), None, len(paths)

    if mask_source.suffix.lower() == ".npy":
        arr = np.load(str(mask_source), mmap_mode="r")
        if arr.ndim not in (3, 4):
            raise ValueError(f"Mask .npy must be [T,H,W] or [T,H,W,C], got {arr.shape}")

        def npy_iter() -> Iterator[np.ndarray]:
            for i in range(arr.shape[0]):
                yield np.asarray(arr[i])

        return npy_iter(), None, int(arr.shape[0])

    cap = cv2.VideoCapture(str(mask_source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open mask source: {mask_source}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def video_iter() -> Iterator[np.ndarray]:
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame
        finally:
            cap.release()

    return video_iter(), fps if fps > 0 else None, count or None


def binarize_mask(mask: np.ndarray, threshold: int) -> np.ndarray:
    # Bool 类型 mask 优先处理：直接视为前景/背景二值，无需走 BGR→灰度或阈值化分支。
    # 兼容 [H,W] 与 [H,W,C]（多通道时按任一通道为 True 即认为前景）。
    if mask.dtype == np.bool_:
        if mask.ndim == 3:
            binary = np.any(mask, axis=-1)
        elif mask.ndim == 2:
            binary = mask
        else:
            raise ValueError(f"Bool mask must be 2D or 3D, got shape {mask.shape}")
        return binary.astype(np.uint8)

    if mask.ndim == 3:
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        gray = mask
    # LU-AVS masks.npy is commonly saved as uint8 values in {0, 1};
    # treat these as binary foreground instead of thresholding 1 against 127.
    max_value = float(np.nanmax(gray)) if gray.size else 0.0
    if max_value <= 1.0:
        gray = (gray.astype(np.float32) * 255.0).astype(np.uint8)
    elif gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return (gray >= threshold).astype(np.uint8)


def mask_centroid_and_area(
    binary: np.ndarray,
    largest_component: bool = False,
    min_inset_px: float = 1.5,
) -> tuple[float, float, float] | None:
    """Locate one representative centre for the whole foreground mask.

    Logic (matches the requested specification):
      1) Compute the global centroid of ALL foreground pixels (treating every
         connected component as one set).  This is equivalent to ``cv2.moments``
         over the full binary mask.
      2) If that centroid already lies inside the foreground AND is at least
         ``min_inset_px`` pixels away from the boundary, return it as-is.
      3) Otherwise, snap the centroid to the nearest foreground pixel that is
         also at least ``min_inset_px`` pixels away from the boundary so it
         never touches the mask edge.  When the mask is thinner than the inset
         everywhere, fall back to the foreground pixel closest to the centroid
         (best effort).

    The ``largest_component`` flag is preserved for backward compatibility but
    no longer changes the centroid logic; it is reduced to a hint and is
    accepted as-is so existing callers keep working.
    """
    if binary is None or binary.size == 0:
        return None

    binary_u8 = (np.asarray(binary) > 0).astype(np.uint8)
    h, w = binary_u8.shape[:2]
    image_area = float(h * w)
    if image_area <= 0:
        return None

    # Step 1: global centroid over ALL foreground pixels.
    moments = cv2.moments(binary_u8, binaryImage=True)
    fg_area = float(moments["m00"])
    if fg_area <= 0:
        return None
    cx = float(moments["m10"] / fg_area)
    cy = float(moments["m01"] / fg_area)
    area_ratio = fg_area / image_area

    # Step 2: distance transform of the whole mask. dist[y, x] is the distance
    # from foreground pixel (x, y) to the nearest background pixel; it is 0 on
    # the boundary and on background. We use it to enforce the inset margin.
    dist_full = cv2.distanceTransform(binary_u8, cv2.DIST_L2, 5)

    ix = int(round(cx))
    iy = int(round(cy))
    if 0 <= ix < w and 0 <= iy < h and binary_u8[iy, ix] != 0:
        if float(dist_full[iy, ix]) >= float(min_inset_px):
            # Already inside the mask and far enough from any edge.
            return float(cx), float(cy), float(area_ratio)

    # Step 3: centroid is outside the mask, or too close to the edge.
    # Snap to the nearest foreground pixel that still satisfies the inset.
    eligible = dist_full >= float(min_inset_px)
    ys, xs = np.nonzero(eligible)
    if ys.size == 0:
        # No foreground pixel is thick enough; relax the inset and use any
        # foreground pixel closest to the centroid.
        ys, xs = np.nonzero(binary_u8)
        if ys.size == 0:
            return None
    dx = xs.astype(np.float64) - cx
    dy = ys.astype(np.float64) - cy
    best = int(np.argmin(dx * dx + dy * dy))
    return float(xs[best]), float(ys[best]), float(area_ratio)


def read_mask_track(
    mask_source: Path,
    video_info: VideoInfo,
    threshold: int,
    largest_component: bool,
    mask_fps_override: Optional[float],
    keep_binary_masks: bool,
    map_masks_over_entire_video_span: bool = False,
) -> MaskTrack:
    frames, source_fps, expected_count = iter_mask_frames(mask_source)
    source_fps = float(mask_fps_override or source_fps or video_info.fps)
    xs: list[float] = []
    ys: list[float] = []
    areas: list[float] = []
    valid: list[bool] = []
    kept_masks: Optional[list[np.ndarray]] = [] if keep_binary_masks else None

    # for mask in tqdm(frames, total=expected_count, desc="Reading masks"):
    for mask in frames:
        if mask.ndim != 2:
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask[0]
            else:
                assert False

        if np.sum(mask) == 0:
            continue
        binary = binarize_mask(mask, threshold)
        if binary.shape != (video_info.height, video_info.width):
            binary_video = cv2.resize(
                binary, (video_info.width, video_info.height), interpolation=cv2.INTER_NEAREST
            )
        else:
            binary_video = binary
        if kept_masks is not None:
            kept_masks.append(binary_video)

        result = mask_centroid_and_area(binary_video, largest_component)
        if result is None:
            xs.append(float("nan"))
            ys.append(float("nan"))
            areas.append(float("nan"))
            valid.append(False)
        else:
            cx, cy, area = result
            xs.append(cx)
            ys.append(cy)
            areas.append(area)
            valid.append(True)

    if not xs:
        raise ValueError("No mask frames were read")
    valid_arr = np.asarray(valid, dtype=bool)
    if not np.any(valid_arr):
        raise ValueError("Every mask frame is empty; no sounding-object trajectory can be obtained")

    if map_masks_over_entire_video_span:
        # In LU-AVS mode, masks may be sparse annotations over the selected
        # segment.  Span them across the whole local clip interval so the final
        # mask constrains the final selected video frame rather than ending early.
        if len(xs) == 1:
            timestamps = np.array([0.0], dtype=np.float32)
        else:
            final_frame_time = max((video_info.total_frames - 1) / video_info.fps, 0.0)
            timestamps = np.linspace(0.0, final_frame_time, len(xs), dtype=np.float32)
    else:
        timestamps = np.arange(len(xs), dtype=np.float32) / source_fps
    return MaskTrack(
        timestamps=timestamps,
        center_x=np.asarray(xs, dtype=np.float32),
        center_y=np.asarray(ys, dtype=np.float32),
        area_ratio=np.asarray(areas, dtype=np.float32),
        valid=valid_arr,
        binary_masks=kept_masks,
    )


def interpolate_missing(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    idx = np.arange(len(values), dtype=np.float32)
    good_idx = idx[valid]
    good_val = values[valid]
    if len(good_idx) == 1:
        return np.full_like(values, good_val[0], dtype=np.float32)
    return np.interp(idx, good_idx, good_val).astype(np.float32)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.astype(np.float32)
    window = int(window)
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def resample_values(source_t: np.ndarray, values: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    return np.interp(target_t, source_t, values).astype(np.float32)


def resample_mask_geometry_to_video_frames(
    track: MaskTrack, info: VideoInfo, smooth_frames: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = interpolate_missing(track.center_x, track.valid)
    y = interpolate_missing(track.center_y, track.valid)
    area = interpolate_missing(track.area_ratio, track.valid)
    frame_t = np.arange(info.total_frames, dtype=np.float32) / info.fps
    x = moving_average(resample_values(track.timestamps, x, frame_t), smooth_frames)
    y = moving_average(resample_values(track.timestamps, y, frame_t), smooth_frames)
    area = np.maximum(moving_average(resample_values(track.timestamps, area, frame_t), smooth_frames), 1e-8)
    return frame_t, x, y, area


# =============================================================================
# Depth interface: user implementation point + a testable .npy provider
# =============================================================================


DepthReturn = Union[float, np.floating, np.ndarray, None]


def estimate_depth_for_sounding_object_m(
    frame_bgr: np.ndarray,
    mask_binary: np.ndarray,
    centroid_xy: tuple[float, float],
    frame_index: int,
    timestamp_s: float,
) -> DepthReturn:
    """USER DEPTH INTERFACE: implement this function when --depth_provider hook is used.

    Parameters
    ----------
    frame_bgr:
        Video frame aligned with this mask, OpenCV BGR layout.
    mask_binary:
        Binary sounding-object mask resized to the video frame size, values 0/1.
    centroid_xy:
        Sound-source mask centroid in video pixels.
    frame_index, timestamp_s:
        Index/time of the mask observation.

    Returns
    -------
    Either:
        scalar float: metric source depth in metres, or
        np.ndarray [H,W]: metric depth map in metres; this file will select the
        valid median depth inside mask_binary.

    Important
    ---------
    Relative monocular depth is insufficient for physically scaled RIR rendering
    unless you calibrate it to metres. Return metric depth or explicitly accept
    that the generated distances are synthetic.
    """
    raise NotImplementedError(
        "Implement estimate_depth_for_sounding_object_m() or use "
        "--depth_provider npy --depth_npy <metric_depth_file.npy>."
    )


def object_depth_from_result(
    result: DepthReturn,
    mask_binary: np.ndarray,
    depth_scale: float,
    min_depth: float,
    max_depth: float,
) -> float | None:
    if result is None:
        return None
    arr = np.asarray(result)
    if arr.ndim == 0:
        value = float(arr) * depth_scale
        return value if np.isfinite(value) and min_depth <= value <= max_depth else None
    if arr.ndim != 2:
        raise ValueError(f"Depth hook/provider must return a scalar or [H,W] depth map, got {arr.shape}")
    if arr.shape != mask_binary.shape:
        mask = cv2.resize(
            mask_binary, (arr.shape[1], arr.shape[0]), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
    else:
        mask = mask_binary.astype(bool)
    depth = arr.astype(np.float32) * float(depth_scale)
    values = depth[mask & np.isfinite(depth) & (depth > 0)]
    values = values[(values >= min_depth) & (values <= max_depth)]
    if values.size == 0:
        return None
    return float(np.median(values))


class DepthProvider(ABC):
    @abstractmethod
    def source_depth_m(
        self,
        observation_index: int,
        timestamp_s: float,
        frame_bgr: np.ndarray,
        mask_binary: np.ndarray,
        centroid_xy: tuple[float, float],
    ) -> float | None:
        raise NotImplementedError


class UserHookDepthProvider(DepthProvider):
    def __init__(self, depth_scale: float, min_depth: float, max_depth: float):
        self.depth_scale = depth_scale
        self.min_depth = min_depth
        self.max_depth = max_depth

    def source_depth_m(
        self,
        observation_index: int,
        timestamp_s: float,
        frame_bgr: np.ndarray,
        mask_binary: np.ndarray,
        centroid_xy: tuple[float, float],
    ) -> float | None:
        result = estimate_depth_for_sounding_object_m(
            frame_bgr, mask_binary, centroid_xy, observation_index, timestamp_s
        )
        return object_depth_from_result(
            result, mask_binary, self.depth_scale, self.min_depth, self.max_depth
        )


class NpyDepthProvider(DepthProvider):
    def __init__(self, depth_npy: Path, depth_scale: float, min_depth: float, max_depth: float):
        self.depth = np.load(str(depth_npy), mmap_mode="r")
        if self.depth.ndim not in (1, 3):
            raise ValueError(f"Depth .npy must have shape [T] or [T,H,W], got {self.depth.shape}")
        self.depth_scale = depth_scale
        self.min_depth = min_depth
        self.max_depth = max_depth

    def source_depth_m(
        self,
        observation_index: int,
        timestamp_s: float,
        frame_bgr: np.ndarray,
        mask_binary: np.ndarray,
        centroid_xy: tuple[float, float],
    ) -> float | None:
        idx = min(observation_index, self.depth.shape[0] - 1)
        return object_depth_from_result(
            self.depth[idx], mask_binary, self.depth_scale, self.min_depth, self.max_depth
        )


def create_depth_provider(args: argparse.Namespace) -> DepthProvider:
    if args.depth_provider == "hook":
        return UserHookDepthProvider(args.depth_scale, args.min_depth, args.max_depth)
    if args.depth_provider == "npy":
        if not args.depth_npy:
            raise ValueError("--depth_provider npy requires --depth_npy")
        return NpyDepthProvider(Path(args.depth_npy), args.depth_scale, args.min_depth, args.max_depth)
    raise ValueError(f"Unknown depth provider: {args.depth_provider}")


def sample_video_frame_at_time(cap: cv2.VideoCapture, timestamp_s: float) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_MSEC, float(timestamp_s) * 1000.0)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Could not read input video frame at {timestamp_s:.3f}s for depth estimation")
    return frame


def compute_metric_depth_track(
    video_path: Path,
    track: MaskTrack,
    provider: DepthProvider,
    smooth_frames: int,
) -> np.ndarray:
    if track.binary_masks is None:
        raise ValueError("Depth computation requires retained binary masks")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for depth estimation: {video_path}")
    depth_values: list[float] = []
    depth_valid: list[bool] = []
    try:
        for i, timestamp_s in enumerate(tqdm(track.timestamps, desc="Estimating/reading depth")):
            if not track.valid[i]:
                depth_values.append(float("nan"))
                depth_valid.append(False)
                continue
            frame = sample_video_frame_at_time(cap, float(timestamp_s))
            value = provider.source_depth_m(
                i,
                float(timestamp_s),
                frame,
                track.binary_masks[i],
                (float(track.center_x[i]), float(track.center_y[i])),
            )
            if value is None or not np.isfinite(value) or value <= 0:
                depth_values.append(float("nan"))
                depth_valid.append(False)
            else:
                depth_values.append(float(value))
                depth_valid.append(True)
    finally:
        cap.release()

    valid = np.asarray(depth_valid, dtype=bool)
    if not np.any(valid):
        raise ValueError("No valid metric depths were produced for non-empty masks")
    if float(np.mean(valid)) < 0.8:
        print("WARNING: More than 20% of mask observations have missing/invalid depth; values are interpolated.")
    values = interpolate_missing(np.asarray(depth_values, dtype=np.float32), valid)
    return moving_average(values, smooth_frames)


# =============================================================================
# Pinhole geometry: pixels + depth -> camera-relative 3-D positions
# =============================================================================


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_args(cls, args: argparse.Namespace, info: VideoInfo) -> "CameraIntrinsics":
        if args.fx is not None or args.fy is not None:
            if args.fx is None or args.fy is None:
                raise ValueError("Provide both --fx and --fy, or neither")
            return cls(
                float(args.fx), float(args.fy),
                float(args.cx if args.cx is not None else info.width / 2.0),
                float(args.cy if args.cy is not None else info.height / 2.0),
            )
        if not (0.0 < args.hfov < 180.0 and 0.0 < args.vfov < 180.0):
            raise ValueError("--hfov and --vfov must be within (0,180) degrees")
        fx = 0.5 * info.width / math.tan(math.radians(args.hfov) / 2.0)
        fy = 0.5 * info.height / math.tan(math.radians(args.vfov) / 2.0)
        return cls(fx, fy, info.width / 2.0, info.height / 2.0)


def pixel_depth_to_camera_xyz(
    center_x: np.ndarray,
    center_y: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics,
    depth_type: str,
) -> np.ndarray:
    """Back-project a sounding-object point into [front, left, up] metres.

    depth_type='axial': depth_m is distance along camera forward axis (+X).
    depth_type='range': depth_m is Euclidean distance from camera/head centre.
    """
    y_left = -(center_x - intrinsics.cx) / intrinsics.fx
    z_up = -(center_y - intrinsics.cy) / intrinsics.fy
    rays = np.stack([np.ones_like(y_left), y_left, z_up], axis=-1).astype(np.float32)
    if depth_type == "axial":
        return rays * depth_m[:, None]
    if depth_type == "range":
        norm = np.linalg.norm(rays, axis=1, keepdims=True)
        return rays / np.maximum(norm, 1e-8) * depth_m[:, None]
    raise ValueError(f"Unknown depth_type: {depth_type}")


def xyz_to_spherical_track(source_relative_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = source_relative_xyz[:, 0]
    y = source_relative_xyz[:, 1]
    z = source_relative_xyz[:, 2]
    distance = np.linalg.norm(source_relative_xyz, axis=1).astype(np.float32)
    azimuth = np.degrees(np.arctan2(y, x)).astype(np.float32)
    elevation = np.degrees(np.arctan2(z, np.sqrt(x * x + y * y))).astype(np.float32)
    return azimuth, elevation, distance



# =============================================================================
# VGGT relative-geometry interface and defensible metric-scale anchoring
# =============================================================================


@dataclass
class VGGTSequencePrediction:
    """Raw sequence-level VGGT geometry outputs.

    The interface deliberately requests one sequence-consistent prediction, not
    a separately normalized depth map per frame.  Raw VGGT depth/camera outputs
    share a coordinate scale within the processed sequence but do not establish
    physical metres on their own.

    Conventions
    -----------
    depth_native:
        [N,Hd,Wd] positive axial depth values in VGGT's native/relative unit.
    intrinsics:
        [N,3,3] intrinsic matrices in either depth-map pixels or original-video
        pixels, selected by --vggt_intrinsics_space.
    extrinsics_cam_from_world:
        Optional [N,3,4] or [N,4,4] OpenCV convention matrices mapping points
        in the VGGT world/first-camera frame into each camera frame.
    depth_confidence:
        Optional [N,Hd,Wd] confidence maps; higher values should mean more
        reliable depth estimates.
    timestamps:
        Optional [N] seconds corresponding to the outputs.  When absent, N must
        equal the number of mask observations.
    normalization:
        Describe preprocessing applied outside this script.  Values indicating
        per-frame normalization are rejected because they destroy metric-ratio
        motion cues required for dynamic acoustic rendering.
    """

    depth_native: np.ndarray
    intrinsics: np.ndarray
    extrinsics_cam_from_world: Optional[np.ndarray] = None
    depth_confidence: Optional[np.ndarray] = None
    timestamps: Optional[np.ndarray] = None
    normalization: str = "raw_sequence_relative"

    def validated(self) -> "VGGTSequencePrediction":
        depth = np.asarray(self.depth_native, dtype=np.float32)
        if depth.ndim == 4 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.ndim != 3:
            raise ValueError(f"VGGT depth must be [N,H,W], got {depth.shape}")
        if not np.all(np.isfinite(depth[np.isfinite(depth)])):
            raise ValueError("VGGT depth contains unexpected non-finite values")

        intr = np.asarray(self.intrinsics, dtype=np.float32)
        if intr.ndim == 2 and intr.shape == (3, 3):
            intr = np.repeat(intr[None], depth.shape[0], axis=0)
        if intr.shape != (depth.shape[0], 3, 3):
            raise ValueError(f"VGGT intrinsics must be [N,3,3], got {intr.shape}")

        ext = None
        if self.extrinsics_cam_from_world is not None:
            ext = np.asarray(self.extrinsics_cam_from_world, dtype=np.float32)
            if ext.ndim == 2:
                ext = np.repeat(ext[None], depth.shape[0], axis=0)
            if ext.shape[1:] == (4, 4):
                ext = ext[:, :3, :4]
            if ext.shape != (depth.shape[0], 3, 4):
                raise ValueError(f"VGGT extrinsics must be [N,3,4] or [N,4,4], got {ext.shape}")

        conf = None
        if self.depth_confidence is not None:
            conf = np.asarray(self.depth_confidence, dtype=np.float32)
            if conf.ndim == 4 and conf.shape[-1] == 1:
                conf = conf[..., 0]
            if conf.shape != depth.shape:
                raise ValueError(f"VGGT depth confidence must match depth shape, got {conf.shape}")

        times = None
        if self.timestamps is not None:
            times = np.asarray(self.timestamps, dtype=np.float32).reshape(-1)
            if times.shape[0] != depth.shape[0]:
                raise ValueError("VGGT timestamps length must equal the number of predicted frames")
            if np.any(np.diff(times) <= 0):
                raise ValueError("VGGT timestamps must be strictly increasing")

        norm = str(self.normalization).lower()
        prohibited = ("per_frame", "per-frame", "framewise", "frame_wise", "minmax_each")
        if any(tag in norm for tag in prohibited):
            raise ValueError(
                "Per-frame-normalized VGGT depth is invalid for acoustic motion rendering. "
                "Return raw sequence-level predictions and apply only one global scale calibration."
            )
        return VGGTSequencePrediction(depth, intr, ext, conf, times, self.normalization)



DEFAULT_VGGT_MODEL_PATH = (
    "/data/vjuicefs_ai_camera_jgroup_video/public_data/Video_Data/VideoAudioData/"
    "VideoAudioModels/VGGT-1B"
)
_VGGT_MODEL_CACHE: dict[tuple[str, str], Any] = {}


def _tensor_to_numpy(value: Any) -> Optional[np.ndarray]:
    """Detach torch tensors lazily without importing torch on NPZ-only runs."""
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _squeeze_vggt_sequence_output(value: Any, expected_frames: int, kind: str) -> Optional[np.ndarray]:
    """Convert common VGGT batched output shapes into per-sequence numpy arrays."""
    arr = _tensor_to_numpy(value)
    if arr is None:
        return None
    if kind in {"depth", "confidence"}:
        # Official forward output is commonly [1,N,H,W,1] or [1,N,H,W].
        if arr.ndim == 5 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 4 and arr.shape[0] == 1 and arr.shape[1] == expected_frames and arr.shape[-1] != 1:
            arr = arr[0]
    elif kind in {"intrinsic", "extrinsic"}:
        # pose decoder output is commonly [1,N,3,3]/[1,N,3,4].
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
    return np.asarray(arr, dtype=np.float32)


def _resolve_vggt_device_and_dtype(torch: Any, requested_device: str) -> tuple[Any, Any]:
    if requested_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested_device)
    if device.type == "cuda":
        capability = torch.cuda.get_device_capability(device)
        dtype = torch.bfloat16 if capability[0] >= 8 else torch.float16
    else:
        dtype = torch.float32
    return device, dtype


def _load_cached_vggt_model(model_path: str, device: Any) -> Any:
    cache_key = (str(model_path), str(device))
    if cache_key not in _VGGT_MODEL_CACHE:
        print(f"Loading VGGT model: {model_path} on {device}")
        _VGGT_MODEL_CACHE[cache_key] = VGGT.from_pretrained(model_path).to(device).eval()
    return _VGGT_MODEL_CACHE[cache_key]


def _export_vggt_prediction_npz(path: Path, prediction: "VGGTSequencePrediction") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "depth_native": prediction.depth_native.astype(np.float32),
        "intrinsics": prediction.intrinsics.astype(np.float32),
        "timestamps": np.asarray(prediction.timestamps, dtype=np.float32),
        "normalization": np.asarray(prediction.normalization),
    }
    if prediction.extrinsics_cam_from_world is not None:
        payload["extrinsics_cam_from_world"] = prediction.extrinsics_cam_from_world.astype(np.float32)
    if prediction.depth_confidence is not None:
        payload["depth_confidence"] = prediction.depth_confidence.astype(np.float32)
    np.savez_compressed(str(path), **payload)
    print(f"Saved raw VGGT prediction cache: {path}")


def run_vggt_geometry_estimator(
    frames_bgr: list[np.ndarray],
    timestamps_s: np.ndarray,
    sounding_masks_binary: list[np.ndarray],
    args: argparse.Namespace,
) -> "VGGTSequencePrediction":
    """Run the user's VGGT prediction code on sampled video views.

    This is the integrated implementation of the formerly empty VGGT hook.  It
    follows the official VGGT API used in the user's example: ``model(images)``
    predicts ``pose_enc``, ``depth`` and ``depth_conf``; camera matrices are
    decoded with ``pose_encoding_to_extri_intri``.  Dense unprojected point maps
    are optional diagnostics because the audio path only needs the masked source
    depth, intrinsics and camera motion.

    Important: depth is returned in VGGT native relative units without per-frame
    normalization.  The single global acoustic scale policy is applied later.
    """
    if not frames_bgr:
        raise ValueError("No video frames were selected for VGGT inference")
    if len(frames_bgr) != len(timestamps_s) or len(frames_bgr) != len(sounding_masks_binary):
        raise ValueError("VGGT frames, timestamps and sound masks must have identical lengths")

    device, dtype = _resolve_vggt_device_and_dtype(torch, args.vggt_device)
    model = _load_cached_vggt_model(args.vggt_model_path, device)

    # with tempfile.TemporaryDirectory(prefix="vggt_sampled_frames_") as frame_dir:
        
    frame_dir = Path(args.video).parent / "frames"
    os.makedirs(str(frame_dir), exist_ok=True)
    image_names: list[str] = []
    for i, frame_bgr in enumerate(frames_bgr):
        image_path = frame_dir / f"frame_{i:06d}.png"
        if not cv2.imwrite(str(image_path), frame_bgr):
            raise RuntimeError(f"Failed to write temporary VGGT frame: {image_path}")
        image_names.append(str(image_path))
    images = load_and_preprocess_images(image_names).to(device)

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=dtype)
        if device.type == "cuda" else nullcontext()
    )
    with torch.no_grad():
        with autocast_ctx:
            predictions = model(images)
            pose_enc = predictions["pose_enc"]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
            depth_map = predictions["depth"]
            depth_conf = predictions.get("depth_conf")

            # Optional diagnostic, mirroring the user's sample code. It is
            # disabled by default because dense point maps can be large and
            # are not required by the spatial-audio renderer.
            point_map_by_unprojection = None
            if args.vggt_save_unprojected_point_map:
                point_map_by_unprojection = unproject_depth_map_to_point_map(
                    depth_map.squeeze(0), extrinsic.squeeze(0), intrinsic.squeeze(0)
                )

    n = len(frames_bgr)
    prediction = VGGTSequencePrediction(
        depth_native=_squeeze_vggt_sequence_output(depth_map, n, "depth"),
        intrinsics=_squeeze_vggt_sequence_output(intrinsic, n, "intrinsic"),
        extrinsics_cam_from_world=_squeeze_vggt_sequence_output(extrinsic, n, "extrinsic"),
        depth_confidence=_squeeze_vggt_sequence_output(depth_conf, n, "confidence"),
        timestamps=np.asarray(timestamps_s, dtype=np.float32),
        normalization="raw_sequence_relative",
    ).validated()

    if args.vggt_save_npz:
        _export_vggt_prediction_npz(Path(args.vggt_save_npz), prediction)
    if args.vggt_save_unprojected_point_map and point_map_by_unprojection is not None:
        point_path = Path(args.vggt_save_unprojected_point_map)
        point_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(point_path), _tensor_to_numpy(point_map_by_unprojection).astype(np.float32))
        print(f"Saved VGGT depth-unprojected point map: {point_path}")
    return prediction


def select_vggt_observation_indices(
    track: MaskTrack, sample_fps: float, max_frames: int
) -> np.ndarray:
    """Select views for VGGT while maintaining coverage of the whole clip."""
    n = len(track.timestamps)
    if n == 0:
        raise ValueError("Mask track is empty")
    if sample_fps <= 0:
        indices = np.arange(n, dtype=np.int64)
    else:
        start_t, end_t = float(track.timestamps[0]), float(track.timestamps[-1])
        targets = np.arange(start_t, end_t + 0.5 / sample_fps, 1.0 / sample_fps, dtype=np.float32)
        indices = np.unique([
            int(np.argmin(np.abs(track.timestamps - t))) for t in targets
        ]).astype(np.int64)
        indices = np.unique(np.concatenate([indices, np.asarray([0, n - 1], dtype=np.int64)]))
    if max_frames > 0 and len(indices) > max_frames:
        keep = np.linspace(0, len(indices) - 1, max_frames).round().astype(np.int64)
        indices = np.unique(indices[keep])
    return indices


def subset_mask_track(track: MaskTrack, indices: np.ndarray) -> MaskTrack:
    masks = None if track.binary_masks is None else [track.binary_masks[int(i)] for i in indices]
    return MaskTrack(
        timestamps=track.timestamps[indices], center_x=track.center_x[indices],
        center_y=track.center_y[indices], area_ratio=track.area_ratio[indices],
        valid=track.valid[indices], binary_masks=masks,
    )

def _npz_first(data: Any, names: tuple[str, ...], required: bool = True) -> Optional[np.ndarray]:
    for name in names:
        if name in data:
            return np.asarray(data[name])
    if required:
        raise ValueError(f"VGGT .npz is missing one of required keys: {names}")
    return None


def load_vggt_npz(path: Path) -> VGGTSequencePrediction:
    """Load reproducible VGGT outputs exported once from the user's VGGT code.

    Accepted keys:
        depth_native or depth_map or depth               [N,H,W]
        intrinsics or intrinsic                          [N,3,3]
        extrinsics_cam_from_world or extrinsic/extrinsics optional [N,3,4]/[N,4,4]
        depth_confidence or depth_conf                   optional [N,H,W]
        timestamps or time_s                             optional [N]
        normalization                                    optional scalar string
    """
    data = np.load(str(path), allow_pickle=False)
    normalization = "raw_sequence_relative"
    if "normalization" in data:
        normalization = str(np.asarray(data["normalization"]).reshape(-1)[0])
    pred = VGGTSequencePrediction(
        depth_native=_npz_first(data, ("depth_native", "depth_map", "depth")),
        intrinsics=_npz_first(data, ("intrinsics", "intrinsic")),
        extrinsics_cam_from_world=_npz_first(
            data, ("extrinsics_cam_from_world", "extrinsic", "extrinsics"), required=False
        ),
        depth_confidence=_npz_first(data, ("depth_confidence", "depth_conf"), required=False),
        timestamps=_npz_first(data, ("timestamps", "time_s"), required=False),
        normalization=normalization,
    )
    return pred.validated()


def collect_video_frames_for_mask_observations(video_path: Path, track: MaskTrack) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for VGGT sampling: {video_path}")
    frames: list[np.ndarray] = []
    try:
        # for t in tqdm(track.timestamps, desc="Reading VGGT input frames"):
        for t in track.timestamps:
            frames.append(sample_video_frame_at_time(cap, float(t)))
    finally:
        cap.release()
    return frames


def acquire_vggt_prediction(video_path: Path, track: MaskTrack, args: argparse.Namespace) -> VGGTSequencePrediction:
    frames = collect_video_frames_for_mask_observations(video_path, track)

    prediction = run_vggt_geometry_estimator(frames, track.timestamps, track.binary_masks, args).validated()

    if prediction.timestamps is None and prediction.depth_native.shape[0] != len(track.timestamps):
        raise ValueError(
            "VGGT output count does not match mask observations; provide VGGT timestamps in the .npz/hook output."
        )
        
    return prediction


def vggt_index_for_observations(prediction: VGGTSequencePrediction, observation_t: np.ndarray) -> np.ndarray:
    if prediction.timestamps is None:
        return np.arange(len(observation_t), dtype=np.int64)
    return np.asarray([
        int(np.argmin(np.abs(prediction.timestamps - t))) for t in observation_t
    ], dtype=np.int64)


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cutoff = 0.5 * float(np.sum(w))
    return float(v[min(int(np.searchsorted(np.cumsum(w), cutoff)), len(v) - 1)])


def robust_vggt_object_depth_native(
    depth_map: np.ndarray,
    confidence_map: Optional[np.ndarray],
    mask_binary_video: np.ndarray,
    mask_erode_px: int,
    conf_keep_quantile: float,
) -> float | None:
    """Estimate object axial depth in VGGT's native unit from a sound mask.

    Eroding the mask prevents object/background boundary pixels from introducing
    false distance changes.  If VGGT confidence is supplied, a confidence-gated
    weighted median is used rather than trusting all masked pixels equally.
    """
    h, w = depth_map.shape
    mask = cv2.resize(mask_binary_video, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    if mask_erode_px > 0:
        k = 2 * int(mask_erode_px) + 1
        eroded = cv2.erode(mask, np.ones((k, k), dtype=np.uint8), iterations=1)
        if np.count_nonzero(eroded) > 0:
            mask = eroded
    valid = mask.astype(bool) & np.isfinite(depth_map) & (depth_map > 0)
    if not np.any(valid):
        return None
    values = depth_map[valid].astype(np.float32)
    if confidence_map is None:
        return float(np.median(values))
    confidence = confidence_map[valid].astype(np.float32)
    finite = np.isfinite(confidence)
    values, confidence = values[finite], confidence[finite]
    if values.size == 0:
        return None
    q = float(np.clip(conf_keep_quantile, 0.0, 0.95))
    threshold = float(np.quantile(confidence, q))
    keep = confidence >= threshold
    values = values[keep]
    confidence = confidence[keep]
    weights = np.maximum(confidence - float(np.min(confidence)) + 1e-6, 1e-6)
    return weighted_median(values, weights)


def vggt_pixel_depth_to_camera_cv_native(
    center_x_video: np.ndarray,
    center_y_video: np.ndarray,
    object_depth_native: np.ndarray,
    prediction: VGGTSequencePrediction,
    pred_index: np.ndarray,
    video_info: VideoInfo,
    intrinsics_space: str,
) -> np.ndarray:
    """Unproject mask centroids to native VGGT camera coordinates (OpenCV axes).

    OpenCV camera axes: +x image-right, +y image-down, +z forward.
    VGGT depth is treated as axial +z depth, matching the camera projection.
    """
    xyz = np.empty((len(center_x_video), 3), dtype=np.float32)
    for i, j in enumerate(pred_index):
        K = prediction.intrinsics[j]
        if intrinsics_space == "depth":
            h, w = prediction.depth_native[j].shape
            u = float(center_x_video[i]) * w / float(video_info.width)
            v = float(center_y_video[i]) * h / float(video_info.height)
        elif intrinsics_space == "video":
            u, v = float(center_x_video[i]), float(center_y_video[i])
        else:
            raise ValueError(f"Unknown --vggt_intrinsics_space: {intrinsics_space}")
        z = float(object_depth_native[i])
        x = (u - float(K[0, 2])) / float(K[0, 0]) * z
        y = (v - float(K[1, 2])) / float(K[1, 1]) * z
        xyz[i] = [x, y, z]
    return xyz


def calibrate_vggt_scale_m_per_unit(
    object_ranges_native: np.ndarray,
    track: MaskTrack,
    prediction: VGGTSequencePrediction,
    args: argparse.Namespace,
) -> float:
    """USER SCALE INTERFACE for --vggt_scale_mode hook.

    Return a single positive metres-per-VGGT-unit factor.  A defensible custom
    implementation should derive this from an external metric cue, such as a
    calibrated known object size, stereo/LiDAR depth, measured camera baseline,
    or a scene-specific distance annotation.  Never infer a different factor
    independently for every frame.
    """
    raise NotImplementedError(
        "Implement calibrate_vggt_scale_m_per_unit() or choose median_anchor/scale_factor/relative_only."
    )


def resolve_vggt_global_scale(
    object_ranges_native: np.ndarray,
    track: MaskTrack,
    prediction: VGGTSequencePrediction,
    args: argparse.Namespace,
) -> tuple[float, str, str]:
    """Return one global scale and a transparent interpretation label."""
    finite = object_ranges_native[np.isfinite(object_ranges_native) & (object_ranges_native > 0)]
    if finite.size == 0:
        raise ValueError("No valid VGGT object ranges for scale calibration")
    mode = args.vggt_scale_mode
    if mode == "relative_only":
        if not args.trajectory_only:
            raise ValueError(
                "--vggt_scale_mode relative_only cannot drive gpuRIR audio, because RIR propagation "
                "requires metres. Use --trajectory_only, median_anchor, scale_factor, or hook."
            )
        return 1.0, "relative_units_only", "VGGT relative 3-D only; not metric and not acoustically rendered."
    if mode == "median_anchor":
        if args.vggt_anchor_distance_m <= 0:
            raise ValueError("--vggt_anchor_distance_m must be positive")
        scale = float(args.vggt_anchor_distance_m / np.median(finite))
        return (
            scale,
            "canonical_metric_anchor",
            "A single per-clip global scale maps the median sounding-object range to the declared canonical distance; physical motion ratios are preserved, but absolute distance is simulated rather than recovered.",
        )
    if mode == "scale_factor":
        if args.vggt_scale_factor_m_per_unit is None or args.vggt_scale_factor_m_per_unit <= 0:
            raise ValueError("--vggt_scale_mode scale_factor requires positive --vggt_scale_factor_m_per_unit")
        return (
            float(args.vggt_scale_factor_m_per_unit),
            "externally_calibrated_scale_factor",
            "A user-supplied single global metric scale was applied to sequence-consistent VGGT geometry.",
        )
    if mode == "hook":
        scale = float(calibrate_vggt_scale_m_per_unit(object_ranges_native, track, prediction, args))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Custom VGGT scale hook must return a positive finite metres-per-unit value")
        return (
            scale,
            "external_scale_hook",
            "A user-implemented external metric calibration supplied one global scale for VGGT geometry.",
        )
    raise ValueError(f"Unknown VGGT scale mode: {mode}")


def cv_camera_xyz_to_acoustic_camera_xyz(xyz_cv: np.ndarray) -> np.ndarray:
    """OpenCV [right, down, forward] -> audio body [forward, left, up]."""
    return np.stack([xyz_cv[:, 2], -xyz_cv[:, 0], -xyz_cv[:, 1]], axis=-1).astype(np.float32)


def construct_world_trajectory_from_vggt(
    source_cam_cv_native: np.ndarray,
    frame_t: np.ndarray,
    prediction: VGGTSequencePrediction,
    pred_index: np.ndarray,
    scale_m_per_unit: float,
    receiver_origin: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert VGGT OpenCV camera-from-world geometry to acoustic room metres.

    The initial VGGT camera origin is placed at receiver_origin.  World axes are
    rotated so the first camera faces +X and image-left is +Y in the acoustic
    room.  The *same* global scale is applied to sources and camera translations.
    """
    if prediction.extrinsics_cam_from_world is None:
        source_body_m = cv_camera_xyz_to_acoustic_camera_xyz(source_cam_cv_native * scale_m_per_unit)
        receiver = np.tile(np.asarray(receiver_origin, dtype=np.float32), (len(frame_t), 1))
        source_world = receiver + source_body_m
        left_to_right = np.tile(np.asarray([[0.0, -1.0, 0.0]], dtype=np.float32), (len(frame_t), 1))
        return source_world, receiver, left_to_right

    A = np.asarray([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float32)
    origin = np.asarray(receiver_origin, dtype=np.float32)
    src_world_native = np.empty_like(source_cam_cv_native, dtype=np.float32)
    cam_world_native = np.empty_like(source_cam_cv_native, dtype=np.float32)
    right_world_native = np.empty_like(source_cam_cv_native, dtype=np.float32)
    for i, j in enumerate(pred_index):
        ext = prediction.extrinsics_cam_from_world[j]
        R, t = ext[:, :3], ext[:, 3]
        cam_world_native[i] = -R.T @ t
        src_world_native[i] = R.T @ (source_cam_cv_native[i] - t)
        right_world_native[i] = R.T @ np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    first_cam = cam_world_native[0].copy()
    receiver = origin + ((cam_world_native - first_cam) * scale_m_per_unit) @ A.T
    source_world = origin + ((src_world_native - first_cam) * scale_m_per_unit) @ A.T
    left_to_right = right_world_native @ A.T
    left_to_right /= np.maximum(np.linalg.norm(left_to_right, axis=1, keepdims=True), 1e-8)
    return source_world.astype(np.float32), receiver.astype(np.float32), left_to_right.astype(np.float32)


def compute_vggt_geometry_at_mask_times(
    video_path: Path,
    track: MaskTrack,
    args: argparse.Namespace,
    info: VideoInfo,
) -> dict[str, Any]:
    
    selected = select_vggt_observation_indices(track, args.vggt_sample_fps, args.vggt_max_frames)
    # print(
    #     f"VGGT geometry views: {len(selected)}/{len(track.timestamps)} mask observations "
    #     f"(sample_fps={args.vggt_sample_fps:g}, max_frames={args.vggt_max_frames})"
    # )
        
    geometry_track = subset_mask_track(track, selected)
    prediction = acquire_vggt_prediction(video_path, geometry_track, args)
    pred_index = vggt_index_for_observations(prediction, geometry_track.timestamps)
    native_depth: list[float] = []
    valid_depth: list[bool] = []
    assert geometry_track.binary_masks is not None
    # for i, j in enumerate(tqdm(pred_index, desc="Extracting VGGT masked depths")):
    for i, j in enumerate(pred_index):
        if not geometry_track.valid[i]:
            native_depth.append(float("nan")); valid_depth.append(False); continue
        value = robust_vggt_object_depth_native(
            prediction.depth_native[j],
            prediction.depth_confidence[j] if prediction.depth_confidence is not None else None,
            geometry_track.binary_masks[i],
            args.vggt_mask_erode_px,
            args.vggt_conf_keep_quantile,
        )
        native_depth.append(float(value) if value is not None else float("nan"))
        valid_depth.append(value is not None and np.isfinite(value))
    depth_valid = np.asarray(valid_depth, dtype=bool)
    if not np.any(depth_valid):
        raise ValueError("No valid mask-region depth was recovered from VGGT outputs")
    if float(np.mean(depth_valid)) < 0.8:
        print("WARNING: More than 20% of mask observations lack valid VGGT depth; values are interpolated.")
    depth_native = moving_average(
        interpolate_missing(np.asarray(native_depth, dtype=np.float32), depth_valid),
        args.depth_smooth_frames,
    )
    source_cv_native = vggt_pixel_depth_to_camera_cv_native(
        geometry_track.center_x, geometry_track.center_y, depth_native, prediction, pred_index, info, args.vggt_intrinsics_space
    )
    range_native = np.linalg.norm(source_cv_native, axis=1).astype(np.float32)
    scale, scale_status, scale_note = resolve_vggt_global_scale(range_native, geometry_track, prediction, args)
    source_cv_m = source_cv_native * scale
    source_body_m = cv_camera_xyz_to_acoustic_camera_xyz(source_cv_m)
    return {
        "prediction": prediction,
        "pred_index": pred_index,
        "observation_t": geometry_track.timestamps,
        "sampled_mask_track": geometry_track,
        "depth_native": depth_native,
        "depth_metric": depth_native * scale,
        "source_cam_cv_native": source_cv_native,
        "source_cam_audio_m": source_body_m,
        "scale_m_per_native_unit": float(scale),
        "scale_status": scale_status,
        "scale_note": scale_note,
    }


def save_vggt_trajectory_csv(
    path: Path,
    frame_t: np.ndarray,
    center_x: np.ndarray,
    center_y: np.ndarray,
    area: np.ndarray,
    depth_native: np.ndarray,
    depth_metric: np.ndarray,
    source_cam_native: np.ndarray,
    source_cam_audio_m: np.ndarray,
    source_world_xyz: np.ndarray,
    receiver_xyz: np.ndarray,
    azimuth: np.ndarray,
    elevation: np.ndarray,
    distance: np.ndarray,
    scale_m_per_unit: float,
    scale_status: str,
) -> None:
    header = [
        "time_s", "center_x_px", "center_y_px", "area_ratio",
        "vggt_depth_native_unit", "scaled_axial_depth_m", "scale_m_per_vggt_unit", "scale_status",
        "source_cam_cv_x_right_native", "source_cam_cv_y_down_native", "source_cam_cv_z_forward_native",
        "source_cam_x_forward_m", "source_cam_y_left_m", "source_cam_z_up_m",
        "source_world_x_m", "source_world_y_m", "source_world_z_m",
        "receiver_world_x_m", "receiver_world_y_m", "receiver_world_z_m",
        "relative_azimuth_deg", "relative_elevation_deg", "relative_range_m",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(header)
        for i in range(len(frame_t)):
            writer.writerow([
                float(frame_t[i]), float(center_x[i]), float(center_y[i]), float(area[i]),
                float(depth_native[i]), float(depth_metric[i]), float(scale_m_per_unit), scale_status,
                *[float(v) for v in source_cam_native[i]], *[float(v) for v in source_cam_audio_m[i]],
                *[float(v) for v in source_world_xyz[i]], *[float(v) for v in receiver_xyz[i]],
                float(azimuth[i]), float(elevation[i]), float(distance[i]),
            ])


# =============================================================================
# Camera pose: camera-relative source trajectory -> room/world trajectory
# =============================================================================


@dataclass
class PoseTrack:
    timestamps: np.ndarray
    receiver_xyz: np.ndarray
    yaw_deg: np.ndarray
    pitch_deg: np.ndarray
    roll_deg: np.ndarray


def read_pose_csv(pose_csv: Path) -> PoseTrack:
    required = {
        "time_s", "receiver_x_m", "receiver_y_m", "receiver_z_m",
        "yaw_deg", "pitch_deg", "roll_deg",
    }
    rows: list[dict[str, str]] = []
    with pose_csv.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Pose CSV is missing required columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("Pose CSV is empty")
    t = np.asarray([float(r["time_s"]) for r in rows], dtype=np.float32)
    if np.any(np.diff(t) <= 0):
        raise ValueError("Pose CSV time_s values must be strictly increasing")
    xyz = np.asarray([
        [float(r["receiver_x_m"]), float(r["receiver_y_m"]), float(r["receiver_z_m"])]
        for r in rows
    ], dtype=np.float32)
    return PoseTrack(
        t, xyz,
        np.asarray([float(r["yaw_deg"]) for r in rows], dtype=np.float32),
        np.asarray([float(r["pitch_deg"]) for r in rows], dtype=np.float32),
        np.asarray([float(r["roll_deg"]) for r in rows], dtype=np.float32),
    )


def interpolate_angles_deg(source_t: np.ndarray, angles_deg: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    unwrapped = np.unwrap(np.deg2rad(angles_deg))
    return np.rad2deg(np.interp(target_t, source_t, unwrapped)).astype(np.float32)


def resample_pose(pose: PoseTrack, target_t: np.ndarray) -> PoseTrack:
    xyz = np.stack([
        np.interp(target_t, pose.timestamps, pose.receiver_xyz[:, axis]) for axis in range(3)
    ], axis=-1).astype(np.float32)
    return PoseTrack(
        target_t,
        xyz,
        interpolate_angles_deg(pose.timestamps, pose.yaw_deg, target_t),
        interpolate_angles_deg(pose.timestamps, pose.pitch_deg, target_t),
        interpolate_angles_deg(pose.timestamps, pose.roll_deg, target_t),
    )


def rotation_matrix_body_to_world(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Body [forward,left,up] to world [X,Y,Z].

    Positive yaw turns forward toward +Y (listener/camera left).
    Positive pitch turns forward upward. Positive roll rotates about forward.
    """
    yaw, pitch, roll = np.deg2rad([yaw_deg, pitch_deg, roll_deg])
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    rz = np.asarray([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    ry = np.asarray([[cp, 0.0, -sp], [0.0, 1.0, 0.0], [sp, 0.0, cp]], dtype=np.float32)
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float32)
    return rz @ ry @ rx


def construct_world_scene_trajectory(
    source_camera_xyz: np.ndarray,
    frame_t: np.ndarray,
    pose_csv: Optional[Path],
    receiver_origin: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Return source world xyz, receiver world xyz, and left-to-right direction.

    Without a pose CSV, the listener is assumed fixed at receiver_origin and the
    coordinate frame of every video frame is assumed unchanged.
    """
    if pose_csv is None:
        receiver = np.tile(np.asarray(receiver_origin, dtype=np.float32), (len(frame_t), 1))
        source_world = receiver + source_camera_xyz
        ear_direction = np.tile(np.asarray([[0.0, -1.0, 0.0]], dtype=np.float32), (len(frame_t), 1))
        return source_world, receiver, ear_direction, False

    pose = resample_pose(read_pose_csv(pose_csv), frame_t)
    source_world = np.empty_like(source_camera_xyz, dtype=np.float32)
    ear_direction = np.empty_like(source_camera_xyz, dtype=np.float32)
    camera_right = np.asarray([0.0, -1.0, 0.0], dtype=np.float32)  # left ear -> right ear
    for i in range(len(frame_t)):
        rot = rotation_matrix_body_to_world(pose.yaw_deg[i], pose.pitch_deg[i], pose.roll_deg[i])
        source_world[i] = pose.receiver_xyz[i] + rot @ source_camera_xyz[i]
        ear_direction[i] = rot @ camera_right
    return source_world, pose.receiver_xyz, ear_direction, True


def enforce_room_bounds(
    source_xyz: np.ndarray,
    receiver_xyz: np.ndarray,
    ear_direction: np.ndarray,
    room_size: tuple[float, float, float],
    ear_distance: float,
    policy: str,
    margin: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    room = np.asarray(room_size, dtype=np.float32)
    norm_dirs = ear_direction / np.maximum(np.linalg.norm(ear_direction, axis=1, keepdims=True), 1e-8)
    ears = np.concatenate([
        receiver_xyz - 0.5 * ear_distance * norm_dirs,
        receiver_xyz + 0.5 * ear_distance * norm_dirs,
    ], axis=0)

    def outside(xyz: np.ndarray) -> np.ndarray:
        return np.any((xyz <= margin) | (xyz >= room - margin), axis=1)

    src_bad = outside(source_xyz)
    ear_bad = outside(ears)
    if not np.any(src_bad) and not np.any(ear_bad):
        return source_xyz, receiver_xyz
    message = (
        f"3-D trajectory leaves the declared room: source frames outside={int(np.sum(src_bad))}, "
        f"ear positions outside={int(np.sum(ear_bad))}. Increase --room_size, provide a valid "
        f"--pose_csv, or use --out_of_room_policy clip."
    )
    if policy == "error":
        raise ValueError(message)
    print("WARNING:", message, "Coordinates are clipped; this changes physical trajectories.")
    source_xyz = np.clip(source_xyz, margin, room - margin)
    receiver_xyz = np.clip(receiver_xyz, margin + ear_distance, room - margin - ear_distance)
    return source_xyz, receiver_xyz


# =============================================================================
# Dynamic RIR audio renderer: 3-D source/receiver trajectories -> stereo audio
# =============================================================================


def safe_beta_sabine(room_sz: np.ndarray, t60: float) -> np.ndarray:
    try:
        return gpuRIR.beta_SabineEstimation(room_sz, t60, [1.0] * 6)
    except TypeError:
        return gpuRIR.beta_SabineEstimation(room_sz, t60)


def resample_vector_track(frame_t: np.ndarray, xyz: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    return np.stack([
        np.interp(target_t, frame_t, xyz[:, axis]) for axis in range(xyz.shape[1])
    ], axis=-1).astype(np.float32)


def downsample_scene_for_rir(
    frame_t: np.ndarray,
    source_xyz: np.ndarray,
    receiver_xyz: np.ndarray,
    ear_direction: np.ndarray,
    duration: float,
    rir_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if rir_hz <= 0:
        raise ValueError("--rir_hz must be positive")
    n = max(2, int(math.ceil(duration * rir_hz)))
    rir_t = np.linspace(0.0, duration, n, endpoint=False, dtype=np.float32)
    src = resample_vector_track(frame_t, source_xyz, rir_t)
    rcv = resample_vector_track(frame_t, receiver_xyz, rir_t)
    ears = resample_vector_track(frame_t, ear_direction, rir_t)
    ears = ears / np.maximum(np.linalg.norm(ears, axis=1, keepdims=True), 1e-8)
    return rir_t, src, rcv, ears


def simulate_dynamic_spatial_stereo(
    source_signal: np.ndarray,
    source_xyz: np.ndarray,
    receiver_xyz: np.ndarray,
    left_to_right_dirs: np.ndarray,
    room_size: tuple[float, float, float],
    t60: float,
    fs: int,
    mic_pattern: str,
    ear_distance: float,
    max_rir_seconds: float,
    normalize: bool = True,
) -> np.ndarray:
    """Render dynamic two-ear stereo using gpuRIR and overlap-add convolution."""
    if gpuRIR is None:
        raise ImportError("gpuRIR is required for audio rendering; use --trajectory_only to inspect 3-D tracks")
    signal = np.asarray(source_signal, dtype=np.float32).reshape(-1)
    if not (len(source_xyz) == len(receiver_xyz) == len(left_to_right_dirs)):
        raise ValueError("Source, receiver, and orientation trajectory lengths must match")
    directions = left_to_right_dirs.astype(np.float32)
    directions = directions / np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-8)
    right_ear = receiver_xyz + 0.5 * ear_distance * directions
    left_ear = receiver_xyz - 0.5 * ear_distance * directions

    room = np.asarray(room_size, dtype=np.float32)
    beta = safe_beta_sabine(room, t60)
    tdiff = float(gpuRIR.att2t_SabineEstimator(15.0, t60))
    tmax = float(gpuRIR.att2t_SabineEstimator(60.0, t60))
    tmax = min(tmax, float(max_rir_seconds))
    tdiff = min(tdiff, tmax)
    nb_img = gpuRIR.t2n(tdiff, room)

    rirs: list[np.ndarray] = []
    # for i in tqdm(range(len(source_xyz)), desc="Generating 3-D dynamic stereo RIRs"):
    for i in range(len(source_xyz)):
        pos_src = source_xyz[i].reshape(1, 3)
        pos_rcv = np.stack([left_ear[i], right_ear[i]], axis=0)
        kwargs: dict[str, object] = {"Tdiff": tdiff, "mic_pattern": mic_pattern}
        if mic_pattern != "omni":
            kwargs["orV_rcv"] = np.stack([-directions[i], directions[i]], axis=0)
        rir = gpuRIR.simulateRIR(room, beta, pos_src, pos_rcv, nb_img, tmax, fs, **kwargs)
        rirs.append(np.asarray(rir, dtype=np.float32)[0])


    rir_len = max(r.shape[-1] for r in rirs)
    rendered = np.zeros((len(signal) + rir_len - 1, 2), dtype=np.float32)
    block_bounds = np.linspace(0, len(signal), len(rirs) + 1).astype(int)
    # for i in tqdm(range(len(rirs)), desc="Applying time-varying RIRs"):
    for i in range(len(rirs)):
        start, end = int(block_bounds[i]), int(block_bounds[i + 1])
        block = signal[start:end]
        if block.size == 0:
            continue
        for channel in (0, 1):
            y = np.convolve(block, rirs[i][channel], mode="full")
            rendered[start:start + len(y), channel] += y

    stereo = rendered[:len(signal)]
    if normalize:
        peak = float(np.max(np.abs(stereo)) + 1e-8)
        if peak > 0.99:
            stereo = stereo / peak * 0.99
    return stereo.astype(np.float32)


@dataclass
class AudioLevelMatchReport:
    """Audit record for source-RMS matching applied after spatial rendering."""
    enabled: bool
    rms_definition: str
    mono_reference_rms: float
    stereo_rms_before_match: float
    requested_shared_gain: float
    requested_shared_gain_db: float
    peak_after_requested_gain: float
    clip_protection: str
    limiter_applied: bool
    limiter_min_gain: float
    final_stereo_rms: float
    final_rms_error_db: float
    final_peak: float
    peak_ceiling: float
    spatial_preservation_note: str

    def as_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "rms_definition": self.rms_definition,
            "mono_reference_rms": self.mono_reference_rms,
            "stereo_rms_before_match": self.stereo_rms_before_match,
            "requested_shared_gain": self.requested_shared_gain,
            "requested_shared_gain_db": self.requested_shared_gain_db,
            "peak_after_requested_gain": self.peak_after_requested_gain,
            "clip_protection": self.clip_protection,
            "limiter_applied": self.limiter_applied,
            "limiter_min_gain": self.limiter_min_gain,
            "final_stereo_rms": self.final_stereo_rms,
            "final_rms_error_db": self.final_rms_error_db,
            "final_peak": self.final_peak,
            "peak_ceiling": self.peak_ceiling,
            "spatial_preservation_note": self.spatial_preservation_note,
        }


def signal_rms(signal: np.ndarray, eps: float = 1e-12) -> float:
    """RMS of a mono waveform over all samples."""
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x)) + eps))


def stereo_energy_rms(stereo: np.ndarray, eps: float = 1e-12) -> float:
    """Stereo energy-RMS: sqrt(mean((L^2 + R^2) / 2)).

    This metric avoids converting spatial audio back to mono for loudness
    matching; a mono downmix can cancel anti-phase/spatial components.
    """
    y = np.asarray(stereo, dtype=np.float64)
    if y.ndim != 2 or y.shape[1] != 2:
        raise ValueError(f"Expected stereo array [N,2], got {y.shape}")
    if y.shape[0] == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.sum(np.square(y), axis=1) / 2.0) + eps))


def amplitude_to_db(value: float, eps: float = 1e-12) -> float:
    return float(20.0 * np.log10(max(float(value), eps)))


def apply_stereo_linked_peak_limiter(
    stereo: np.ndarray,
    fs: int,
    ceiling: float = 0.98,
    release_ms: float = 80.0,
) -> tuple[np.ndarray, float]:
    """Prevent clipping while retaining the spatial L/R relationship.

    The detector observes the larger absolute sample across L/R.  The exact
    same gain is applied to both channels at every sample, so gain protection
    cannot independently flatten an ILD cue.  Gain reduction attacks
    immediately for safety and releases smoothly to avoid pumping.
    """
    if not (0.0 < ceiling < 1.0):
        raise ValueError("--output_peak_ceiling must be in (0, 1)")
    y = np.asarray(stereo, dtype=np.float32)
    if y.ndim != 2 or y.shape[1] != 2:
        raise ValueError(f"Expected stereo array [N,2], got {y.shape}")
    if y.shape[0] == 0:
        return y.copy(), 1.0
    peak_by_sample = np.max(np.abs(y), axis=1)
    demanded = np.minimum(1.0, ceiling / np.maximum(peak_by_sample, 1e-12)).astype(np.float32)
    release_samples = max(1, int(round(float(release_ms) * fs / 1000.0)))
    release_coeff = float(np.exp(-1.0 / release_samples))
    gain = np.ones_like(demanded)
    previous = 1.0
    for i, required in enumerate(demanded):
        if required < previous:
            previous = float(required)  # immediate linked attack: no clipped samples
        else:
            previous = min(float(required), min(1.0, release_coeff * previous + (1.0 - release_coeff)))
        gain[i] = previous
    limited = y * gain[:, None]
    return limited.astype(np.float32), float(np.min(gain))


def match_spatial_stereo_rms_to_mono(
    mono_reference: np.ndarray,
    stereo_rendered: np.ndarray,
    fs: int,
    enabled: bool = True,
    peak_ceiling: float = 0.98,
    clip_protection: str = "linked_limiter",
    limiter_release_ms: float = 80.0,
    limiter_iterations: int = 4,
) -> tuple[np.ndarray, AudioLevelMatchReport]:
    """Match stereo energy-RMS to the original mono without re-panning it.

    Only a shared global gain is used for the RMS target.  No left/right
    channel is independently normalised.  If a target-RMS gain would clip, the
    selected protection policy is applied after that shared gain:
      - linked_limiter: shared time-varying attenuation for L/R; tries a few
        iterations to approach target RMS while protecting peaks.
      - uniform_headroom: shared fixed attenuation; preserves all dynamics
        exactly but final RMS can remain below target when clipping would occur.
      - none: do not constrain peaks (not recommended for integer PCM output).
    """
    mono = np.asarray(mono_reference, dtype=np.float32).reshape(-1)
    stereo = np.asarray(stereo_rendered, dtype=np.float32)
    if stereo.ndim != 2 or stereo.shape[1] != 2:
        raise ValueError(f"Expected stereo array [N,2], got {stereo.shape}")
    n = min(len(mono), len(stereo))
    mono, stereo = mono[:n], stereo[:n]
    mono_rms = signal_rms(mono)
    stereo_before = stereo_energy_rms(stereo)
    if stereo_before <= 1e-10 or mono_rms <= 1e-10:
        report = AudioLevelMatchReport(
            enabled=enabled, rms_definition="sqrt(mean((L^2+R^2)/2))",
            mono_reference_rms=mono_rms, stereo_rms_before_match=stereo_before,
            requested_shared_gain=1.0, requested_shared_gain_db=0.0,
            peak_after_requested_gain=float(np.max(np.abs(stereo)) if stereo.size else 0.0),
            clip_protection=clip_protection, limiter_applied=False, limiter_min_gain=1.0,
            final_stereo_rms=stereo_before, final_rms_error_db=0.0,
            final_peak=float(np.max(np.abs(stereo)) if stereo.size else 0.0),
            peak_ceiling=peak_ceiling,
            spatial_preservation_note="Silent or near-silent signal; no level correction applied.",
        )
        return stereo.astype(np.float32), report
    desired_gain = mono_rms / stereo_before if enabled else 1.0
    target_rms = mono_rms if enabled else stereo_before
    requested = stereo * float(desired_gain)
    peak_after_request = float(np.max(np.abs(requested)))
    limiter_applied = False
    limiter_min_gain = 1.0
    output = requested
    if peak_after_request > peak_ceiling:
        if clip_protection == "none":
            pass
        elif clip_protection == "uniform_headroom":
            protection = peak_ceiling / max(peak_after_request, 1e-12)
            output = requested * float(protection)
            limiter_applied = True
            limiter_min_gain = float(protection)
        elif clip_protection == "linked_limiter":
            # Iteration raises a single common pre-gain only when the linked
            # limiter reduced RMS below the requested target.  The limiting
            # envelope itself remains shared across L/R at every iteration.
            pre_gain = float(desired_gain)
            output = requested
            for _ in range(max(1, int(limiter_iterations))):
                candidate = stereo * pre_gain
                output, iteration_min_gain = apply_stereo_linked_peak_limiter(
                    candidate, fs, peak_ceiling, limiter_release_ms
                )
                limiter_min_gain = min(limiter_min_gain, iteration_min_gain)
                achieved = stereo_energy_rms(output)
                if achieved >= target_rms * 0.999 or target_rms <= 1e-10:
                    break
                correction = min(target_rms / max(achieved, 1e-12), 2.0)
                pre_gain *= float(correction)
            limiter_applied = True
        else:
            raise ValueError(f"Unknown --rms_clip_protection: {clip_protection}")
    final_rms = stereo_energy_rms(output)
    final_peak = float(np.max(np.abs(output)) if output.size else 0.0)
    rms_error_db = amplitude_to_db(final_rms / max(target_rms, 1e-12))
    report = AudioLevelMatchReport(
        enabled=enabled,
        rms_definition="sqrt(mean((L^2+R^2)/2))",
        mono_reference_rms=mono_rms,
        stereo_rms_before_match=stereo_before,
        requested_shared_gain=float(desired_gain),
        requested_shared_gain_db=amplitude_to_db(desired_gain),
        peak_after_requested_gain=peak_after_request,
        clip_protection=clip_protection,
        limiter_applied=limiter_applied,
        limiter_min_gain=float(limiter_min_gain),
        final_stereo_rms=final_rms,
        final_rms_error_db=rms_error_db,
        final_peak=final_peak,
        peak_ceiling=float(peak_ceiling),
        spatial_preservation_note=(
            "RMS gain is shared by L/R. Peak protection, when used, is stereo-linked: "
            "the same gain envelope is applied to both channels, preserving channel balance/ILD."
        ),
    )
    return output.astype(np.float32), report


# =============================================================================
# Outputs / diagnostics
# =============================================================================


def save_3d_trajectory_csv(
    path: Path,
    frame_t: np.ndarray,
    center_x: np.ndarray,
    center_y: np.ndarray,
    area: np.ndarray,
    depth_m: np.ndarray,
    source_camera_xyz: np.ndarray,
    source_world_xyz: np.ndarray,
    receiver_xyz: np.ndarray,
    azimuth: np.ndarray,
    elevation: np.ndarray,
    distance: np.ndarray,
) -> None:
    header = [
        "time_s", "center_x_px", "center_y_px", "area_ratio", "depth_input_m",
        "source_cam_x_front_m", "source_cam_y_left_m", "source_cam_z_up_m",
        "source_world_x_m", "source_world_y_m", "source_world_z_m",
        "receiver_world_x_m", "receiver_world_y_m", "receiver_world_z_m",
        "relative_azimuth_deg", "relative_elevation_deg", "relative_range_m",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(len(frame_t)):
            writer.writerow([
                float(frame_t[i]), float(center_x[i]), float(center_y[i]), float(area[i]), float(depth_m[i]),
                *[float(v) for v in source_camera_xyz[i]],
                *[float(v) for v in source_world_xyz[i]],
                *[float(v) for v in receiver_xyz[i]],
                float(azimuth[i]), float(elevation[i]), float(distance[i]),
            ])


def _transcode_to_h264(src: Path, dst: Path) -> None:
    """Use ffmpeg to transcode an existing video into H.264 (libx264) in-place safe.

    若 ffmpeg 不可用或转码失败，则保留原始（OpenCV 写出的）视频，仅打印告警，
    不中断主流程。
    """
    if shutil.which("ffmpeg") is None:
        print(f"[overlay] ffmpeg 不可用，跳过 H.264 转码，保留原视频: {src}")
        return
    tmp_out = dst.with_suffix(dst.suffix + ".h264.tmp.mp4")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "veryfast", "-crf", "20",
        "-movflags", "+faststart",
        "-an",  # overlay 视频不含音频；如有需要可移除该选项
        str(tmp_out),
    ]
    try:
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"[overlay] ffmpeg H.264 转码失败，保留原视频。原因: {e}")
        if tmp_out.exists():
            try:
                tmp_out.unlink()
            except OSError:
                pass
        return
    # 用 H.264 版本替换目标文件
    os.replace(str(tmp_out), str(dst))


def save_overlay_video(
    input_video: Path,
    output_video: Path,
    center_x: np.ndarray,
    center_y: np.ndarray,
    azimuth: np.ndarray,
    elevation: np.ndarray,
    distance: np.ndarray,
    info: VideoInfo,
) -> None:
    """先用 OpenCV (mp4v) 写一份普通的 mp4，再调用 ffmpeg 转码为 H.264 覆盖输出。"""
    output_video = Path(output_video)
    cap = cv2.VideoCapture(str(input_video))

    # Stage 1: 用 OpenCV 以 mp4v 写一个临时普通视频。
    tmp_raw = output_video.with_suffix(output_video.suffix + ".raw.mp4")
    writer = cv2.VideoWriter(
        str(tmp_raw), cv2.VideoWriter_fourcc(*"mp4v"),
        info.fps, (info.width, info.height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create overlay video: {tmp_raw}")

    i = 0
    while i < info.total_frames:
        ok, frame = cap.read()
        if not ok:
            break
        c = (int(round(center_x[i])), int(round(center_y[i])))
        cv2.circle(frame, c, 8, (0, 0, 255), -1)
        cv2.line(frame, (info.width // 2, info.height // 2), c, (0, 255, 255), 2)
        texts = [
            f"az={azimuth[i]:+.1f} deg  el={elevation[i]:+.1f} deg",
            f"range={distance[i]:.2f} m",
        ]
        for j, text in enumerate(texts):
            cv2.putText(frame, text, (20, 40 + 38 * j), cv2.FONT_HERSHEY_SIMPLEX,
                        0.85, (0, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)
        i += 1
    cap.release()
    writer.release()

    # Stage 2: 用 ffmpeg 转码为 H.264，输出到目标路径。
    _transcode_to_h264(tmp_raw, output_video)

    # 清理临时 raw 文件（若转码成功已被 os.replace 移除，这里仅兜底）
    if tmp_raw.exists() and tmp_raw.resolve() != output_video.resolve():
        try:
            tmp_raw.unlink()
        except OSError:
            pass


# =============================================================================
# End-to-end pipeline
# =============================================================================


def process(args: argparse.Namespace) -> None:
    video = Path(args.video)
    mask = Path(args.mask)
    output = video.parent
    info = get_video_info(video)
    map_masks_over_entire_video_span = bool(getattr(args, "_luavs_map_masks_over_entire_video_span", False))
    if args.use_vggt and args.use_depth:
        raise ValueError("Choose one geometry path: --use_vggt or legacy --use_depth, not both")

    need_masks = True
    track = read_mask_track(
        mask, info, threshold=args.mask_threshold,
        largest_component=not args.use_all_components,
        mask_fps_override=args.mask_fps, keep_binary_masks=need_masks,
        map_masks_over_entire_video_span=map_masks_over_entire_video_span,
    )
    
    
    frame_t, center_x, center_y, area = resample_mask_geometry_to_video_frames(track, info, args.smooth_frames)
    valid_mask_ratio = float(np.mean(track.valid))
    if valid_mask_ratio < 0.8:
        print("WARNING: Many empty mask observations were interpolated; verify active-sound alignment.")

    metric_status = "fixed_assumed_distance"
    metric_note = "No depth estimator enabled; direction follows mask while distance is a fixed simulation parameter."
    scale_m_per_unit: Optional[float] = None
    native_export: Optional[dict[str, Any]] = None
    geometry_intrinsics: Any = None
    camera_motion_source = "none"

    geometry = compute_vggt_geometry_at_mask_times(video, track, args, info)
    native_export = geometry
    pred: VGGTSequencePrediction = geometry["prediction"]
    scale_m_per_unit = geometry["scale_m_per_native_unit"]
    metric_status = geometry["scale_status"]
    metric_note = geometry["scale_note"]
    # Interpolate VGGT camera-relative geometry from selected sequence views to the full video time axis.
    geometry_t = geometry["observation_t"]
    source_cam_audio_m = resample_vector_track(geometry_t, geometry["source_cam_audio_m"], frame_t)
    depth_frame = resample_values(geometry_t, geometry["depth_metric"], frame_t).astype(np.float32)
    source_cv_native_frame = resample_vector_track(geometry_t, geometry["source_cam_cv_native"], frame_t)
    depth_native_frame = resample_values(geometry_t, geometry["depth_native"], frame_t).astype(np.float32)
    geometry_intrinsics = "VGGT per-frame intrinsics; see supplied VGGT outputs"
    if args.pose_csv:
        # An externally calibrated pose CSV overrides VGGT pose and places scaled source offsets in metric room coordinates.
        source_world_xyz, receiver_xyz, ear_direction, _ = construct_world_scene_trajectory(
            source_cam_audio_m, frame_t, Path(args.pose_csv), tuple(args.receiver_origin or (
                args.room_size[0] / 2.0, args.room_size[1] / 2.0, args.ear_height
            ))
        )
        camera_motion_source = "external_pose_csv"
    else:
        receiver_origin = tuple(args.receiver_origin or (
            args.room_size[0] / 2.0, args.room_size[1] / 2.0, args.ear_height
        ))
        source_obs, receiver_obs, ear_obs = construct_world_trajectory_from_vggt(
            geometry["source_cam_cv_native"], geometry_t, pred, geometry["pred_index"],
            scale_m_per_unit, receiver_origin,
        )
        source_world_xyz = resample_vector_track(geometry_t, source_obs, frame_t)
        receiver_xyz = resample_vector_track(geometry_t, receiver_obs, frame_t)
        ear_direction = resample_vector_track(geometry_t, ear_obs, frame_t)
        ear_direction /= np.maximum(np.linalg.norm(ear_direction, axis=1, keepdims=True), 1e-8)
        camera_motion_source = (
            "vggt_extrinsics" if pred.extrinsics_cam_from_world is not None else "fixed_receiver_no_vggt_extrinsics"
        )
        # if pred.extrinsics_cam_from_world is not None:
        #     print(
        #         "NOTE: VGGT extrinsics are being used as relative camera/listener motion. "
        #         "For clips with large independently moving foreground sound sources, "
        #         "prefer an externally calibrated --pose_csv for metric evaluation."
        #     )

    azimuth, elevation, source_range = xyz_to_spherical_track(source_cam_audio_m)
    source_world_xyz, receiver_xyz = enforce_room_bounds(
        source_world_xyz, receiver_xyz, ear_direction,
        tuple(args.room_size), args.ear_distance, args.out_of_room_policy,
    )

    trajectory_csv = output / ".trajectory_3d.csv"
    
    if args.use_vggt and native_export is not None:
        save_vggt_trajectory_csv(
            trajectory_csv, frame_t, center_x, center_y, area, depth_native_frame, depth_frame,
            source_cv_native_frame, source_cam_audio_m, source_world_xyz, receiver_xyz,
            azimuth, elevation, source_range, float(scale_m_per_unit), metric_status,
        )
    else:
        save_3d_trajectory_csv(
            trajectory_csv, frame_t, center_x, center_y, area, depth_frame,
            source_cam_audio_m, source_world_xyz, receiver_xyz, azimuth, elevation, source_range,
        )
    overlay_path = output / "trajectory_overlay.mp4"
    
    if args.save_overlay:
        save_overlay_video(video, overlay_path, center_x, center_y, azimuth, elevation, source_range, info)

    output_wav: Optional[Path] = None
    audio_level_report: Optional[dict[str, object]] = None
    if not args.trajectory_only:
        if args.use_vggt and metric_status == "relative_units_only":
            raise ValueError("Relative-only VGGT tracks cannot be rendered as physically scaled RIR audio")
        if gpuRIR is None:
            raise ImportError("gpuRIR is not installed/importable; install it or add --trajectory_only")
        # with tempfile.TemporaryDirectory() as temp_dir:
            
        # temp = Path(temp_dir)
        # mono_path = Path(args.mono_audio) if args.mono_audio else temp / 
        mono_path = output / "source_mono.wav"
        if not mono_path.exists():
            extract_mono_audio_from_video(video, mono_path)
        mono, fs = read_audio_as_mono(mono_path)
        duration = min(info.duration, len(mono) / fs)
        mono = mono[: int(round(duration * fs))]
        _, src_rir, rcv_rir, dir_rir = downsample_scene_for_rir(
            frame_t, source_world_xyz, receiver_xyz, ear_direction, duration, args.rir_hz
        )
        stereo_raw = simulate_dynamic_spatial_stereo(
            mono, src_rir, rcv_rir, dir_rir, tuple(args.room_size), args.t60, fs,
            args.mic_pattern, args.ear_distance, args.max_rir_seconds, normalize=False,
        )
        stereo, level_report = match_spatial_stereo_rms_to_mono(
            mono_reference=mono,
            stereo_rendered=stereo_raw,
            fs=fs,
            enabled=args.match_mono_rms,
            peak_ceiling=args.output_peak_ceiling,
            clip_protection=args.rms_clip_protection,
            limiter_release_ms=args.rms_limiter_release_ms,
            limiter_iterations=args.rms_limiter_iterations,
        )
        audio_level_report = level_report.as_dict()
        # print(
        #     "Audio RMS match: mono_ref={:.6f}, raw_stereo={:.6f}, final_stereo={:.6f}, "
        #     "gain={:+.2f} dB, peak={:.4f}, limiter={}".format(
        #         level_report.mono_reference_rms, level_report.stereo_rms_before_match,
        #         level_report.final_stereo_rms, level_report.requested_shared_gain_db,
        #         level_report.final_peak, level_report.limiter_applied,
        #     )
        # )
        output_wav = output / "spatial_stereo.wav"
        
        sf.write(str(output_wav), stereo, fs, subtype="PCM_16")
        output_spatialized_video = output / "spatialized_video.mp4"
        mux_video_and_audio(video, output_wav, output_spatialized_video)

    metadata = {
        "video": str(video), "mask": str(mask), "output": str(output),
        "video_info": info.__dict__, "valid_mask_ratio": valid_mask_ratio,
        "geometry_mode": "vggt_relative_geometry" if args.use_vggt else ("metric_depth" if args.use_depth else "fixed_range"),
        "intrinsics": geometry_intrinsics,
        "use_vggt": bool(args.use_vggt),
        "vggt_provider": args.vggt_provider if args.use_vggt else None,
        "vggt_scale_mode": args.vggt_scale_mode if args.use_vggt else None,
        "vggt_model_path": args.vggt_model_path if args.use_vggt and args.vggt_provider == "hook" else None,
        "vggt_sample_fps": args.vggt_sample_fps if args.use_vggt and args.vggt_provider == "hook" else None,
        "vggt_max_frames": args.vggt_max_frames if args.use_vggt and args.vggt_provider == "hook" else None,
        "vggt_raw_prediction_cache": args.vggt_save_npz if args.use_vggt and args.vggt_provider == "hook" else None,
        "scale_m_per_vggt_unit": scale_m_per_unit,
        "metric_status": metric_status,
        "metric_interpretation": metric_note,
        "camera_motion_source": camera_motion_source,
        "coordinate_system": "+X forward, +Y listener/image-left, +Z up",
        "room_size_m": args.room_size, "t60_s": args.t60, "rir_hz": args.rir_hz,
        "audio_rms_matching": audio_level_report if audio_level_report is not None else {
            "enabled": bool(args.match_mono_rms),
            "status": "not_run_in_trajectory_only_mode" if args.trajectory_only else "not_available",
            "rms_definition": "sqrt(mean((L^2+R^2)/2))",
            "clip_protection": args.rms_clip_protection,
            "peak_ceiling": args.output_peak_ceiling,
        },
        "receiver_model": "two virtual ear receivers rendered by gpuRIR; not personalized HRTF/BRIR",
        "dynamic_scene_caveat": (
            "VGGT extrinsics are interpreted as camera/listener motion only when background geometry is "
            "sufficiently dominant and stable. For strong independent foreground/source motion, use an "
            "external calibrated pose CSV for metric evaluation."
        ),
        "files": {
            "trajectory_3d_csv": str(trajectory_csv),
            "overlay_video": str(overlay_path) if args.save_overlay else None,
            "rendered_stereo_wav": str(output_wav) if output_wav else None,
        },
        "reviewer_safe_claim": (
            "VGGT supplies sequence-consistent relative geometry. Absolute acoustic distance is claimed only "
            "when an external metric scale is provided; median_anchor produces scale-calibrated synthetic "
            "3-D audio with preserved relative motion, not ground-truth metric recovery."
        ),
    }
    metadata_path = output / "metadata.json"

    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    # print(f"Saved 3-D trajectory CSV: {trajectory_csv}")
    # if args.save_overlay: print(f"Saved trajectory overlay video: {overlay_path}")
    # if output_wav is not None:
    #     print(f"Saved spatial stereo WAV: {output_wav}"); print(f"Saved output video: {output}")
    # else:
    #     print("Trajectory-only mode: audio rendering and muxing were skipped")
    # print(f"Saved metadata: {metadata_path}")


# ---------------------------------------------------------------------------
# YAML 配置加载：除少量必需的命令行参数外，其它参数均在 config.yaml 中维护。
# ---------------------------------------------------------------------------
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def _flatten_config(cfg: dict) -> dict:
    """Flatten one-level grouped YAML config into a flat {param_name: value} dict."""
    flat: dict = {}
    if not isinstance(cfg, dict):
        return flat
    for key, val in cfg.items():
        if isinstance(val, dict):
            for sub_key, sub_val in val.items():
                flat[sub_key] = sub_val
        else:
            flat[key] = val
    return flat


def load_config(config_path: Path) -> dict:
    """Load YAML config and flatten it into a flat dict of parameter overrides."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return _flatten_config(raw)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mask + VGGT relative geometry/optional metric depth -> scale-audited 3-D trajectory -> dynamic spatial stereo"
    )

    # --- 仍由命令行控制的参数 -------------------------------------------------
    parser.add_argument(
        "--dataset_root",
        nargs="+",
        type=Path,
        required=True,
        help="一个或多个包含样本子目录的父目录；每个样本子目录中含 masks.npy 和 bbox.npy。",
    )
    parser.add_argument("--mask_fps", type=float, default=None)
    parser.add_argument("--mask_threshold", type=int, default=127)
    parser.add_argument("--use_all_components", action="store_true")
    parser.add_argument("--smooth_frames", type=int, default=7)
    parser.add_argument("--use_vggt", action="store_true",
                        help="Use VGGT relative geometry rather than metric/fixed-depth paths")
    parser.add_argument("--pose_csv", default=None,
                        help="Optional external metric camera/listener pose CSV; overrides VGGT extrinsics")

    # --- 配置文件路径 --------------------------------------------------------
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                        help="YAML 配置文件路径（包含其余所有参数）。")

    # match_mono_rms 在原代码中通过 set_defaults 写入 args；此处保留默认 True
    parser.set_defaults(match_mono_rms=True)
    return parser


def parse_args_with_config() -> argparse.Namespace:
    """Parse CLI args, then merge YAML config values into the namespace.

    Precedence (low -> high):
        hard-coded defaults  <  YAML config  <  CLI args (currently the small whitelist).
    """
    args = build_parser().parse_args()
    cfg = load_config(args.config)

    # Inject YAML values; only set if attribute is missing on args (avoid clobbering
    # user-provided CLI flags within the small whitelist above).
    for key, value in cfg.items():
        if not hasattr(args, key):
            setattr(args, key, value)

    # Backwards-compat: code paths reference DEFAULT_VGGT_MODEL_PATH when path missing.
    if getattr(args, "vggt_model_path", None) in (None, "", "null"):
        args.vggt_model_path = DEFAULT_VGGT_MODEL_PATH

    return args

def analyze_bool_npy_sequence(npy_path: str | Path):
    """
    读取形状为 (T, H, W) 的布尔类型 npy 文件，分析时间轴上有效帧的分布情况。
    
    分析逻辑：
    - 每一帧 (H, W) 内只要有任意像素为 True，该帧即为有效帧(1)；全为 False 则为空置帧(0)。
    - 函数会检测在有效帧区间开启后，中间或结尾是否存在空置帧(0)，并统计各自的数量。
    """
    video_array = np.load(npy_path)
    if video_array.ndim != 3:
        raise ValueError(f"期望输入的数组维度为 3 (T, H, W)，但得到的形状是 {video_array.shape}")
        
    T, H, W = video_array.shape
    
    frame_has_signal = np.any(video_array, axis=(1, 2))
    valid_indices = np.where(frame_has_signal)[file_has_signal_index := 0]
    if len(valid_indices) == 0:
        print(f"统计结果：该文件共 {T} 帧，全序列均为 0 (无任何有效目标)。")
        return {
            "head_zeros": T,
            "middle_zeros": 0,
            "tail_zeros": 0
        }
        
    first_valid_idx = valid_indices[0]
    last_valid_idx = valid_indices[-1]
    head_zeros = int(first_valid_idx)
    tail_zeros = int(T - 1 - last_valid_idx)
    middle_sequence = frame_has_signal[first_valid_idx : last_valid_idx + 1]
    middle_zeros = int(np.sum(~middle_sequence))
    return {
        "head_zeros": head_zeros,
        "middle_zeros": middle_zeros,
        "tail_zeros": tail_zeros
    }

def check_temporal_alignment(video_path: str | Path, npy_path: str | Path) -> bool:
    """
    检查视频文件的帧数与 npy 文件的时间轴(T维度)是否完全一致。
    
    Args:
        video_path: 视频文件路径 (.mp4, .mkv 等)
        npy_path: 矩阵文件路径 (.npy)，期望其第一维是时间轴 T
        
    Returns:
        bool: 如果对齐返回 True，否则返回 False
    """
    vid_p = Path(video_path)
    npy_p = Path(npy_path)
    
    # 1. 安全性检查
    if not vid_p.exists():
        return False
    if not npy_p.exists():
        return False

    # 2. 快速获取视频的实际总帧数
    cap = cv2.VideoCapture(str(vid_p))
    if not cap.isOpened():
        return False
    
    # CAP_PROP_FRAME_COUNT 读取的是视频头信息中的帧数，无需遍历视频，耗时为 0
    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release() # 及时释放视频指针

    # 3. 获取 npy 文件的形状
    try:
        # mmap_mode='r' 表示只读取内存映射，不把整个大矩阵载入内存，极大地节省显存和内存
        npy_shape = np.load(str(npy_p), mmap_mode='r').shape
    except Exception as e:
        return False

    if len(npy_shape) == 0:
        return False
        
    npy_frames = npy_shape[0]  # 默认第一维是时间轴 T

    if video_frames == npy_frames:
        return True
    else:
        diff = abs(video_frames - npy_frames)
        return False

def main() -> None:
    args = parse_args_with_config()
    if gpuRIR is not None and not args.trajectory_only:
        gpuRIR.activateMixedPrecision(False)

    dataset_root_list = args.dataset_root
    max_count = 1e8

    def _iter_sample_dirs(dataset_root: Path):
        """Yield sample directories from either a directory root or a txt manifest.

        - If ``dataset_root`` is a directory, each immediate sub-directory is a sample.
        - If ``dataset_root`` is a ``.txt`` file, each non-empty line is a path to a
          ``clean.mp4`` file, and the sample directory is its parent directory.
        """
        dataset_root = Path(dataset_root)
        if dataset_root.is_file() and dataset_root.suffix.lower() == ".txt":
            with open(dataset_root, "r", encoding="utf-8") as fp:
                for raw_line in fp:
                    line = raw_line.strip()
                    if not line or line.startswith("#"):
                        continue
                    clean_mp4 = Path(line)
                    yield clean_mp4.parent
        elif dataset_root.is_dir():
            # 情形 A：传入的就是单个样本子目录（含 clean.mp4 或 masks.npy）。
            if (dataset_root / "masks.npy").exists() or (dataset_root / "clean.mp4").exists():
                yield dataset_root
            else:
                # 情形 B：传入的是包含若干样本子目录的父目录。
                for sample_name in os.listdir(dataset_root):
                    yield dataset_root / sample_name
        else:
            raise ValueError(
                f"--dataset_root must be a directory or a .txt manifest file, got: {dataset_root}"
            )

    for dataset_root in dataset_root_list:
        print(dataset_root)        
        print(f"Proccessing {dataset_root}")
        sample_dirs = list(_iter_sample_dirs(dataset_root))
        
        for index, sample_dir in enumerate(tqdm(sample_dirs)):
            if index > max_count:
                break
            sample_dir = Path(sample_dir)

            extracted_segment = sample_dir / "clean.mp4"
            mask_npy = sample_dir / "masks.npy"

            routed_args = argparse.Namespace(**vars(args))
            routed_args.video = str(extracted_segment)
            routed_args.mask = str(mask_npy)

            # If sparse masks annotate a longer interval, map them over that full extracted span.
            routed_args._luavs_map_masks_over_entire_video_span = not check_temporal_alignment(routed_args.video, routed_args.mask)
            
            try:
                process(routed_args)
            except Exception as e:
                import traceback
                print(f"[ERROR] process() failed for sample_dir: {sample_dir}")
                print(f"[ERROR] Exception: {type(e).__name__}: {e}")
                traceback.print_exc()
                continue


if __name__ == "__main__":
    main()




# cv2.imwrite("11.jpg", geometry["prediction"].depth_native[0] / geometry["prediction"].depth_native[0].max() * 255)
