#!/usr/bin/env python3
"""Batch-render multi-track motion videos from masks.npy files.

This script is a bounded multi-threaded renderer. It recursively finds
``masks.npy`` files below a parent directory, derives one smoothed centroid
trajectory per mask channel, and writes a sibling ``motion.mp4``.

Supported mask layouts
----------------------
    [T, H, W]       : one trajectory
    [T, C, H, W]    : C independent trajectories, one per mask channel

For example, a boolean array with shape ``[121, 3, 720, 406]`` produces one
406x720 video containing three independently smoothed trajectories.

Output guarantees
-----------------
    * exactly 121 frames
    * 24 fps
    * output width/height exactly match masks.npy (W x H)
    * black background
    * one distinct color per mask channel when C > 1

Robustness / smoothing pipeline per channel
-------------------------------------------
    1. Compute the centroid of the largest connected component (default) or
       all foreground pixels for every frame.
    2. Ignore empty/tiny masks.
    3. Linearly interpolate invalid frames from valid temporal neighbours.
    4. Replace isolated local-median outliers.
    5. Apply a symmetric triangular temporal smoother.

Concurrency safety
------------------
    * A bounded ThreadPoolExecutor processes samples concurrently.
    * Only the main thread updates tqdm.
    * Each FFmpeg process has a hard timeout and is tracked for cancellation.
    * FFmpeg stderr goes to a temporary file rather than a PIPE, preventing
      stderr-buffer deadlocks.
    * Videos are first written to a temporary path and atomically replaced
      only after a successful encode.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
from tqdm import tqdm


NUM_FRAMES = 121
FPS = 24

# Sparse-track-style trail rendering constants.
_MIN_RADIUS = 2
_MAX_RADIUS = 8
_MAX_TRAIL = 50
_REF_SHORT_SIDE = 1080

# Distinct RGB colours for channel identities in multi-track videos.
# Channel 0/1/2 are red/green/blue respectively, so a [T,3,H,W] input is
# immediately readable. Further channels cycle through additional hues.
_TRACK_COLORS_RGB: tuple[tuple[int, int, int], ...] = (
    (255, 72, 72),    # red
    (72, 230, 112),   # green
    (76, 148, 255),   # blue
    (255, 210, 62),   # yellow
    (225, 82, 255),   # magenta
    (58, 228, 228),   # cyan
    (255, 142, 58),   # orange
    (170, 110, 255),  # violet
)


class SkipSample(RuntimeError):
    """Expected non-fatal skip for one sample."""


class BatchCancelled(RuntimeError):
    """Raised inside a worker after Ctrl+C requests cancellation."""


class ActiveProcessRegistry:
    """Thread-safe registry of active FFmpeg processes."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._processes: dict[int, subprocess.Popen[Any]] = {}

    def add(self, process: subprocess.Popen[Any]) -> None:
        with self._lock:
            self._processes[id(process)] = process

    def discard(self, process: subprocess.Popen[Any]) -> None:
        with self._lock:
            self._processes.pop(id(process), None)

    def terminate_all(self) -> None:
        with self._lock:
            processes = list(self._processes.values())

        for process in processes:
            if process.poll() is None:
                try:
                    process.kill()
                except (ProcessLookupError, OSError):
                    pass


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def odd_positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1 or parsed % 2 == 0:
        raise argparse.ArgumentTypeError("value must be a positive odd integer")
    return parsed


def default_workers() -> int:
    """Conservative default to avoid overloading storage and CPU."""
    return min(8, max(1, os.cpu_count() or 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render one multi-track, 121-frame, 24-fps motion.mp4 beside each "
            "masks.npy found recursively below parent_dir."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "parent_dir",
        type=Path,
        help="Parent directory containing sample subdirectories.",
    )
    parser.add_argument(
        "--output-name",
        default="motion.mp4",
        help="Output filename written beside every masks.npy.",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Skip samples whose output video already exists.",
    )
    parser.add_argument(
        "--workers",
        type=positive_int,
        default=default_workers(),
        help="Number of samples handled concurrently.",
    )
    parser.add_argument(
        "--max-in-flight",
        type=positive_int,
        default=None,
        help=(
            "Maximum submitted-but-unfinished samples. Defaults to 2 * workers; "
            "the bound prevents a huge pending Future queue."
        ),
    )
    parser.add_argument(
        "--ffmpeg-threads",
        type=positive_int,
        default=1,
        help=(
            "Encoder threads per FFmpeg child process. Keep this at 1 when "
            "using multiple --workers."
        ),
    )
    parser.add_argument(
        "--ffmpeg-timeout-seconds",
        type=positive_float,
        default=180.0,
        help="Hard timeout for one FFmpeg encoding job.",
    )
    parser.add_argument(
        "--centroid-mode",
        choices=("largest-component", "all-foreground"),
        default="largest-component",
        help=(
            "largest-component ignores small stray fragments per mask channel; "
            "all-foreground uses all foreground pixels in a channel."
        ),
    )
    parser.add_argument(
        "--min-mask-pixels",
        type=positive_int,
        default=16,
        help="Masks/components smaller than this are invalid for that frame.",
    )
    parser.add_argument(
        "--min-valid-ratio",
        type=nonnegative_float,
        default=0.10,
        help=(
            "Minimum valid-centroid ratio required per channel among the first "
            "121 frames. A channel below this threshold is omitted, but other "
            "valid channels in the same file are still rendered."
        ),
    )
    parser.add_argument(
        "--smooth-window",
        type=odd_positive_int,
        default=7,
        help=(
            "Odd temporal window for spike repair and symmetric triangular "
            "smoothing. Set to 1 to disable smoothing."
        ),
    )
    parser.add_argument(
        "--spike-threshold-ratio",
        type=nonnegative_float,
        default=0.03,
        help=(
            "A point farther than this fraction of the image diagonal from its "
            "local median is repaired as an isolated spike. Set 0 to disable "
            "spike repair only."
        ),
    )
    args = parser.parse_args()

    if not (0.0 <= args.min_valid_ratio <= 1.0):
        parser.error("--min-valid-ratio must be in [0, 1].")
    return args


def mask_layout_and_shape(mask_array: np.ndarray) -> tuple[int, int, int, int, str]:
    """Validate masks layout and return T, C, H, W, layout name.

    ``[T,H,W]`` is a single-track compatibility form. Any ``[T,C,H,W]`` with
    C >= 1 is treated as C independent mask trajectories.
    """
    if mask_array.ndim == 3:
        total_frames, height, width = mask_array.shape
        channels = 1
        layout = "[T,H,W]"
    elif mask_array.ndim == 4:
        total_frames, channels, height, width = mask_array.shape
        layout = "[T,C,H,W]"
    else:
        raise ValueError(
            f"Unsupported masks shape {tuple(mask_array.shape)}; expected "
            "[T,H,W] or [T,C,H,W]."
        )

    if total_frames <= 0 or channels <= 0 or height <= 0 or width <= 0:
        raise ValueError(
            "Invalid masks dimensions: "
            f"T={total_frames}, C={channels}, H={height}, W={width}."
        )
    return int(total_frames), int(channels), int(height), int(width), layout


def frame_mask(
    mask_array: np.ndarray,
    frame_index: int,
    channel_index: int,
) -> np.ndarray:
    """Return one HxW mask view for either accepted input layout."""
    if mask_array.ndim == 3:
        if channel_index != 0:
            raise IndexError("[T,H,W] mask input only has channel 0.")
        return mask_array[frame_index]
    return mask_array[frame_index, channel_index]


def centroid_from_mask(
    mask: np.ndarray,
    centroid_mode: str,
    min_mask_pixels: int,
) -> tuple[float, float] | None:
    """Return a foreground centroid (x, y) or None for an invalid frame."""
    foreground = np.asarray(mask, dtype=bool)
    foreground_count = int(np.count_nonzero(foreground))
    if foreground_count < min_mask_pixels:
        return None

    if centroid_mode == "all-foreground":
        ys, xs = np.nonzero(foreground)
        return float(xs.mean()), float(ys.mean())

    # Largest connected component suppresses small unrelated segmentation
    # fragments that may otherwise abruptly pull the centroid.
    binary_u8 = foreground.astype(np.uint8, copy=False)
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
        binary_u8,
        connectivity=8,
    )
    if num_labels <= 1:
        return None

    component_areas = stats[1:, cv2.CC_STAT_AREA]
    component_label = int(np.argmax(component_areas)) + 1
    if int(stats[component_label, cv2.CC_STAT_AREA]) < min_mask_pixels:
        return None

    center_x, center_y = centroids[component_label]
    if not (np.isfinite(center_x) and np.isfinite(center_y)):
        return None
    return float(center_x), float(center_y)


def interpolate_centers(
    centers: np.ndarray,
    valid: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Linearly fill missing 2D centers, with nearest-value edge filling."""
    if centers.shape != (NUM_FRAMES, 2) or valid.shape != (NUM_FRAMES,):
        raise ValueError("Unexpected center/validity shape during interpolation.")
    if not np.any(valid):
        raise ValueError("Cannot interpolate a trajectory with no valid frames.")

    output = centers.astype(np.float32, copy=True)
    indices = np.arange(NUM_FRAMES, dtype=np.float32)
    valid_indices = indices[valid]
    for dimension in range(2):
        output[:, dimension] = np.interp(
            indices,
            valid_indices,
            output[valid, dimension],
        ).astype(np.float32)

    output[:, 0] = np.clip(output[:, 0], 0.0, float(width - 1))
    output[:, 1] = np.clip(output[:, 1], 0.0, float(height - 1))
    return output


def extract_mask_tracks(
    mask_path: Path,
    centroid_mode: str,
    min_mask_pixels: int,
    min_valid_ratio: float,
) -> tuple[
    list[tuple[int, np.ndarray, np.ndarray]],
    list[dict[str, Any]],
    int,
    int,
    int,
    int,
    tuple[int, ...],
    str,
]:
    """Extract independently interpolated trajectories for every mask channel.

    Returns
    -------
    active_tracks:
        ``[(channel_index, centers[T,2], valid[T]), ...]``. Channels lacking
        sufficient valid frames are excluded, while valid siblings survive.
    channel_stats:
        One record per original channel, including whether it was rendered.
    total_frames, channels, height, width, original_shape, layout
    """
    masks = np.load(mask_path, mmap_mode="r", allow_pickle=False)
    try:
        original_shape = tuple(masks.shape)
        total_frames, channels, height, width, layout = mask_layout_and_shape(masks)

        if total_frames < NUM_FRAMES:
            raise SkipSample(
                f"masks has {total_frames} frames, fewer than required {NUM_FRAMES}."
            )

        min_valid_count = max(2, int(np.ceil(NUM_FRAMES * min_valid_ratio)))
        active_tracks: list[tuple[int, np.ndarray, np.ndarray]] = []
        channel_stats: list[dict[str, Any]] = []

        # Process one channel at a time. This keeps peak memory low even for a
        # large number of mask channels and avoids loading all masks into RAM.
        for channel_index in range(channels):
            centers = np.full((NUM_FRAMES, 2), np.nan, dtype=np.float32)
            valid = np.zeros(NUM_FRAMES, dtype=bool)

            for frame_index in range(NUM_FRAMES):
                center = centroid_from_mask(
                    frame_mask(masks, frame_index, channel_index),
                    centroid_mode=centroid_mode,
                    min_mask_pixels=min_mask_pixels,
                )
                if center is not None:
                    centers[frame_index] = center
                    valid[frame_index] = True

            valid_count = int(valid.sum())
            is_active = valid_count >= min_valid_count
            stat: dict[str, Any] = {
                "channel_index": channel_index,
                "valid_mask_frames": valid_count,
                "interpolated_empty_or_invalid_frames": int(NUM_FRAMES - valid_count),
                "active": is_active,
                "spike_repaired_frames": 0,
            }

            if is_active:
                centers = interpolate_centers(centers, valid, width, height)
                active_tracks.append((channel_index, centers, valid))
            else:
                stat["inactive_reason"] = (
                    f"valid_frames={valid_count} < min_required={min_valid_count}"
                )

            channel_stats.append(stat)
    finally:
        # Explicitly release the mmap before the next sample is processed.
        del masks

    if not active_tracks:
        raise SkipSample(
            "No mask channel has enough valid centroid frames to render "
            f"(requires at least {min_valid_count}/{NUM_FRAMES} per channel)."
        )

    return (
        active_tracks,
        channel_stats,
        total_frames,
        channels,
        height,
        width,
        original_shape,
        layout,
    )


def local_median(points: np.ndarray, window: int) -> np.ndarray:
    """Per-frame local temporal median with edge-value padding."""
    if window <= 1:
        return points.astype(np.float32, copy=True)

    radius = window // 2
    padded = np.pad(points, ((radius, radius), (0, 0)), mode="edge")
    output = np.empty_like(points, dtype=np.float32)
    for index in range(points.shape[0]):
        output[index] = np.median(padded[index:index + window], axis=0)
    return output


def triangular_smooth(points: np.ndarray, window: int) -> np.ndarray:
    """Symmetric triangular moving average with no causal time lag."""
    if window <= 1:
        return points.astype(np.float32, copy=True)

    radius = window // 2
    ascending = np.arange(1, radius + 2, dtype=np.float32)
    weights = np.concatenate((ascending, ascending[-2::-1]))
    weights /= weights.sum()

    padded = np.pad(points, ((radius, radius), (0, 0)), mode="edge")
    smoothed = np.empty_like(points, dtype=np.float32)
    for index in range(points.shape[0]):
        smoothed[index] = np.sum(
            padded[index:index + window] * weights[:, None],
            axis=0,
        )
    return smoothed


def smooth_centers(
    centers: np.ndarray,
    width: int,
    height: int,
    smooth_window: int,
    spike_threshold_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Repair isolated outliers and symmetrically smooth a trajectory."""
    if centers.shape != (NUM_FRAMES, 2):
        raise ValueError(
            f"Expected centers shape {(NUM_FRAMES, 2)}, got {centers.shape}."
        )

    if smooth_window <= 1:
        return centers.astype(np.float32, copy=True), np.zeros(NUM_FRAMES, dtype=bool)

    repaired = centers.astype(np.float32, copy=True)
    median_points = local_median(repaired, smooth_window)
    diagonal = float(np.hypot(width, height))

    if spike_threshold_ratio > 0.0:
        deviations = np.linalg.norm(repaired - median_points, axis=1)
        spike_threshold_px = spike_threshold_ratio * diagonal
        spike_mask = deviations > spike_threshold_px
        repaired[spike_mask] = median_points[spike_mask]
    else:
        spike_mask = np.zeros(NUM_FRAMES, dtype=bool)

    smoothed = triangular_smooth(repaired, smooth_window)
    smoothed[:, 0] = np.clip(smoothed[:, 0], 0.0, float(width - 1))
    smoothed[:, 1] = np.clip(smoothed[:, 1], 0.0, float(height - 1))
    return smoothed, spike_mask


def single_track_age_color_rgb(ratio: float) -> tuple[int, int, int]:
    """Legacy single-track colour progression: blue -> green -> yellow -> red."""
    if ratio <= 1.0 / 3.0:
        value = ratio * 3.0
        r, g, b = 0.0, value, 1.0 - value
    elif ratio <= 2.0 / 3.0:
        value = (ratio - 1.0 / 3.0) * 3.0
        r, g, b = value, 1.0, 0.0
    else:
        value = (ratio - 2.0 / 3.0) * 3.0
        r, g, b = 1.0, 1.0 - value, 0.0
    return int(round(r * 255)), int(round(g * 255)), int(round(b * 255))


def multi_track_color_rgb(channel_index: int, ratio: float) -> tuple[int, int, int]:
    """Distinct channel colour with brightness increasing toward the newest point."""
    base = _TRACK_COLORS_RGB[channel_index % len(_TRACK_COLORS_RGB)]
    # Keep old trajectory history visible but visually subordinate to the newest
    # point. This encodes age without conflating channel identity.
    brightness = 0.26 + 0.74 * float(np.clip(ratio, 0.0, 1.0))
    return tuple(int(round(component * brightness)) for component in base)


def render_resolution(width: int, height: int) -> tuple[int, int, float, float]:
    """Render with 1080 short-side canvas, then resize to the source resolution."""
    if height <= width:
        render_height = _REF_SHORT_SIDE
        render_width = max(1, int(round(width * _REF_SHORT_SIDE / height)))
    else:
        render_width = _REF_SHORT_SIDE
        render_height = max(1, int(round(height * _REF_SHORT_SIDE / width)))

    return (
        render_width,
        render_height,
        render_width / float(width),
        render_height / float(height),
    )


def generate_track_frames(
    smoothed_tracks: list[tuple[int, np.ndarray]],
    total_mask_channels: int,
    width: int,
    height: int,
    cancel_event: threading.Event,
) -> Iterator[np.ndarray]:
    """Render all active mask-channel trajectories into one black RGB video."""
    if not smoothed_tracks:
        raise ValueError("At least one active track is required for rendering.")

    render_width, render_height, scale_x, scale_y = render_resolution(width, height)
    render_tracks: list[tuple[int, np.ndarray]] = []
    for channel_index, centers in smoothed_tracks:
        if centers.shape != (NUM_FRAMES, 2):
            raise ValueError(
                f"Channel {channel_index} has invalid centers shape {centers.shape}; "
                f"expected {(NUM_FRAMES, 2)}."
            )
        scaled = centers.astype(np.float32, copy=True)
        scaled[:, 0] *= scale_x
        scaled[:, 1] *= scale_y
        render_tracks.append((channel_index, scaled))

    use_legacy_single_track_colour = total_mask_channels == 1 and len(render_tracks) == 1

    for frame_index in range(NUM_FRAMES):
        if cancel_event.is_set():
            raise BatchCancelled("Batch cancellation requested.")

        canvas_rgb = np.zeros((render_height, render_width, 3), dtype=np.uint8)
        trail_start = max(0, frame_index - _MAX_TRAIL)

        # Draw one trajectory at a time. Within each one, old points are painted
        # first and newer points are painted last, keeping the latest location
        # prominent. Channel order is deterministic and follows input channel id.
        for channel_index, centers in render_tracks:
            for point_index in range(trail_start, frame_index + 1):
                x = int(round(float(centers[point_index, 0])))
                y = int(round(float(centers[point_index, 1])))
                if not (0 <= x < render_width and 0 <= y < render_height):
                    continue

                age = frame_index - point_index
                ratio = float(np.clip(1.0 - age / float(_MAX_TRAIL), 0.0, 1.0))
                radius = max(
                    1,
                    int(round(_MIN_RADIUS + (_MAX_RADIUS - _MIN_RADIUS) * ratio)),
                )
                color = (
                    single_track_age_color_rgb(ratio)
                    if use_legacy_single_track_colour
                    else multi_track_color_rgb(channel_index, ratio)
                )
                # canvas_rgb is raw RGB byte storage. FFmpeg receives it with
                # -pix_fmt rgb24, so no OpenCV BGR conversion is performed here.
                cv2.circle(
                    canvas_rgb,
                    center=(x, y),
                    radius=radius,
                    color=color,
                    thickness=-1,
                    lineType=cv2.LINE_8,
                )

        yield cv2.resize(
            canvas_rgb,
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )


def tail_text(path: Path, max_bytes: int = 16_384) -> str:
    """Read a bounded FFmpeg stderr tail without pipe-buffer deadlock risk."""
    if not path.exists():
        return "<FFmpeg did not create an error log.>"
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes), os.SEEK_SET)
            return handle.read().decode("utf-8", errors="replace").strip()
    except OSError as exc:
        return f"<Could not read FFmpeg log: {exc}>"


def make_temp_output_path(output_path: Path) -> Path:
    token = uuid.uuid4().hex
    return output_path.with_name(
        f".{output_path.stem}.partial-{os.getpid()}-{token}{output_path.suffix}"
    )


def save_h264_video(
    frames: Iterator[np.ndarray],
    output_path: Path,
    width: int,
    height: int,
    fps: int,
    ffmpeg_threads: int,
    timeout_seconds: float,
    cancel_event: threading.Event,
    process_registry: ActiveProcessRegistry,
) -> None:
    """Encode a video safely; only promote the temporary file after success."""
    if cancel_event.is_set():
        raise BatchCancelled("Batch cancellation requested before FFmpeg start.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output_path = make_temp_output_path(output_path)

    # yuv420p requires even spatial dimensions. For an odd H/W mask, retain the
    # exact shape using yuv444p instead of silently resizing or padding.
    if width % 2 == 0 and height % 2 == 0:
        out_pix_fmt = "yuv420p"
        profile = "high"
    else:
        out_pix_fmt = "yuv444p"
        profile = "high444"

    stderr_log_path: Path | None = None
    process: subprocess.Popen[Any] | None = None
    watchdog: threading.Timer | None = None
    timeout_triggered = threading.Event()
    success = False

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".motion_ffmpeg_",
            suffix=".log",
            dir=output_path.parent,
            delete=False,
        ) as log_handle:
            stderr_log_path = Path(log_handle.name)
            command = [
                "ffmpeg",
                "-y",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "10",
                "-threads",
                str(ffmpeg_threads),
                "-profile:v",
                profile,
                "-pix_fmt",
                out_pix_fmt,
                "-movflags",
                "+faststart",
                "-frames:v",
                str(NUM_FRAMES),
                str(temp_output_path),
            ]
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=log_handle,
            )
            process_registry.add(process)

            def kill_if_timed_out() -> None:
                if process is not None and process.poll() is None:
                    timeout_triggered.set()
                    try:
                        process.kill()
                    except (ProcessLookupError, OSError):
                        pass

            watchdog = threading.Timer(timeout_seconds, kill_if_timed_out)
            watchdog.daemon = True
            watchdog.start()

            assert process.stdin is not None
            frames_written = 0
            pipe_broken = False
            try:
                for frame in frames:
                    if cancel_event.is_set():
                        raise BatchCancelled("Batch cancellation requested during encoding.")
                    if timeout_triggered.is_set():
                        raise TimeoutError(
                            f"FFmpeg exceeded {timeout_seconds:.1f} seconds."
                        )
                    if frame.shape != (height, width, 3):
                        raise ValueError(
                            f"Frame shape mismatch: expected {(height, width, 3)}, "
                            f"got {frame.shape}."
                        )
                    if frame.dtype != np.uint8:
                        raise ValueError(f"Expected uint8 frame, got {frame.dtype}.")
                    process.stdin.write(frame.tobytes())
                    frames_written += 1
            except BrokenPipeError:
                pipe_broken = True
            finally:
                try:
                    process.stdin.close()
                except (BrokenPipeError, OSError):
                    pass

            # Do not call communicate() after closing stdin manually. Polling in
            # short intervals lets Ctrl+C and the watchdog kill be observed fast.
            while process.poll() is None:
                if cancel_event.is_set() or timeout_triggered.is_set():
                    try:
                        process.kill()
                    except (ProcessLookupError, OSError):
                        pass
                try:
                    process.wait(timeout=0.25)
                except subprocess.TimeoutExpired:
                    continue

            return_code = process.returncode

        if timeout_triggered.is_set():
            details = tail_text(stderr_log_path) if stderr_log_path else ""
            raise TimeoutError(
                f"FFmpeg exceeded {timeout_seconds:.1f} seconds and was killed.\n{details}"
            )
        if cancel_event.is_set():
            raise BatchCancelled("Batch cancellation requested.")
        if return_code != 0:
            details = tail_text(stderr_log_path) if stderr_log_path else ""
            raise RuntimeError(f"FFmpeg encoding failed (code {return_code}):\n{details}")
        if frames_written != NUM_FRAMES or pipe_broken:
            raise RuntimeError(
                "FFmpeg did not receive all frames: "
                f"received={frames_written}, expected={NUM_FRAMES}."
            )

        os.replace(temp_output_path, output_path)
        success = True

    finally:
        if watchdog is not None:
            watchdog.cancel()

        if process is not None:
            process_registry.discard(process)
            if process.poll() is None:
                try:
                    process.kill()
                except (ProcessLookupError, OSError):
                    pass
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    pass

        if not success:
            try:
                temp_output_path.unlink(missing_ok=True)
            except OSError:
                pass

        if stderr_log_path is not None:
            try:
                stderr_log_path.unlink(missing_ok=True)
            except OSError:
                pass


def process_sample(
    mask_path: Path,
    args: argparse.Namespace,
    cancel_event: threading.Event,
    process_registry: ActiveProcessRegistry,
) -> dict[str, Any]:
    """Render one multi-track sibling motion.mp4 from one masks.npy file."""
    if cancel_event.is_set():
        raise BatchCancelled("Batch cancellation requested before processing.")

    output_path = mask_path.parent / args.output_name
    if args.keep_existing and output_path.exists():
        raise SkipSample(f"{args.output_name} already exists.")

    (
        active_tracks,
        channel_stats,
        total_frames,
        channels,
        height,
        width,
        mask_shape,
        mask_layout,
    ) = extract_mask_tracks(
        mask_path=mask_path,
        centroid_mode=args.centroid_mode,
        min_mask_pixels=args.min_mask_pixels,
        min_valid_ratio=args.min_valid_ratio,
    )

    smoothed_tracks: list[tuple[int, np.ndarray]] = []
    total_spikes = 0
    stat_by_channel = {int(stat["channel_index"]): stat for stat in channel_stats}
    for channel_index, centers, _valid in active_tracks:
        smoothed_centers, spike_mask = smooth_centers(
            centers=centers,
            width=width,
            height=height,
            smooth_window=args.smooth_window,
            spike_threshold_ratio=args.spike_threshold_ratio,
        )
        stat_by_channel[channel_index]["spike_repaired_frames"] = int(spike_mask.sum())
        total_spikes += int(spike_mask.sum())
        smoothed_tracks.append((channel_index, smoothed_centers))

    save_h264_video(
        frames=generate_track_frames(
            smoothed_tracks=smoothed_tracks,
            total_mask_channels=channels,
            width=width,
            height=height,
            cancel_event=cancel_event,
        ),
        output_path=output_path,
        width=width,
        height=height,
        fps=FPS,
        ffmpeg_threads=args.ffmpeg_threads,
        timeout_seconds=args.ffmpeg_timeout_seconds,
        cancel_event=cancel_event,
        process_registry=process_registry,
    )

    active_channel_indices = [channel_index for channel_index, _ in smoothed_tracks]
    return {
        "mask_shape": mask_shape,
        "mask_layout": mask_layout,
        "mask_total_frames": total_frames,
        "mask_channels": channels,
        "active_tracks": len(smoothed_tracks),
        "active_channel_indices": active_channel_indices,
        "inactive_tracks": channels - len(smoothed_tracks),
        "width": width,
        "height": height,
        "channel_stats": channel_stats,
        "valid_mask_centers": int(
            sum(int(stat["valid_mask_frames"]) for stat in channel_stats)
        ),
        "interpolated_empty_or_invalid_frames": int(
            sum(int(stat["interpolated_empty_or_invalid_frames"]) for stat in channel_stats)
        ),
        "spike_repaired_frames": total_spikes,
    }


def submit_until_full(
    executor: ThreadPoolExecutor,
    pending: dict[Future[Any], Path],
    paths_iterator: Iterator[Path],
    max_in_flight: int,
    args: argparse.Namespace,
    cancel_event: threading.Event,
    process_registry: ActiveProcessRegistry,
) -> bool:
    """Keep only a bounded number of submitted-but-unfinished jobs."""
    more_paths_exist = True
    while len(pending) < max_in_flight:
        try:
            mask_path = next(paths_iterator)
        except StopIteration:
            more_paths_exist = False
            break

        future = executor.submit(
            process_sample,
            mask_path,
            args,
            cancel_event,
            process_registry,
        )
        pending[future] = mask_path
    return more_paths_exist


def main() -> None:
    args = parse_args()
    parent_dir = args.parent_dir.expanduser().resolve()

    if not parent_dir.is_dir():
        raise SystemExit(f"Not a directory: {parent_dir}")
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg was not found in PATH. Please install FFmpeg first.")

    mask_paths = sorted(path for path in parent_dir.rglob("masks.npy") if path.is_file())
    if not mask_paths:
        raise SystemExit(f"No masks.npy files found below: {parent_dir}")

    # Avoid nested OpenCV pools inside every Python worker thread.
    try:
        cv2.setNumThreads(1)
    except cv2.error:
        pass

    workers = args.workers
    max_in_flight = args.max_in_flight or workers * 2
    max_in_flight = max(workers, max_in_flight)

    saved = 0
    skipped = 0
    failed = 0
    cancelled = 0
    total_input_channels = 0
    total_active_tracks = 0
    total_inactive_tracks = 0
    total_valid_centers = 0
    total_interpolated = 0
    total_spikes_repaired = 0

    cancel_event = threading.Event()
    process_registry = ActiveProcessRegistry()
    pending: dict[Future[Any], Path] = {}
    paths_iterator = iter(mask_paths)
    executor = ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="mask-motion-render",
    )

    start_time = time.perf_counter()
    interrupted = False

    try:
        submit_until_full(
            executor,
            pending,
            paths_iterator,
            max_in_flight,
            args,
            cancel_event,
            process_registry,
        )

        with tqdm(
            total=len(mask_paths),
            desc="Rendering multi-track motion.mp4",
            unit="sample",
            dynamic_ncols=True,
        ) as progress:
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)

                # Main thread owns every tqdm operation. Each completed Future
                # moves the bar exactly once, regardless of success/failure/skip.
                for future in done:
                    mask_path = pending.pop(future)
                    try:
                        result = future.result()
                        saved += 1
                        total_input_channels += int(result["mask_channels"])
                        total_active_tracks += int(result["active_tracks"])
                        total_inactive_tracks += int(result["inactive_tracks"])
                        total_valid_centers += int(result["valid_mask_centers"])
                        total_interpolated += int(
                            result["interpolated_empty_or_invalid_frames"]
                        )
                        total_spikes_repaired += int(result["spike_repaired_frames"])
                    except SkipSample as exc:
                        skipped += 1
                        tqdm.write(f"[SKIP] {mask_path.parent}: {exc}")
                    except BatchCancelled:
                        cancelled += 1
                    except Exception as exc:
                        failed += 1
                        tqdm.write(
                            f"[FAIL] {mask_path.parent}: "
                            f"{type(exc).__name__}: {exc}"
                        )
                    finally:
                        progress.update(1)

                    progress.set_postfix(
                        saved=saved,
                        skipped=skipped,
                        failed=failed,
                        tracks=total_active_tracks,
                        inactive=total_inactive_tracks,
                        spikes=total_spikes_repaired,
                    )

                submit_until_full(
                    executor,
                    pending,
                    paths_iterator,
                    max_in_flight,
                    args,
                    cancel_event,
                    process_registry,
                )

    except KeyboardInterrupt:
        interrupted = True
        cancel_event.set()
        process_registry.terminate_all()
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        print("\nInterrupted: active FFmpeg processes were terminated and queued jobs cancelled.")
    finally:
        if not interrupted:
            executor.shutdown(wait=True, cancel_futures=False)

    elapsed = time.perf_counter() - start_time
    print("\nDone." if not interrupted else "\nStopped.")
    print(f"Parent directory       : {parent_dir}")
    print(f"masks.npy found        : {len(mask_paths)}")
    print(f"Videos written         : {saved}")
    print(f"Skipped                : {skipped}")
    print(f"Failed                 : {failed}")
    if cancelled:
        print(f"Cancelled              : {cancelled}")
    print(f"Centroid mode          : {args.centroid_mode}")
    print(f"Input mask channels    : {total_input_channels}")
    print(f"Rendered trajectories  : {total_active_tracks}")
    print(f"Omitted weak channels  : {total_inactive_tracks}")
    print(f"Valid mask centers     : {total_valid_centers}")
    print(f"Interpolated frames    : {total_interpolated}")
    print(f"Repaired spike frames  : {total_spikes_repaired}")
    print(f"Workers                : {workers} (FFmpeg threads/worker={args.ffmpeg_threads})")
    print(f"Elapsed                : {elapsed:.2f}s")

    if interrupted:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
