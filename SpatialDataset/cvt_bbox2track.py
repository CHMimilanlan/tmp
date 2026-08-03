#!/usr/bin/env python3
"""
Render a single-object raw-pixel xyxy bbox.npy as an LTX Motion Track IC-LoRA
tracker video.

Color encoding note
-------------------
- The sparse-track frame buffer uses the official Motion-Track RGB -> BGR numerical
  channel convention, but is encoded as standard YUV H.264 rather than libx264rgb.
  This avoids nonstandard planar-GBR (gbrp) playback artifacts such as green casts.

Input modes
-----------
1) One bbox.npy
   - Reads sibling masks.npy.
   - Writes sibling tracker.mp4.

2) One .txt file
   - Each non-empty line is an MP4 path.
   - Reads <mp4_parent>/bbox.npy and <mp4_parent>/masks.npy.
   - Writes <mp4_parent>/tracker.mp4.

3) One parent directory
   - Recursively finds bbox.npy files.
   - For each bbox.npy, reads sibling masks.npy and writes sibling tracker.mp4.

Strict data contract
--------------------
- bbox.npy must be raw, unnormalized xyxy pixel coordinates.
- Supported bbox shapes: [T, 4] or [T, 1, 4].
- masks.npy must be next to bbox.npy and must expose video dimensions as
  [T, H, W] (or [T, N, H, W]; the last two dimensions are used as H, W).
- Output tracker.mp4 is written at exactly W x H, where W/H come from masks.npy.
- bbox and masks must have the same temporal length T. This prevents silently
  creating a temporally misaligned tracker video.

Invalid bbox frames (NaN/Inf, x2 <= x1, or y2 <= y1) are treated as missing
observations. Their centers are linearly interpolated from valid frames; leading
and trailing gaps use the nearest valid center.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import cv2
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# Keep these values consistent with the supplied sparse_tracks.py-style renderer.
_MIN_RADIUS = 2
_MAX_RADIUS = 8
_MAX_TRAIL = 50
_REF_SHORT_SIDE = 1080


@dataclass(frozen=True)
class Job:
    bbox_path: Path
    masks_path: Path
    output_path: Path
    label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert raw-pixel xyxy bbox.npy trajectories to LTX Motion Track "
            "tracker videos. Input may be bbox.npy, a txt list of MP4 paths, "
            "or a parent directory."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="One bbox.npy, one .txt list, or a parent directory containing bbox.npy files.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=24.0,
        help="Tracker video FPS.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=0,
        help=(
            "Output frame count. 0 preserves the bbox/masks temporal length exactly; "
            "a positive value linearly resamples the center trajectory."
        ),
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="tracker.mp4",
        help="Tracker filename saved beside each bbox.npy.",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Also save the final integer track points beside tracker.mp4 as tracker.json.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing tracker.mp4.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first failed sample in txt/directory batch mode.",
    )
    args = parser.parse_args()

    if args.fps <= 0:
        parser.error("--fps must be positive.")
    if args.num_frames < 0:
        parser.error("--num-frames must be >= 0.")
    if not args.output_name.lower().endswith(".mp4"):
        parser.error("--output-name must end with .mp4.")
    return args


def make_job(bbox_path: Path, output_name: str, label: str | None = None) -> Job:
    bbox_path = bbox_path.expanduser()
    return Job(
        bbox_path=bbox_path,
        masks_path=bbox_path.parent / "masks.npy",
        output_path=bbox_path.parent / output_name,
        label=label or str(bbox_path),
    )


def collect_jobs(args: argparse.Namespace) -> list[Job]:
    input_path = args.input_path.expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"Input does not exist: {input_path}")

    jobs: list[Job] = []
    suffix = input_path.suffix.lower()

    if input_path.is_file() and suffix == ".npy":
        if input_path.name != "bbox.npy":
            print(
                f"[WARN] Input file is named {input_path.name!r}, not 'bbox.npy'. "
                "It will still be interpreted as raw xyxy boxes.",
                file=sys.stderr,
            )
        jobs.append(make_job(input_path, args.output_name))

    elif input_path.is_file() and suffix == ".txt":
        for line_number, raw_line in enumerate(
            input_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            mp4_path = Path(line).expanduser()
            if not mp4_path.is_absolute():
                raise ValueError(
                    f"TXT line {line_number} must be an absolute MP4 path, got: {line!r}"
                )
            if mp4_path.suffix.lower() != ".mp4":
                raise ValueError(
                    f"TXT line {line_number} must point to an MP4 file, got: {mp4_path}"
                )

            bbox_path = mp4_path.parent / "bbox.npy"
            jobs.append(
                make_job(
                    bbox_path,
                    args.output_name,
                    label=f"{input_path}:{line_number} -> {mp4_path}",
                )
            )

    elif input_path.is_dir():
        # rglob supports both immediate and nested sample directories.
        for bbox_path in sorted(input_path.rglob("bbox.npy")):
            jobs.append(make_job(bbox_path, args.output_name))

    else:
        raise ValueError(
            "input_path must be a bbox.npy file, a .txt file, or a directory. "
            f"Got: {input_path}"
        )

    # A txt list can contain repeated MP4 paths. Render each bbox only once.
    deduplicated: dict[Path, Job] = {}
    for job in jobs:
        deduplicated.setdefault(job.bbox_path.resolve(strict=False), job)

    result = list(deduplicated.values())
    if not result:
        raise RuntimeError(f"No bbox.npy files found for input: {input_path}")
    return result


def load_raw_xyxy_bboxes(bbox_path: Path) -> np.ndarray:
    """Load raw-pixel xyxy boxes with strict, unambiguous shape handling."""
    if not bbox_path.is_file():
        raise FileNotFoundError(f"bbox.npy not found: {bbox_path}")

    bboxes = np.asarray(np.load(bbox_path, allow_pickle=False))
    if bboxes.ndim == 3 and bboxes.shape[1:] == (1, 4):
        bboxes = bboxes[:, 0, :]
    elif bboxes.ndim == 1 and bboxes.shape == (4,):
        bboxes = bboxes[None, :]

    if bboxes.ndim != 2 or bboxes.shape[1] != 4:
        raise ValueError(
            "bbox.npy must have raw xyxy shape [T, 4] or [T, 1, 4]. "
            f"Got {bboxes.shape} from {bbox_path}."
        )
    if bboxes.shape[0] == 0:
        raise ValueError(f"bbox.npy has zero frames: {bbox_path}")
    if not np.issubdtype(bboxes.dtype, np.number):
        raise TypeError(f"bbox.npy must contain numeric values, got dtype={bboxes.dtype}")

    return bboxes.astype(np.float64, copy=False)


def read_masks_video_shape(masks_path: Path) -> tuple[int, int, int]:
    """Return (T, H, W) from sibling masks.npy without loading its full payload."""
    if not masks_path.is_file():
        raise FileNotFoundError(
            f"masks.npy not found beside bbox.npy: {masks_path}. "
            "This script deliberately uses masks.npy as the source of original video size."
        )

    masks = np.load(masks_path, mmap_mode="r", allow_pickle=False)
    if masks.ndim < 3:
        raise ValueError(
            f"masks.npy must expose at least [T, H, W], got shape {masks.shape} from {masks_path}"
        )

    # Standard project format is [T, H, W]. For [T, N, H, W], last two are still H/W.
    frames = int(masks.shape[0])
    height = int(masks.shape[-2])
    width = int(masks.shape[-1])
    if frames <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"Invalid masks.npy shape {masks.shape} from {masks_path}")
    return frames, height, width


def valid_raw_xyxy_mask(bboxes: np.ndarray) -> np.ndarray:
    """A valid observation is finite and satisfies x2>x1, y2>y1."""
    x1, y1, x2, y2 = bboxes.T
    return np.isfinite(bboxes).all(axis=1) & (x2 > x1) & (y2 > y1)


def validate_raw_xyxy_against_image(
    bboxes: np.ndarray,
    valid: np.ndarray,
    width: int,
    height: int,
    bbox_path: Path,
) -> None:
    """Reject normalized boxes and raw boxes inconsistent with masks.npy dimensions."""
    if not valid.any():
        raise ValueError(
            f"No valid xyxy boxes in {bbox_path}. A valid box needs finite values and x2>x1, y2>y1."
        )

    values = bboxes[valid]
    eps = 1e-6

    # The user-defined contract is explicitly non-normalized pixel xyxy.
    # Failing early prevents a nearly invisible trajectory in a large output canvas.
    if float(values.min()) >= -eps and float(values.max()) <= 1.0 + eps:
        raise ValueError(
            f"bbox.npy appears normalized to [0, 1], but this script expects raw pixel xyxy boxes. "
            f"bbox: {bbox_path}"
        )

    x1, y1, x2, y2 = bboxes.T
    tolerance = 1.0  # Accept small detector-rounding overshoots only.
    out_of_bounds = valid & (
        (x1 < -tolerance)
        | (y1 < -tolerance)
        | (x2 > (width + tolerance))
        | (y2 > (height + tolerance))
    )
    if out_of_bounds.any():
        bad = np.flatnonzero(out_of_bounds)[:5]
        examples = "; ".join(
            f"frame {int(i)}: {bboxes[i].tolist()}" for i in bad
        )
        raise ValueError(
            "Raw xyxy bbox coordinates are inconsistent with sibling masks.npy dimensions "
            f"W={width}, H={height}. Examples: {examples}. "
            "Check that bbox.npy and masks.npy originate from the same video and that bbox format is xyxy."
        )


def fill_missing_centers(centers: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fill invalid detections via 1D temporal interpolation independently for x/y."""
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) == 0:
        raise ValueError("Cannot interpolate: there are no valid bbox observations.")

    frame_indices = np.arange(len(centers), dtype=np.float64)
    filled = np.empty_like(centers, dtype=np.float64)
    for coordinate in range(2):
        filled[:, coordinate] = np.interp(
            frame_indices,
            valid_indices,
            centers[valid_indices, coordinate],
        )
    return filled


def raw_xyxy_to_centers(
    bboxes: np.ndarray,
    valid: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Convert validated raw xyxy pixels directly to raw pixel centers; never normalize."""
    x1, y1, x2, y2 = bboxes.T
    centers = np.column_stack(((x1 + x2) * 0.5, (y1 + y2) * 0.5))
    centers = fill_missing_centers(centers, valid)

    # Valid boxes may be <=1 px outside due to detector rounding. The tracker pixel must remain valid.
    centers[:, 0] = np.clip(centers[:, 0], 0.0, float(width - 1))
    centers[:, 1] = np.clip(centers[:, 1], 0.0, float(height - 1))
    return centers


def resample_centers(centers: np.ndarray, output_frames: int) -> np.ndarray:
    if output_frames <= 0 or output_frames == len(centers):
        return centers
    if output_frames == 1:
        return centers[[0]]

    old_positions = np.arange(len(centers), dtype=np.float64)
    new_positions = np.linspace(0.0, float(len(centers) - 1), output_frames)
    result = np.empty((output_frames, 2), dtype=np.float64)
    for coordinate in range(2):
        result[:, coordinate] = np.interp(new_positions, old_positions, centers[:, coordinate])
    return result


def centers_to_track(centers: np.ndarray, width: int, height: int) -> list[dict[str, int]]:
    """Round raw pixel centers to integer pixel positions in the same W/H coordinate system."""
    track: list[dict[str, int]] = []
    for center_x, center_y in centers:
        x = int(np.clip(np.rint(center_x), 0, width - 1))
        y = int(np.clip(np.rint(center_y), 0, height - 1))
        track.append({"x": x, "y": y})
    return track


def age_color_rgb(ratio: float) -> tuple[int, int, int]:
    """Same age-color mapping as the supplied renderer, with numeric RGB output."""
    if ratio <= 1.0 / 3.0:
        transition = ratio * 3.0
        r, g, b = 0.0, transition, 1.0 - transition
    elif ratio <= 2.0 / 3.0:
        transition = (ratio - 1.0 / 3.0) * 3.0
        r, g, b = transition, 1.0, 0.0
    else:
        transition = (ratio - 2.0 / 3.0) * 3.0
        r, g, b = 1.0, 1.0 - transition, 0.0
    return int(round(r * 255)), int(round(g * 255)), int(round(b * 255))


def render_resolution(
    width: int,
    height: int,
    reference_short_side: int = _REF_SHORT_SIDE,
) -> tuple[int, int, float, float]:
    """Render on a 1080-short-side canvas, then downsample, matching supplied code."""
    if height <= width:
        render_width = int(width * reference_short_side / height)
        render_height = reference_short_side
    else:
        render_width = reference_short_side
        render_height = int(height * reference_short_side / width)
    return (
        render_width,
        render_height,
        render_width / width,
        render_height / height,
    )


def generate_track_frames(
    tracks: Sequence[Sequence[dict[str, int]]],
    width: int,
    height: int,
) -> Iterator[np.ndarray]:
    """Generate sparse-track frames using the same RGB/BGR convention as supplied code."""
    if not tracks or any(len(track) == 0 for track in tracks):
        raise ValueError("tracks must contain at least one non-empty trajectory.")

    render_width, render_height, scale_x, scale_y = render_resolution(width, height)
    num_frames = max(len(track) for track in tracks)

    scaled_tracks: list[list[dict[str, float]]] = []
    for track in tracks:
        scaled_tracks.append(
            [
                {"x": point["x"] * scale_x, "y": point["y"] * scale_y}
                for point in track
            ]
        )

    for frame_index in range(num_frames):
        # OpenCV writes raw array channel values; numerically this array is RGB.
        highres_rgb = np.zeros((render_height, render_width, 3), dtype=np.uint8)
        trail_start = max(0, frame_index - _MAX_TRAIL)

        for track in scaled_tracks:
            end_index = min(frame_index, len(track) - 1)
            for point_index in range(trail_start, end_index + 1):
                point = track[point_index]
                age = frame_index - point_index
                ratio = float(np.clip(1.0 - age / _MAX_TRAIL, 0.0, 1.0))
                radius = max(
                    1,
                    int(round(_MIN_RADIUS + (_MAX_RADIUS - _MIN_RADIUS) * ratio)),
                )
                x = int(round(point["x"]))
                y = int(round(point["y"]))
                if not (0 <= x < render_width and 0 <= y < render_height):
                    continue

                cv2.circle(
                    highres_rgb,
                    center=(x, y),
                    radius=radius,
                    color=age_color_rgb(ratio),
                    thickness=-1,
                    lineType=cv2.LINE_8,
                )

        frame_rgb = cv2.resize(
            highres_rgb,
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )

        # This channel swap is deliberately retained from the user-provided LTX-compatible code.
        yield frame_rgb[..., [2, 1, 0]].copy()


def save_h264_video(
    frames: Iterator[np.ndarray],
    output_path: Path,
    width: int,
    height: int,
    fps: float,
) -> None:
    """Write an LTX-safe H.264 MP4 with a black background and faithful track colors.

    Important:
    - ``generate_track_frames`` returns the *numerical RGB bytes expected by the
      Motion-Track IC-LoRA* (it already performs the required RGB -> BGR swap).
    - Do not use ``libx264rgb`` here. That encoder commonly stores the stream as
      planar GBR (``gbrp``). Although technically valid, some players and video
      readers display such MP4s with a green cast/background.
    - Encode standard YUV H.264 instead. ``yuv444p`` preserves the tiny blue /
      green / yellow / red trajectory markers far better than yuv420p, while LTX
      decodes the video through PyAV and converts every frame back to RGB.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg was not found on PATH. Please install FFmpeg first.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")

    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        # ``frames`` are raw numerical RGB bytes. Keep this declaration as RGB24;
        # changing it to BGR24 would undo the Motion-Track channel convention.
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-an",
        # Standard YUV H.264, not planar-RGB H.264 (gbrp).
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "0",
        "-profile:v", "high444",
        "-pix_fmt", "yuv444p",
        "-colorspace", "bt709",
        "-color_primaries", "bt709",
        "-color_trc", "bt709",
        "-color_range", "tv",
        "-movflags", "+faststart",
        str(temporary_path),
    ]

    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        assert process.stdin is not None
        for frame in frames:
            if frame.dtype != np.uint8 or frame.shape != (height, width, 3):
                raise ValueError(
                    f"Invalid rendered frame: expected uint8 {(height, width, 3)}, "
                    f"got {frame.dtype} {frame.shape}"
                )
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        assert process.stderr is not None
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        return_code = process.wait()
    except Exception:
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        process.kill()
        process.wait()
        temporary_path.unlink(missing_ok=True)
        raise

    if return_code != 0:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f"FFmpeg H.264 encoding failed (code={return_code}):\n{stderr}")

    temporary_path.replace(output_path)


def process_job(job: Job, args: argparse.Namespace) -> str:
    if job.output_path.exists() and not args.overwrite:
        return "skipped"

    bboxes = load_raw_xyxy_bboxes(job.bbox_path)
    mask_frames, height, width = read_masks_video_shape(job.masks_path)

    if len(bboxes) != mask_frames:
        raise ValueError(
            "Temporal mismatch: bbox.npy and masks.npy must describe the same original video. "
            f"bbox T={len(bboxes)}, masks T={mask_frames}. "
            f"bbox={job.bbox_path}, masks={job.masks_path}"
        )

    valid = valid_raw_xyxy_mask(bboxes)
    validate_raw_xyxy_against_image(bboxes, valid, width, height, job.bbox_path)
    centers = raw_xyxy_to_centers(bboxes, valid, width, height)

    output_frames = len(centers) if args.num_frames == 0 else args.num_frames
    centers = resample_centers(centers, output_frames)
    tracks = [centers_to_track(centers, width, height)]

    save_h264_video(
        frames=generate_track_frames(tracks, width, height),
        output_path=job.output_path,
        width=width,
        height=height,
        fps=args.fps,
    )

    if args.save_json:
        job.output_path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "bbox_format": "xyxy",
                    "bbox_coordinate_space": "raw_pixel",
                    "source_size": {"width": width, "height": height},
                    "num_frames": len(tracks[0]),
                    "tracks": tracks,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return "created"


def main() -> int:
    args = parse_args()
    try:
        jobs = collect_jobs(args)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    created = 0
    skipped = 0
    failures: list[tuple[Job, Exception]] = []

    iterator = tqdm(jobs, desc="Rendering trackers", unit="sample") if tqdm else jobs
    for job in iterator:
        try:
            result = process_job(job, args)
            if result == "created":
                created += 1
            else:
                skipped += 1
        except Exception as exc:
            failures.append((job, exc))
            print(f"\n[FAILED] {job.label}\n  {exc}", file=sys.stderr)
            if args.fail_fast:
                break

    print(f"\nDone. created={created}, skipped={skipped}, failed={len(failures)}")
    if failures:
        print("Failed samples:", file=sys.stderr)
        for job, exc in failures:
            print(f"- {job.bbox_path}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
