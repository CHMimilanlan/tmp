#!/usr/bin/env python3
"""
Batch runner for ``python -m ltx_pipelines.ic_lora_audio_full``.

Main features
-------------
1. Read samples from a JSON file.
2. Generate ``sample_num`` results with different seeds for every sample.
3. Detect visible GPUs and run one independent worker process per GPU.
4. Bind each worker to exactly one GPU through CUDA_VISIBLE_DEVICES.
5. Save generated videos and side-by-side reference/generated videos.
6. Create a versioned output directory to avoid overwriting previous runs.

The JSON field ``track`` is preserved in the run manifest, but it is not passed
into ``ltx_pipelines.ic_lora_audio_full`` because the CLI command supplied for
this task does not contain a track argument.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import re
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_DISTILLED_CHECKPOINT_PATH = (
    "/data/vjuicefs_ai_camera_jgroup_video/public_data/Video_Data/"
    "VideoAudioData/VideoAudioModels/LTX-2.3/"
    "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
)
DEFAULT_SPATIAL_UPSAMPLER_PATH = (
    "/data/vjuicefs_ai_camera_jgroup_video/public_data/Video_Data/"
    "VideoAudioData/VideoAudioModels/LTX-2.3/"
    "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
)
DEFAULT_GEMMA_ROOT = (
    "/data/vjuicefs_ai_camera_jgroup_video/public_data/Video_Data/"
    "VideoAudioData/VideoAudioModels/gemma-3-12b-it-qat-q4_0-unquantized"
)
DEFAULT_LORA_PATH = (
    "/data/vjuicefs_ai_camera_jgroup_video/public_data/Video_Data/"
    "VideoAudioData/VideoAudioModels/LTX-2.3-22b-IC-LoRA-Motion-Track-Control/"
    "ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors"
)


@dataclass(frozen=True)
class Sample:
    sample_id: str
    prompt: str
    reference_video: str | None
    track: str | None
    source_index: int


@dataclass(frozen=True)
class Task:
    task_id: int
    sample_id: str
    prompt: str
    reference_video: str | None
    track: str | None
    seed: int


@dataclass(frozen=True)
class CommonConfig:
    python_executable: str
    inference_module: str
    work_dir: str
    evaluation_mode: str
    distilled_checkpoint_path: str
    spatial_upsampler_path: str
    gemma_root: str
    lora_path: str
    lora_scale: float
    video_conditioning_scale: float
    height: int
    width: int
    num_frames: int
    frame_rate: float
    seed_arg: str
    run_root: str
    ffmpeg_bin: str
    skip_concat: bool
    skip_stage_2: bool
    show_subprocess_output: bool
    extra_cli_args: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read an evaluation JSON, distribute sample/seed tasks over all "
            "visible GPUs, and call python -m ltx_pipelines.ic_lora_audio_full independently."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required batch arguments.
    parser.add_argument("--input-json", required=True, help="Input JSON file.")
    parser.add_argument(
        "--dst-dir",
        required=True,
        help="Root directory used to create the versioned result tree.",
    )
    parser.add_argument(
        "--sample-num",
        required=True,
        type=int,
        help="Number of different seeds generated for each JSON sample.",
    )

    # Runtime / scheduling.
    parser.add_argument(
        "--work-dir",
        default=".",
        help="Working directory from which the inference script is launched.",
    )
    parser.add_argument(
        "--inference-module",
        default="ltx_pipelines.audio_full_motion_ic_lora",
        help=(
            "Python module launched as: python -m <module>. "
            "The default produces: python -m ltx_pipelines.audio_full_motion_ic_lora."
        ),
    )
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python executable used for every child CLI command.",
    )
    parser.add_argument(
        "--gpu-ids",
        default=None,
        help=(
            "Optional comma-separated GPU tokens, e.g. 0,1,2,3. When omitted, "
            "the script respects CUDA_VISIBLE_DEVICES or detects all GPUs."
        ),
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help=(
            "For every sample, seeds are base_seed, base_seed+1, ..., "
            "base_seed+sample_num-1."
        ),
    )
    parser.add_argument(
        "--seed-arg",
        default="--seed",
        help="Seed option understood by the inference script.",
    )
    parser.add_argument(
        "--show-subprocess-output",
        action="store_true",
        help=(
            "Stream child-process output directly to the terminal. By default, "
            "each task writes to its own log file to avoid interleaved multi-GPU logs."
        ),
    )

    # Fixed model/configuration parameters from the supplied shell command.
    parser.add_argument(
        "--evaluation-mode",
        default="audio_full_ic_lora",
        help="Value passed to --evaluation-mode. This argument is always forwarded.",
    )
    parser.add_argument(
        "--distilled-checkpoint-path",
        default=DEFAULT_DISTILLED_CHECKPOINT_PATH,
    )
    parser.add_argument(
        "--spatial-upsampler-path",
        default=DEFAULT_SPATIAL_UPSAMPLER_PATH,
    )
    parser.add_argument("--gemma-root", default=DEFAULT_GEMMA_ROOT)
    parser.add_argument("--lora-path", default=DEFAULT_LORA_PATH)
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--video-conditioning-scale", type=float, default=1.0)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--frame-rate", type=float, default=24.0)

    # Concatenation / forwarding.
    parser.add_argument(
        "--ffmpeg-bin",
        default="ffmpeg",
        help="ffmpeg executable used to horizontally concatenate videos.",
    )
    parser.add_argument(
        "--skip-concat",
        action="store_true",
        help="Only save generated videos; do not create reference/generated comparisons.",
    )
    parser.add_argument(
        "--skip-stage-2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Forward --skip-stage-2 to the inference command (default: enabled).",
    )
    parser.add_argument(
        "--extra-cli-arg",
        action="append",
        default=[],
        help=(
            "Append one extra token to every inference command. Repeat this option "
            "for multiple tokens, for example: --extra-cli-arg=--foo --extra-cli-arg=bar"
        ),
    )

    args = parser.parse_args()

    if args.sample_num <= 0:
        parser.error("--sample-num must be greater than 0")
    if args.base_seed < 0:
        parser.error("--base-seed must be non-negative")
    if args.base_seed + args.sample_num - 1 > 2_147_483_647:
        parser.error("The largest generated seed exceeds 2,147,483,647")
    if args.height <= 0 or args.width <= 0:
        parser.error("--height and --width must be greater than 0")
    if args.num_frames <= 0:
        parser.error("--num-frames must be greater than 0")
    if args.frame_rate <= 0:
        parser.error("--frame-rate must be greater than 0")
    if not args.seed_arg.startswith("-"):
        parser.error("--seed-arg must look like a CLI option, for example --seed")
    if not args.inference_module.strip():
        parser.error("--inference-module cannot be empty")

    return args


def resolve_path(path: str | Path, base_dir: Path | None = None) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute() and base_dir is not None:
        value = base_dir / value
    return value.resolve()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist or is not a file: {path}")


def require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise NotADirectoryError(f"{label} does not exist or is not a directory: {path}")


def sanitize_id(value: str, fallback: str) -> str:
    value = value.strip()
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    value = value.strip("._-")
    return value or fallback


def load_samples(input_json: Path) -> list[Sample]:
    with input_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        raw_samples = payload.get("samples")
    elif isinstance(payload, list):
        raw_samples = payload
    else:
        raw_samples = None

    if not isinstance(raw_samples, list):
        raise ValueError(
            "Input JSON must be either a list or an object containing a 'samples' list."
        )
    if not raw_samples:
        raise ValueError("The input JSON contains no samples.")

    samples: list[Sample] = []
    seen_ids: set[str] = set()

    for index, item in enumerate(raw_samples):
        if not isinstance(item, dict):
            raise TypeError(f"samples[{index}] must be an object")

        prompt_value = item.get("prompt", "")
        prompt = "" if prompt_value is None else str(prompt_value)

        reference_value = item.get("reference_video")
        reference_video: str | None
        if reference_value is None or str(reference_value).strip() == "":
            reference_video = None
        else:
            reference_path = resolve_path(str(reference_value))
            require_file(reference_path, f"samples[{index}].reference_video")
            reference_video = str(reference_path)

        track_value = item.get("track")
        track: str | None
        if track_value is None or str(track_value).strip() == "":
            track = None
        else:
            track_path = resolve_path(str(track_value))
            # The current inference module does not consume track, so a missing track is
            # reported as a warning rather than treated as a fatal error.
            if not track_path.exists():
                print(
                    f"[warning] samples[{index}].track does not exist and will only "
                    f"be recorded in the manifest: {track_path}",
                    file=sys.stderr,
                    flush=True,
                )
            track = str(track_path)

        explicit_id = item.get("id", item.get("sample_id"))
        if explicit_id is not None and str(explicit_id).strip():
            raw_id = str(explicit_id)
        elif reference_video is not None:
            raw_id = Path(reference_video).parent.name
        else:
            raw_id = f"sample_{index + 1:04d}"

        sample_id = sanitize_id(raw_id, fallback=f"sample_{index + 1:04d}")
        if sample_id in seen_ids:
            raise ValueError(
                f"Duplicate sample id after filename sanitization: {sample_id!r}. "
                "Add unique 'id' fields to the JSON samples."
            )
        seen_ids.add(sample_id)

        samples.append(
            Sample(
                sample_id=sample_id,
                prompt=prompt,
                reference_video=reference_video,
                track=track,
                source_index=index,
            )
        )

    return samples


def detect_gpu_tokens(explicit_gpu_ids: str | None) -> list[str]:
    """Return CUDA_VISIBLE_DEVICES tokens, respecting an existing visibility mask."""
    if explicit_gpu_ids is not None:
        tokens = [x.strip() for x in explicit_gpu_ids.split(",") if x.strip()]
        if not tokens:
            raise RuntimeError("--gpu-ids was provided but contains no GPU token")
        if len(tokens) != len(set(tokens)):
            raise RuntimeError(f"--gpu-ids contains duplicates: {tokens}")
        return tokens

    # Respect a scheduler/user-provided visibility mask. This is important on a
    # shared server: if CUDA_VISIBLE_DEVICES=2,5, workers must bind to 2 and 5,
    # not accidentally expose physical GPUs 0 and 1.
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        visible = os.environ["CUDA_VISIBLE_DEVICES"].strip()
        if visible in {"", "-1"}:
            return []
        tokens = [x.strip() for x in visible.split(",") if x.strip()]
        if len(tokens) != len(set(tokens)):
            raise RuntimeError(
                f"CUDA_VISIBLE_DEVICES contains duplicate tokens: {visible!r}"
            )
        return tokens

    try:
        import torch  # type: ignore

        count = int(torch.cuda.device_count())
        if count > 0:
            return [str(i) for i in range(count)]
    except Exception:
        pass

    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
        tokens = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        return tokens
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []


def create_versioned_run_root(dst_dir: Path, run_name: str) -> Path:
    """
    Create:
      dst_dir / run_name[_vN]

    The final directory is created atomically so two launchers cannot select the
    same version at the same time.
    """
    base_name = sanitize_id(run_name, fallback="run")

    dst_dir.mkdir(parents=True, exist_ok=True)

    version = 0
    while True:
        folder_name = base_name if version == 0 else f"{base_name}_v{version}"
        candidate = dst_dir / folder_name
        try:
            candidate.mkdir(parents=False, exist_ok=False)
            return candidate.resolve()
        except FileExistsError:
            version += 1


def build_tasks(samples: Sequence[Sample], sample_num: int, base_seed: int) -> list[Task]:
    tasks: list[Task] = []
    task_id = 0
    for sample in samples:
        for offset in range(sample_num):
            tasks.append(
                Task(
                    task_id=task_id,
                    sample_id=sample.sample_id,
                    prompt=sample.prompt,
                    reference_video=sample.reference_video,
                    track=sample.track,
                    seed=base_seed + offset,
                )
            )
            task_id += 1
    return tasks


def inference_command(task: Task, config: CommonConfig, generated_path: Path) -> list[str]:
    command = [
        config.python_executable,
        "-m",
        config.inference_module,
        "--distilled-lora",
        config.distilled_checkpoint_path,
        "--spatial-upsampler-path",
        config.spatial_upsampler_path,
        "--gemma-root",
        config.gemma_root,
        "--lora",
        config.lora_path,
        str(config.lora_scale),
    ]

    if task.reference_video is not None:
        command.extend(
            [
                "--video-conditioning",
                task.reference_video,
                str(config.video_conditioning_scale),
            ]
        )

    command.extend(
        [
            "--prompt",
            task.prompt,
            "--height",
            str(config.height),
            "--width",
            str(config.width),
            "--num-frames",
            str(config.num_frames),
            "--frame-rate",
            str(config.frame_rate),
            config.seed_arg,
            str(task.seed),
        ]
    )
    if config.skip_stage_2:
        command.append("--skip-stage-2")
    command.extend(
        [
            "--output-path",
            str(generated_path),
        ]
    )
    command.extend(config.extra_cli_args)
    return command


def concat_command(
    ffmpeg_bin: str,
    reference_video: Path,
    generated_video: Path,
    output_video: Path,
    target_height: int,
) -> list[str]:
    # Both inputs are scaled to the requested height while preserving aspect
    # ratio, then stacked horizontally. The generated video's audio is kept.
    filter_complex = (
        f"[0:v]scale=-2:{target_height}:flags=lanczos,setsar=1,"
        "setpts=PTS-STARTPTS[ref];"
        f"[1:v]scale=-2:{target_height}:flags=lanczos,setsar=1,"
        "setpts=PTS-STARTPTS[gen];"
        "[ref][gen]hstack=inputs=2:shortest=1[v]"
    )
    return [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(reference_video),
        "-i",
        str(generated_video),
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
        "-map",
        "1:a?",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        "-movflags",
        "+faststart",
        str(output_video),
    ]


def run_logged_command(
    command: Sequence[str],
    cwd: Path,
    env: dict[str, str],
    log_file: Path,
    show_subprocess_output: bool,
) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)

    command_text = shlex.join(list(command))
    header = (
        f"cwd: {cwd}\n"
        f"CUDA_VISIBLE_DEVICES: {env.get('CUDA_VISIBLE_DEVICES', '')}\n"
        f"command: {command_text}\n"
        f"{'=' * 100}\n"
    )

    if show_subprocess_output:
        print(header, flush=True)
        completed = subprocess.run(command, cwd=str(cwd), env=env, check=False)
        if completed.returncode != 0:
            raise subprocess.CalledProcessError(completed.returncode, command)
        return

    with log_file.open("a", encoding="utf-8") as log:
        log.write(header)
        log.flush()
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
        if completed.returncode != 0:
            raise subprocess.CalledProcessError(completed.returncode, command)


def read_log_tail(log_path: str | None, max_lines: int = 80) -> str:
    """Read the last lines of a task log without letting reporting failures crash the batch."""
    if not log_path:
        return ""

    path = Path(log_path)
    if not path.is_file():
        return ""

    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError as exc:
        return f"<failed to read log: {exc}>"

    return "".join(lines[-max_lines:]).rstrip()


def print_failure_details(result: dict[str, Any], max_log_lines: int = 80) -> None:
    """Print the useful error context immediately after a task is reported FAILED."""
    separator = "=" * 100
    print(separator, file=sys.stderr, flush=True)
    print(
        f"FAILED DETAIL: gpu={result.get('gpu_token')} "
        f"sample={result.get('sample_id')} seed={result.get('seed')}",
        file=sys.stderr,
        flush=True,
    )
    print(f"error: {result.get('error')}", file=sys.stderr, flush=True)

    log_path = result.get("log")
    if log_path:
        print(f"log: {log_path}", file=sys.stderr, flush=True)
        log_tail = read_log_tail(str(log_path), max_lines=max_log_lines)
        if log_tail:
            print(
                f"----- last {max_log_lines} log lines -----",
                file=sys.stderr,
                flush=True,
            )
            print(log_tail, file=sys.stderr, flush=True)
            print("----- end log tail -----", file=sys.stderr, flush=True)
    else:
        print(
            "log: unavailable (the GPU worker may have crashed before creating one)",
            file=sys.stderr,
            flush=True,
        )
    print(separator, file=sys.stderr, flush=True)


def run_one_task(task: Task, config: CommonConfig, gpu_token: str) -> dict[str, Any]:
    start_time = time.time()
    run_root = Path(config.run_root)
    stem = f"{task.sample_id}_{task.seed}"

    generated_dir = run_root / "generate" / task.sample_id
    generated_dir.mkdir(parents=True, exist_ok=True)
    generated_path = generated_dir / f"{stem}.mp4"

    concat_path: Path | None = None
    if task.reference_video is not None and not config.skip_concat:
        concat_dir = run_root / "concat" / task.sample_id
        concat_dir.mkdir(parents=True, exist_ok=True)
        concat_path = concat_dir / f"{stem}.mp4"

    log_path = run_root / "logs" / task.sample_id / f"{stem}.log"

    env = os.environ.copy()
    # Every worker sees one and only one GPU. Inside the child command that GPU
    # becomes cuda:0, which prevents two workers from accidentally sharing a GPU.
    env["CUDA_VISIBLE_DEVICES"] = gpu_token
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["PYTHONUNBUFFERED"] = "1"

    command = inference_command(task, config, generated_path)

    try:
        run_logged_command(
            command=command,
            cwd=Path(config.work_dir),
            env=env,
            log_file=log_path,
            show_subprocess_output=config.show_subprocess_output,
        )

        if not generated_path.is_file():
            raise FileNotFoundError(
                "Inference command returned successfully but did not create the "
                f"expected output: {generated_path}"
            )

        if concat_path is not None:
            ffmpeg_cmd = concat_command(
                ffmpeg_bin=config.ffmpeg_bin,
                reference_video=Path(task.reference_video),
                generated_video=generated_path,
                output_video=concat_path,
                target_height=config.height,
            )
            run_logged_command(
                command=ffmpeg_cmd,
                cwd=Path(config.work_dir),
                env=env,
                log_file=log_path,
                show_subprocess_output=config.show_subprocess_output,
            )
            if not concat_path.is_file():
                raise FileNotFoundError(
                    f"ffmpeg returned successfully but did not create: {concat_path}"
                )

        return {
            "task_id": task.task_id,
            "sample_id": task.sample_id,
            "seed": task.seed,
            "gpu_token": gpu_token,
            "success": True,
            "generated_video": str(generated_path),
            "concat_video": str(concat_path) if concat_path is not None else None,
            "log": str(log_path),
            "duration_seconds": round(time.time() - start_time, 3),
            "error": None,
        }
    except Exception as exc:
        error_text = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        with log_path.open("a", encoding="utf-8") as log:
            log.write("\n" + "=" * 100 + "\n")
            log.write("TASK FAILED\n")
            log.write(traceback.format_exc())

        return {
            "task_id": task.task_id,
            "sample_id": task.sample_id,
            "seed": task.seed,
            "gpu_token": gpu_token,
            "success": False,
            "generated_video": str(generated_path) if generated_path.exists() else None,
            "concat_video": str(concat_path) if concat_path and concat_path.exists() else None,
            "log": str(log_path),
            "duration_seconds": round(time.time() - start_time, 3),
            "error": error_text,
        }


def gpu_worker(
    gpu_token: str,
    task_queue: Any,
    result_queue: Any,
    config: CommonConfig,
) -> None:
    while True:
        task = task_queue.get()
        if task is None:
            break
        result = run_one_task(task=task, config=config, gpu_token=gpu_token)
        result_queue.put(result)


def collect_results(
    processes: Sequence[mp.Process],
    result_queue: Any,
    tasks: Sequence[Task],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    while any(process.is_alive() for process in processes):
        try:
            result = result_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        results.append(result)
        status = "done" if result["success"] else "FAILED"
        print(
            f"[{len(results):>4}/{len(tasks)}] {status:<6} "
            f"gpu={result['gpu_token']} sample={result['sample_id']} "
            f"seed={result['seed']}",
            flush=True,
        )
        if not result["success"]:
            print_failure_details(result)

    for process in processes:
        process.join()

    while True:
        try:
            result = result_queue.get_nowait()
        except queue.Empty:
            break
        results.append(result)
        status = "done" if result["success"] else "FAILED"
        print(
            f"[{len(results):>4}/{len(tasks)}] {status:<6} "
            f"gpu={result['gpu_token']} sample={result['sample_id']} "
            f"seed={result['seed']}",
            flush=True,
        )
        if not result["success"]:
            print_failure_details(result)

    completed_task_ids = {int(result["task_id"]) for result in results}
    for task in tasks:
        if task.task_id not in completed_task_ids:
            missing_result = {
                "task_id": task.task_id,
                "sample_id": task.sample_id,
                "seed": task.seed,
                "gpu_token": None,
                "success": False,
                "generated_video": None,
                "concat_video": None,
                "log": None,
                "duration_seconds": None,
                "error": "No result was returned; a GPU worker may have crashed.",
            }
            results.append(missing_result)
            print(
                f"[{len(results):>4}/{len(tasks)}] FAILED "
                f"gpu=None sample={task.sample_id} seed={task.seed}",
                flush=True,
            )
            print_failure_details(missing_result)

    return sorted(results, key=lambda item: int(item["task_id"]))


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    temporary.replace(path)


def validate_runtime(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    input_json = resolve_path(args.input_json)
    dst_dir = resolve_path(args.dst_dir)
    work_dir = resolve_path(args.work_dir)

    require_file(input_json, "--input-json")
    require_dir(work_dir, "--work-dir")

    require_file(
        resolve_path(args.distilled_checkpoint_path),
        "--distilled-checkpoint-path",
    )
    require_file(resolve_path(args.spatial_upsampler_path), "--spatial-upsampler-path")
    require_dir(resolve_path(args.gemma_root), "--gemma-root")
    require_file(resolve_path(args.lora_path), "--lora-path")

    if not args.skip_concat and shutil.which(args.ffmpeg_bin) is None:
        # Also accept an explicit executable path that is not on PATH.
        ffmpeg_path = Path(args.ffmpeg_bin).expanduser()
        if not (ffmpeg_path.is_file() and os.access(ffmpeg_path, os.X_OK)):
            raise FileNotFoundError(
                f"ffmpeg executable was not found: {args.ffmpeg_bin}. "
                "Install ffmpeg, pass --ffmpeg-bin, or use --skip-concat."
            )

    return input_json, dst_dir, work_dir


def main() -> int:
    args = parse_args()

    try:
        input_json, dst_dir, work_dir = validate_runtime(args)
        samples = load_samples(input_json)
        gpu_tokens = detect_gpu_tokens(args.gpu_ids)
        if not gpu_tokens:
            raise RuntimeError(
                "No visible GPU was detected. Check CUDA_VISIBLE_DEVICES, PyTorch CUDA, "
                "nvidia-smi, or pass --gpu-ids explicitly."
            )

        run_root = create_versioned_run_root(dst_dir=dst_dir, run_name=input_json.stem)
        (run_root / "generate").mkdir(parents=True, exist_ok=True)
        (run_root / "logs").mkdir(parents=True, exist_ok=True)
        if any(sample.reference_video is not None for sample in samples) and not args.skip_concat:
            (run_root / "concat").mkdir(parents=True, exist_ok=True)

        tasks = build_tasks(
            samples=samples,
            sample_num=args.sample_num,
            base_seed=args.base_seed,
        )

        config = CommonConfig(
            python_executable=str(resolve_path(args.python_executable))
            if os.path.sep in args.python_executable
            else args.python_executable,
            inference_module=args.inference_module.strip(),
            work_dir=str(work_dir),
            evaluation_mode=args.evaluation_mode,
            distilled_checkpoint_path=str(resolve_path(args.distilled_checkpoint_path)),
            spatial_upsampler_path=str(resolve_path(args.spatial_upsampler_path)),
            gemma_root=str(resolve_path(args.gemma_root)),
            lora_path=str(resolve_path(args.lora_path)),
            lora_scale=args.lora_scale,
            video_conditioning_scale=args.video_conditioning_scale,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            frame_rate=args.frame_rate,
            seed_arg=args.seed_arg,
            run_root=str(run_root),
            ffmpeg_bin=args.ffmpeg_bin,
            skip_concat=args.skip_concat,
            skip_stage_2=args.skip_stage_2,
            show_subprocess_output=args.show_subprocess_output,
            extra_cli_args=tuple(args.extra_cli_arg),
        )

        run_manifest = {
            "created_at_unix": time.time(),
            "input_json": str(input_json),
            "run_root": str(run_root),
            "gpu_tokens": gpu_tokens,
            "num_gpus": len(gpu_tokens),
            "sample_num": args.sample_num,
            "base_seed": args.base_seed,
            "num_samples": len(samples),
            "num_tasks": len(tasks),
            "samples": [asdict(sample) for sample in samples],
            "config": asdict(config),
        }
        dump_json(run_root / "run_config.json", run_manifest)

        print(f"Output root : {run_root}", flush=True)
        print(f"Visible GPUs: {gpu_tokens}", flush=True)
        print(
            f"Samples/Tasks: {len(samples)} samples x {args.sample_num} seeds "
            f"= {len(tasks)} tasks",
            flush=True,
        )

        ctx = mp.get_context("spawn")
        task_queue = ctx.Queue()
        result_queue = ctx.Queue()

        processes: list[mp.Process] = []
        for worker_index, gpu_token in enumerate(gpu_tokens):
            process = ctx.Process(
                target=gpu_worker,
                args=(gpu_token, task_queue, result_queue, config),
                name=f"gpu-worker-{worker_index}-device-{gpu_token}",
            )
            process.start()
            processes.append(process)

        for task in tasks:
            task_queue.put(task)
        for _ in processes:
            task_queue.put(None)

        try:
            results = collect_results(
                processes=processes,
                result_queue=result_queue,
                tasks=tasks,
            )
        except KeyboardInterrupt:
            print("\nInterrupted: terminating all GPU workers...", file=sys.stderr)
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join()
            raise

        process_exit_codes = {
            process.name: process.exitcode for process in processes
        }
        summary = {
            "run_root": str(run_root),
            "total": len(results),
            "succeeded": sum(bool(item["success"]) for item in results),
            "failed": sum(not bool(item["success"]) for item in results),
            "worker_exit_codes": process_exit_codes,
            "results": results,
        }
        dump_json(run_root / "results.json", summary)

        failed = [item for item in results if not item["success"]]
        print(
            f"Finished: {summary['succeeded']} succeeded, {summary['failed']} failed.",
            flush=True,
        )
        print(f"Result manifest: {run_root / 'results.json'}", flush=True)

        if failed:
            print("Failed tasks:", file=sys.stderr)
            for item in failed:
                print(
                    f"  sample={item['sample_id']} seed={item['seed']} "
                    f"gpu={item['gpu_token']} error={item['error']} log={item['log']}",
                    file=sys.stderr,
                )
            return 1

        if any(code not in (0, None) for code in process_exit_codes.values()):
            print(
                f"At least one GPU worker exited abnormally: {process_exit_codes}",
                file=sys.stderr,
            )
            return 1

        return 0

    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
