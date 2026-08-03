#!/usr/bin/env python3
"""
Runnable no-HRTF spatial-audio version for tmp5.py.

Usage
-----
Place this file beside the original ``tmp5.py`` and run this file with the same
arguments that you normally pass to tmp5.py:

    python tmp5_no_hrtf.py --dataset_root ... --config config.yaml --use_vggt

This launcher leaves the original data/VGGT/RMS pipeline unchanged and replaces
only ``simulate_dynamic_spatial_stereo``.  The replacement:

1. keeps the two physically separated virtual-ear receivers;
2. uses gpuRIR.simulateTrajectory for smooth time-varying filtering;
3. falls back to a partition-of-unity crossfade renderer when needed;
4. optionally suppresses late diffuse reverberation so direct-path ITD remains
   measurable by GCC-PHAT;
5. reports the geometrically expected ITD in samples.

Optional YAML keys
------------------
binaural_early_window_ms: 8.0
binaural_transition_ms: 8.0
binaural_early_gain_db: 0.0
binaural_late_gain_db: -6.0
binaural_sound_speed: 343.0
binaural_print_itd_stats: true
binaural_force_crossfade_fallback: false
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np


@dataclass
class NoHRTFOptions:
    early_window_ms: float = 8.0
    transition_ms: float = 8.0
    early_gain_db: float = 0.0
    late_gain_db: float = -6.0
    sound_speed: float = 343.0
    print_itd_stats: bool = True
    force_crossfade_fallback: bool = False

    def validate(self) -> None:
        if self.early_window_ms < 0:
            raise ValueError("binaural_early_window_ms must be non-negative")
        if self.transition_ms < 0:
            raise ValueError("binaural_transition_ms must be non-negative")
        if not np.isfinite(self.early_gain_db):
            raise ValueError("binaural_early_gain_db must be finite")
        if not np.isfinite(self.late_gain_db):
            raise ValueError("binaural_late_gain_db must be finite")
        if self.sound_speed <= 0 or not np.isfinite(self.sound_speed):
            raise ValueError("binaural_sound_speed must be positive and finite")


_OPTIONS = NoHRTFOptions()
_BASE: ModuleType | None = None


def _load_original_tmp5() -> ModuleType:
    original_path = Path(__file__).resolve().with_name("tmp5.py")
    if not original_path.exists():
        raise FileNotFoundError(
            f"Original tmp5.py was not found beside this launcher: {original_path}"
        )

    module_name = "_tmp5_original_no_hrtf"
    spec = importlib.util.spec_from_file_location(module_name, original_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import specification for {original_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off", ""}:
            return False
    raise ValueError(f"Cannot interpret {value!r} as a boolean")


def _db_to_linear(db_value: float) -> float:
    return float(10.0 ** (float(db_value) / 20.0))


def _normalise_rows(vectors: np.ndarray, *, name: str) -> np.ndarray:
    x = np.asarray(vectors, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(f"{name} must have shape [T,3], got {x.shape}")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError(f"{name} contains a zero-length direction")
    return x / norms


def _validate_scene_arrays(
    source_xyz: np.ndarray,
    receiver_xyz: np.ndarray,
    left_to_right_dirs: np.ndarray,
    ear_distance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = np.asarray(source_xyz, dtype=np.float64)
    receiver = np.asarray(receiver_xyz, dtype=np.float64)
    directions = _normalise_rows(left_to_right_dirs, name="left_to_right_dirs")

    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f"source_xyz must have shape [T,3], got {source.shape}")
    if receiver.ndim != 2 or receiver.shape[1] != 3:
        raise ValueError(f"receiver_xyz must have shape [T,3], got {receiver.shape}")
    if not (len(source) == len(receiver) == len(directions)):
        raise ValueError("Source, receiver, and orientation trajectory lengths must match")
    if len(source) < 1:
        raise ValueError("At least one trajectory state is required")
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(receiver)):
        raise ValueError("Source and receiver trajectories must be finite")
    if not np.isfinite(ear_distance) or ear_distance <= 0:
        raise ValueError("ear_distance must be positive and finite")
    if ear_distance > 0.30:
        raise ValueError(
            f"ear_distance={ear_distance:.3f} m is not a plausible human interaural distance"
        )

    return source, receiver, directions


def _geometric_itd_samples(
    source_xyz: np.ndarray,
    receiver_xyz: np.ndarray,
    directions: np.ndarray,
    ear_distance: float,
    fs: int,
    sound_speed: float,
) -> np.ndarray:
    right_ear = receiver_xyz + 0.5 * float(ear_distance) * directions
    left_ear = receiver_xyz - 0.5 * float(ear_distance) * directions
    distance_left = np.linalg.norm(source_xyz - left_ear, axis=1)
    distance_right = np.linalg.norm(source_xyz - right_ear, axis=1)

    # Positive means the right ear receives the direct sound later.
    return (distance_right - distance_left) / float(sound_speed) * int(fs)


def _shape_early_and_late_rir(
    rir: np.ndarray,
    direct_index: int,
    fs: int,
    options: NoHRTFOptions,
) -> np.ndarray:
    """Change only the direct/late balance; do not move either ear's RIR in time."""
    h = np.asarray(rir, dtype=np.float64).copy()
    if h.ndim != 1:
        raise ValueError(f"One RIR channel must be one-dimensional, got {h.shape}")

    early_end = int(direct_index) + int(round(options.early_window_ms * fs / 1000.0))
    transition = int(round(options.transition_ms * fs / 1000.0))
    early_end = int(np.clip(early_end, 0, len(h)))

    early_gain = _db_to_linear(options.early_gain_db)
    late_gain = _db_to_linear(options.late_gain_db)
    gains = np.full(len(h), late_gain, dtype=np.float64)

    if early_end > 0:
        gains[:early_end] = early_gain

    if transition > 0 and early_end < len(h):
        transition_end = min(len(h), early_end + transition)
        phase = np.linspace(0.0, 1.0, transition_end - early_end, endpoint=False)
        # Raised-cosine transition from early_gain to late_gain.
        blend = 0.5 - 0.5 * np.cos(np.pi * phase)
        gains[early_end:transition_end] = (
            (1.0 - blend) * early_gain + blend * late_gain
        )

    return (h * gains).astype(np.float32)


def _fft_or_direct_convolve(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    try:
        from scipy.signal import fftconvolve
    except ImportError:
        return np.convolve(x, h, mode="full")
    return fftconvolve(x, h, mode="full")


def _crossfade_trajectory_filter(
    signal: np.ndarray,
    rirs: np.ndarray,
) -> np.ndarray:
    """Smooth time-varying FIR filtering using hat functions that sum to one."""
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    h = np.asarray(rirs, dtype=np.float64)
    if h.ndim != 3:
        raise ValueError(f"rirs must have shape [T,C,L], got {h.shape}")
    if h.shape[1] != 2:
        raise ValueError(f"Expected two receiver channels, got {h.shape[1]}")
    if len(x) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    states, channels, rir_len = h.shape
    output = np.zeros((len(x) + rir_len - 1, channels), dtype=np.float64)

    if states == 1:
        for channel in range(channels):
            output[:, channel] = _fft_or_direct_convolve(x, h[0, channel])
        return output[: len(x)].astype(np.float32)

    centers = np.linspace(0.0, float(len(x) - 1), states)
    for state in range(states):
        left_support = 0.0 if state == 0 else centers[state - 1]
        right_support = float(len(x) - 1) if state == states - 1 else centers[state + 1]
        start = max(0, int(math.floor(left_support)))
        stop = min(len(x), int(math.ceil(right_support)) + 1)
        sample_positions = np.arange(start, stop, dtype=np.float64)

        if state == 0:
            denominator = max(centers[1] - centers[0], 1.0)
            weights = (centers[1] - sample_positions) / denominator
        elif state == states - 1:
            denominator = max(centers[-1] - centers[-2], 1.0)
            weights = (sample_positions - centers[-2]) / denominator
        else:
            weights = np.empty_like(sample_positions)
            left_side = sample_positions <= centers[state]
            left_denominator = max(centers[state] - centers[state - 1], 1.0)
            right_denominator = max(centers[state + 1] - centers[state], 1.0)
            weights[left_side] = (
                sample_positions[left_side] - centers[state - 1]
            ) / left_denominator
            weights[~left_side] = (
                centers[state + 1] - sample_positions[~left_side]
            ) / right_denominator

        weights = np.clip(weights, 0.0, 1.0)
        weighted_block = x[start:stop] * weights
        for channel in range(channels):
            filtered = _fft_or_direct_convolve(weighted_block, h[state, channel])
            output[start : start + len(filtered), channel] += filtered

    return output[: len(x)].astype(np.float32)


def _trajectory_filter(signal: np.ndarray, rirs: np.ndarray) -> np.ndarray:
    if _BASE is None:
        raise RuntimeError("The original tmp5 module has not been loaded")

    if not _OPTIONS.force_crossfade_fallback:
        simulate_trajectory = getattr(_BASE.gpuRIR, "simulateTrajectory", None)
        if callable(simulate_trajectory):
            rendered = simulate_trajectory(
                np.asarray(signal, dtype=np.float32),
                np.asarray(rirs, dtype=np.float32),
            )
            rendered = np.asarray(rendered, dtype=np.float32)
            if rendered.ndim == 2 and rendered.shape[0] == 2 and rendered.shape[1] != 2:
                rendered = rendered.T
            if rendered.ndim != 2 or rendered.shape[1] != 2:
                raise RuntimeError(
                    "gpuRIR.simulateTrajectory returned an unexpected shape: "
                    f"{rendered.shape}"
                )
            if len(rendered) < len(signal):
                rendered = np.pad(rendered, ((0, len(signal) - len(rendered)), (0, 0)))
            return rendered[: len(signal)]

    return _crossfade_trajectory_filter(signal, rirs)


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
    """Drop-in replacement for tmp5.simulate_dynamic_spatial_stereo."""
    if _BASE is None:
        raise RuntimeError("The original tmp5 module has not been loaded")
    if fs <= 0:
        raise ValueError("fs must be positive")
    if t60 <= 0 or not np.isfinite(t60):
        raise ValueError("t60 must be positive and finite")
    if max_rir_seconds <= 0 or not np.isfinite(max_rir_seconds):
        raise ValueError("max_rir_seconds must be positive and finite")

    signal = np.asarray(source_signal, dtype=np.float32).reshape(-1)
    source, receiver, directions = _validate_scene_arrays(
        source_xyz, receiver_xyz, left_to_right_dirs, ear_distance
    )

    right_ear = receiver + 0.5 * float(ear_distance) * directions
    left_ear = receiver - 0.5 * float(ear_distance) * directions

    room = np.asarray(room_size, dtype=np.float32)
    if room.shape != (3,) or np.any(room <= 0):
        raise ValueError(f"room_size must contain three positive values, got {room_size}")

    beta = _BASE.safe_beta_sabine(room, float(t60))
    tdiff = float(_BASE.gpuRIR.att2t_SabineEstimator(15.0, float(t60)))
    tmax = float(_BASE.gpuRIR.att2t_SabineEstimator(60.0, float(t60)))
    tmax = min(tmax, float(max_rir_seconds))
    tdiff = min(tdiff, tmax)
    if tmax <= 0:
        raise ValueError("The computed RIR duration is not positive")

    try:
        nb_img = _BASE.gpuRIR.t2n(tdiff, room, c=_OPTIONS.sound_speed)
    except TypeError:
        nb_img = _BASE.gpuRIR.t2n(tdiff, room)

    rir_states: list[np.ndarray] = []
    for index in range(len(source)):
        positions = np.stack([left_ear[index], right_ear[index]], axis=0)
        kwargs: dict[str, object] = {
            "Tdiff": tdiff,
            "mic_pattern": str(mic_pattern),
            "c": _OPTIONS.sound_speed,
        }
        if str(mic_pattern) != "omni":
            kwargs["orV_rcv"] = np.stack(
                [-directions[index], directions[index]], axis=0
            ).astype(np.float32)

        rir = _BASE.gpuRIR.simulateRIR(
            room,
            beta,
            source[index].reshape(1, 3),
            positions,
            nb_img,
            tmax,
            int(fs),
            **kwargs,
        )
        rir_pair = np.asarray(rir, dtype=np.float32)
        if rir_pair.ndim != 3 or rir_pair.shape[0] != 1 or rir_pair.shape[1] != 2:
            raise RuntimeError(
                "gpuRIR.simulateRIR returned an unexpected shape: "
                f"{rir_pair.shape}; expected [1,2,L]"
            )
        rir_pair = rir_pair[0]

        distances = np.linalg.norm(source[index][None, :] - positions, axis=1)
        direct_indices = np.rint(
            distances / _OPTIONS.sound_speed * int(fs)
        ).astype(np.int64)
        shaped = np.stack(
            [
                _shape_early_and_late_rir(
                    rir_pair[channel],
                    int(direct_indices[channel]),
                    int(fs),
                    _OPTIONS,
                )
                for channel in range(2)
            ],
            axis=0,
        )
        rir_states.append(shaped)

    rirs = np.stack(rir_states, axis=0).astype(np.float32)
    stereo = _trajectory_filter(signal, rirs)

    if _OPTIONS.print_itd_stats:
        itd_samples = _geometric_itd_samples(
            source,
            receiver,
            directions,
            float(ear_distance),
            int(fs),
            _OPTIONS.sound_speed,
        )
        print(
            "[No-HRTF renderer] geometric direct-path ITD samples: "
            f"min={np.min(itd_samples):+.3f}, "
            f"median={np.median(itd_samples):+.3f}, "
            f"max={np.max(itd_samples):+.3f}, "
            f"mean_abs={np.mean(np.abs(itd_samples)):.3f}, "
            "fraction_abs_below_0.5="
            f"{np.mean(np.abs(itd_samples) < 0.5):.3f}"
        )

    if normalize and stereo.size:
        peak = float(np.max(np.abs(stereo)))
        if peak > 0.99:
            stereo = stereo * (0.99 / max(peak, 1e-12))

    return np.asarray(stereo, dtype=np.float32)


def _update_metadata_after_process(args: Any) -> None:
    metadata_path = Path(args.video).resolve().parent / "metadata.json"
    if not metadata_path.exists():
        return
    try:
        import json

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["receiver_model"] = (
            "two separated virtual-ear receivers rendered with gpuRIR; "
            "smooth time-varying trajectory filtering; no HRTF/HRIR"
        )
        metadata["spatial_renderer"] = {
            "type": "gpuRIR_two_receiver_no_hrtf",
            "early_window_ms": _OPTIONS.early_window_ms,
            "transition_ms": _OPTIONS.transition_ms,
            "early_gain_db": _OPTIONS.early_gain_db,
            "late_gain_db": _OPTIONS.late_gain_db,
            "sound_speed_m_s": _OPTIONS.sound_speed,
            "trajectory_filter": (
                "crossfade_fallback"
                if _OPTIONS.force_crossfade_fallback
                else "gpuRIR.simulateTrajectory"
            ),
        }
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"WARNING: could not augment metadata.json: {exc}")


def main() -> None:
    global _BASE, _OPTIONS
    _BASE = _load_original_tmp5()

    original_parse_args = _BASE.parse_args_with_config
    original_process = _BASE.process

    def patched_parse_args() -> Any:
        global _OPTIONS
        args = original_parse_args()
        _OPTIONS = NoHRTFOptions(
            early_window_ms=float(
                getattr(args, "binaural_early_window_ms", 8.0)
            ),
            transition_ms=float(
                getattr(args, "binaural_transition_ms", 8.0)
            ),
            early_gain_db=float(
                getattr(args, "binaural_early_gain_db", 0.0)
            ),
            late_gain_db=float(
                getattr(args, "binaural_late_gain_db", -6.0)
            ),
            sound_speed=float(
                getattr(args, "binaural_sound_speed", 343.0)
            ),
            print_itd_stats=_as_bool(
                getattr(args, "binaural_print_itd_stats", True)
            ),
            force_crossfade_fallback=_as_bool(
                getattr(args, "binaural_force_crossfade_fallback", False)
            ),
        )
        _OPTIONS.validate()
        return args

    def patched_process(args: Any) -> None:
        original_process(args)
        _update_metadata_after_process(args)

    _BASE.parse_args_with_config = patched_parse_args
    _BASE.process = patched_process
    _BASE.simulate_dynamic_spatial_stereo = simulate_dynamic_spatial_stereo
    _BASE.main()


if __name__ == "__main__":
    main()
