#!/usr/bin/env python3
"""
Runnable HRIR/HRTF spatial-audio version for tmp5.py.

Usage
-----
1. Place this file beside the original ``tmp5.py``.
2. Install the additional dependencies:

       pip install scipy sofar

3. Add a valid SimpleFreeFieldHRIR SOFA file to config.yaml:

       hrir_sofa_path: /absolute/path/to/subject.sofa

4. Run this file with the same arguments used for tmp5.py:

       python tmp5_hrtf.py --dataset_root ... --config config.yaml --use_vggt

The original VGGT, source-trajectory, output, limiter, and RMS-matching code is
kept.  Only the spatial renderer and renderer metadata are replaced.

Optional YAML keys
------------------
hrir_sofa_path: /absolute/path/to/file.sofa
hrir_verify: auto
hrir_reference_distance_m: 1.0
hrir_min_distance_m: 0.25
hrir_min_gain_db: -24.0
hrir_max_gain_db: 12.0
hrir_room_tail_db: -12.0
hrir_room_tail_onset_ms: 20.0
hrir_room_tail_fade_ms: 10.0
hrir_sound_speed: 343.0
hrir_print_itd_stats: true
hrir_force_crossfade_fallback: false

Notes
-----
- HRIRs are selected by nearest angular direction.  Nearest-direction selection
  avoids phase/ITD smearing that naive time-domain averaging can cause.
- gpuRIR.simulateTrajectory crossfades the time-varying HRIR states.
- A low-level, late-only gpuRIR tail is optional.  Its direct and early response
  is removed to avoid double counting the HRIR direct path.
- SOFA Data.Delay is applied in samples, including fractional delays.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np


@dataclass
class HRTFOptions:
    sofa_path: Path | None = None
    verify: str | bool = "auto"
    reference_distance_m: float = 1.0
    min_distance_m: float = 0.25
    min_gain_db: float = -24.0
    max_gain_db: float = 12.0
    room_tail_db: float = -12.0
    room_tail_onset_ms: float = 20.0
    room_tail_fade_ms: float = 10.0
    sound_speed: float = 343.0
    print_itd_stats: bool = True
    force_crossfade_fallback: bool = False

    def validate(self) -> None:
        if self.sofa_path is None:
            raise ValueError(
                "Set hrir_sofa_path in config.yaml or TMP5_HRIR_SOFA in the environment"
            )
        self.sofa_path = self.sofa_path.expanduser().resolve()
        if not self.sofa_path.exists():
            raise FileNotFoundError(f"HRIR SOFA file does not exist: {self.sofa_path}")
        if self.reference_distance_m <= 0:
            raise ValueError("hrir_reference_distance_m must be positive")
        if self.min_distance_m <= 0:
            raise ValueError("hrir_min_distance_m must be positive")
        if self.min_gain_db > self.max_gain_db:
            raise ValueError("hrir_min_gain_db must not exceed hrir_max_gain_db")
        if self.room_tail_onset_ms < 0 or self.room_tail_fade_ms < 0:
            raise ValueError("Room-tail onset and fade durations must be non-negative")
        if self.sound_speed <= 0 or not np.isfinite(self.sound_speed):
            raise ValueError("hrir_sound_speed must be positive and finite")


@dataclass(frozen=True)
class HRIRDatabase:
    ir: np.ndarray                 # [M,2,N], ordered [left,right]
    direction_vectors: np.ndarray  # [M,3] in listener [front,left,up]
    delays: np.ndarray             # [M,2] in samples at ir sampling rate
    sampling_rate: int
    left_receiver_index: int
    right_receiver_index: int
    source_position_type: str


_OPTIONS = HRTFOptions()
_BASE: ModuleType | None = None
_HRIR_CACHE: dict[tuple[str, int], HRIRDatabase] = {}


def _load_original_tmp5() -> ModuleType:
    original_path = Path(__file__).resolve().with_name("tmp5.py")
    if not original_path.exists():
        raise FileNotFoundError(
            f"Original tmp5.py was not found beside this launcher: {original_path}"
        )

    module_name = "_tmp5_original_hrtf"
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


def _normalise_rows(vectors: np.ndarray, *, name: str) -> np.ndarray:
    x = np.asarray(vectors, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(f"{name} must have shape [T,3], got {x.shape}")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError(f"{name} contains a zero-length direction")
    return x / norms


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray) and value.size == 1:
        return _text(value.reshape(-1)[0])
    return str(value)


def _db_to_linear(value_db: float) -> float:
    return float(10.0 ** (float(value_db) / 20.0))


def _source_positions_to_vectors(
    source_position: np.ndarray,
    position_type: str,
    position_units: str,
) -> np.ndarray:
    positions = np.asarray(source_position, dtype=np.float64)
    positions = np.squeeze(positions)
    if positions.ndim == 1:
        positions = positions[None, :]
    if positions.ndim != 2 or positions.shape[1] < 3:
        raise ValueError(
            f"SOFA SourcePosition must be convertible to [M,3], got {positions.shape}"
        )
    positions = positions[:, :3]

    kind = position_type.strip().lower()
    if "spherical" in kind:
        azimuth = positions[:, 0]
        elevation = positions[:, 1]
        if "radian" not in position_units.lower():
            azimuth = np.deg2rad(azimuth)
            elevation = np.deg2rad(elevation)
        cos_elevation = np.cos(elevation)
        vectors = np.stack(
            [
                cos_elevation * np.cos(azimuth),
                cos_elevation * np.sin(azimuth),
                np.sin(elevation),
            ],
            axis=1,
        )
    elif "cartesian" in kind:
        vectors = positions
    else:
        raise ValueError(
            f"Unsupported SOFA SourcePosition_Type={position_type!r}; "
            "expected spherical or cartesian"
        )

    return _normalise_rows(vectors, name="SOFA source directions")


def _receiver_channel_order(sofa: Any, receiver_count: int) -> tuple[int, int]:
    if receiver_count != 2:
        raise ValueError(
            f"This renderer requires exactly two SOFA receivers, got {receiver_count}"
        )

    receiver_position = getattr(sofa, "ReceiverPosition", None)
    receiver_type = _text(
        getattr(sofa, "ReceiverPosition_Type", "cartesian")
    ).lower()
    if receiver_position is None or "cartesian" not in receiver_type:
        print(
            "WARNING: SOFA receiver positions are unavailable/non-cartesian; "
            "assuming Data.IR channel order [left,right]."
        )
        return 0, 1

    positions = np.asarray(receiver_position, dtype=np.float64)
    positions = np.squeeze(positions)

    # Typical SOFA layouts reduce to [R,3].  Handle extra singleton dimensions.
    if positions.ndim > 2:
        positions = positions.reshape(-1, positions.shape[-1])
    if positions.ndim != 2 or positions.shape[-1] < 3:
        print(
            "WARNING: could not interpret ReceiverPosition; assuming "
            "Data.IR channel order [left,right]."
        )
        return 0, 1

    if positions.shape[0] != receiver_count:
        if positions.shape[1] == receiver_count and positions.shape[0] >= 3:
            positions = positions.T
        else:
            print(
                "WARNING: ReceiverPosition receiver count does not match Data.IR; "
                "assuming [left,right]."
            )
            return 0, 1

    # Standard listener coordinates: +x front, +y left, +z up.
    lateral = positions[:, 1]
    if float(np.ptp(lateral)) <= 1e-8:
        print(
            "WARNING: SOFA receiver lateral positions are ambiguous; "
            "assuming Data.IR channel order [left,right]."
        )
        return 0, 1
    return int(np.argmax(lateral)), int(np.argmin(lateral))


def _normalise_delay_array(delay: np.ndarray, measurements: int, receivers: int) -> np.ndarray:
    values = np.asarray(delay, dtype=np.float64)
    values = np.squeeze(values)

    if values.ndim == 0:
        return np.full((measurements, receivers), float(values), dtype=np.float64)
    if values.ndim == 1:
        if values.size == receivers:
            return np.broadcast_to(values[None, :], (measurements, receivers)).copy()
        if values.size == measurements and receivers == 1:
            return values[:, None]
    if values.ndim == 2:
        if values.shape == (measurements, receivers):
            return values.copy()
        if values.shape == (1, receivers):
            return np.broadcast_to(values, (measurements, receivers)).copy()
        if values.shape == (receivers, measurements):
            return values.T.copy()

    raise ValueError(
        f"Cannot map SOFA Data.Delay shape {values.shape} to "
        f"[M,R]=[{measurements},{receivers}]"
    )


def _resample_hrirs(ir: np.ndarray, source_fs: int, target_fs: int) -> np.ndarray:
    if source_fs == target_fs:
        return np.asarray(ir, dtype=np.float32)

    try:
        from scipy.signal import resample_poly
    except ImportError as exc:
        raise ImportError(
            "The HRTF version needs scipy for HRIR sample-rate conversion: pip install scipy"
        ) from exc

    ratio = Fraction(int(target_fs), int(source_fs)).limit_denominator(4096)
    result = resample_poly(
        np.asarray(ir, dtype=np.float64),
        ratio.numerator,
        ratio.denominator,
        axis=-1,
    )
    return np.asarray(result, dtype=np.float32)


def _load_hrir_database(target_fs: int) -> HRIRDatabase:
    if _OPTIONS.sofa_path is None:
        raise RuntimeError("HRTF options were not initialised")
    key = (str(_OPTIONS.sofa_path), int(target_fs))
    if key in _HRIR_CACHE:
        return _HRIR_CACHE[key]

    try:
        import sofar as sf
    except ImportError as exc:
        raise ImportError(
            "The HRTF version needs sofar for SOFA loading: pip install sofar"
        ) from exc

    sofa = sf.read_sofa(
        str(_OPTIONS.sofa_path),
        verify=_OPTIONS.verify,
        verbose=False,
    )

    raw_ir = np.asarray(getattr(sofa, "Data_IR"), dtype=np.float64)
    if raw_ir.ndim != 3:
        raise ValueError(
            f"Expected SOFA Data.IR with shape [M,R,N], got {raw_ir.shape}"
        )
    measurements, receivers, _ = raw_ir.shape
    left_index, right_index = _receiver_channel_order(sofa, receivers)
    raw_ir = raw_ir[:, [left_index, right_index], :]

    source_type = _text(
        getattr(sofa, "SourcePosition_Type", "spherical")
    )
    source_units = _text(
        getattr(sofa, "SourcePosition_Units", "degree, degree, metre")
    )
    direction_vectors = _source_positions_to_vectors(
        getattr(sofa, "SourcePosition"),
        source_type,
        source_units,
    )
    if len(direction_vectors) != measurements:
        if len(direction_vectors) == 1:
            direction_vectors = np.repeat(direction_vectors, measurements, axis=0)
        else:
            raise ValueError(
                "SOFA SourcePosition count does not match Data.IR measurements: "
                f"{len(direction_vectors)} vs {measurements}"
            )

    sampling_rates = np.asarray(
        getattr(sofa, "Data_SamplingRate"), dtype=np.float64
    ).reshape(-1)
    if sampling_rates.size == 0 or np.any(~np.isfinite(sampling_rates)):
        raise ValueError("SOFA Data.SamplingRate is missing or invalid")
    if not np.allclose(sampling_rates, sampling_rates[0], rtol=0.0, atol=1e-6):
        raise ValueError("Per-measurement SOFA sampling rates are not supported")
    source_fs = int(round(float(sampling_rates[0])))
    if source_fs <= 0:
        raise ValueError(f"Invalid SOFA sampling rate: {source_fs}")

    raw_delay = getattr(sofa, "Data_Delay", np.zeros((1, receivers)))
    delays = _normalise_delay_array(raw_delay, measurements, receivers)
    delays = delays[:, [left_index, right_index]]
    if np.any(delays < -1e-7):
        raise ValueError("Negative SOFA Data.Delay values are not supported")
    delays = np.maximum(delays, 0.0) * (float(target_fs) / float(source_fs))

    ir = _resample_hrirs(raw_ir, source_fs, int(target_fs))
    database = HRIRDatabase(
        ir=ir,
        direction_vectors=direction_vectors.astype(np.float64),
        delays=delays.astype(np.float64),
        sampling_rate=int(target_fs),
        left_receiver_index=left_index,
        right_receiver_index=right_index,
        source_position_type=source_type,
    )
    _HRIR_CACHE[key] = database
    print(
        f"[HRTF renderer] loaded {measurements} HRIR directions from "
        f"{_OPTIONS.sofa_path.name} at {target_fs} Hz; "
        f"SOFA receiver channels reordered as left={left_index}, right={right_index}."
    )
    return database


def _fractional_delay(signal: np.ndarray, delay_samples: float) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    delay = float(delay_samples)
    if delay <= 1e-10:
        return x.astype(np.float32)

    output_length = len(x) + int(math.ceil(delay)) + 8
    fft_length = 1 << max(1, output_length - 1).bit_length()
    spectrum = np.fft.rfft(x, n=fft_length)
    bins = np.arange(spectrum.size, dtype=np.float64)
    phase = np.exp(-2j * np.pi * bins * delay / fft_length)
    delayed = np.fft.irfft(spectrum * phase, n=fft_length)
    return delayed[:output_length].astype(np.float32)


def _listener_local_source_vectors(
    source_xyz: np.ndarray,
    receiver_xyz: np.ndarray,
    left_to_right_dirs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source_xyz, dtype=np.float64)
    receiver = np.asarray(receiver_xyz, dtype=np.float64)
    right_axis = _normalise_rows(
        left_to_right_dirs, name="left_to_right_dirs"
    )

    if not (source.shape == receiver.shape == right_axis.shape):
        raise ValueError(
            "source_xyz, receiver_xyz, and left_to_right_dirs must all be [T,3]"
        )

    relative = source - receiver
    distances = np.linalg.norm(relative, axis=1)
    if np.any(distances <= 1e-8):
        raise ValueError("A source position coincides with the listener position")

    nominal_up = np.tile(np.asarray([0.0, 0.0, 1.0]), (len(source), 1))
    forward = np.cross(nominal_up, right_axis)
    degenerate = np.linalg.norm(forward, axis=1) <= 1e-8
    if np.any(degenerate):
        fallback_up = np.asarray([0.0, 1.0, 0.0])
        forward[degenerate] = np.cross(fallback_up, right_axis[degenerate])
    forward = _normalise_rows(forward, name="listener forward axes")
    up = _normalise_rows(np.cross(right_axis, forward), name="listener up axes")
    left_axis = -right_axis

    local = np.stack(
        [
            np.sum(relative * forward, axis=1),
            np.sum(relative * left_axis, axis=1),
            np.sum(relative * up, axis=1),
        ],
        axis=1,
    )
    local = _normalise_rows(local, name="local source directions")
    return local, distances


def _stabilise_nearest_indices(indices: np.ndarray) -> np.ndarray:
    result = np.asarray(indices, dtype=np.int64).copy()
    if len(result) < 3:
        return result
    # Remove isolated one-state angular quantisation flicker.
    for index in range(1, len(result) - 1):
        if result[index - 1] == result[index + 1] != result[index]:
            result[index] = result[index - 1]
    return result


def _select_hrir_track(
    database: HRIRDatabase,
    target_directions: np.ndarray,
    distances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    target = _normalise_rows(target_directions, name="target HRIR directions")
    similarities = np.clip(
        target @ database.direction_vectors.T,
        -1.0,
        1.0,
    )
    selected = _stabilise_nearest_indices(np.argmax(similarities, axis=1))

    selected_pairs: list[np.ndarray] = []
    peak_itd_samples: list[float] = []
    for state, measurement in enumerate(selected):
        channel_responses: list[np.ndarray] = []
        for channel in range(2):
            response = _fractional_delay(
                database.ir[measurement, channel],
                database.delays[measurement, channel],
            )
            channel_responses.append(response)

        pair_length = max(len(channel_responses[0]), len(channel_responses[1]))
        pair = np.zeros((2, pair_length), dtype=np.float32)
        for channel in range(2):
            pair[channel, : len(channel_responses[channel])] = channel_responses[channel]

        distance_gain = _OPTIONS.reference_distance_m / max(
            float(distances[state]), _OPTIONS.min_distance_m
        )
        gain_db = 20.0 * math.log10(max(distance_gain, 1e-12))
        gain_db = float(
            np.clip(gain_db, _OPTIONS.min_gain_db, _OPTIONS.max_gain_db)
        )
        pair *= _db_to_linear(gain_db)
        selected_pairs.append(pair)

        left_peak = int(np.argmax(np.abs(pair[0])))
        right_peak = int(np.argmax(np.abs(pair[1])))
        peak_itd_samples.append(float(right_peak - left_peak))

    max_length = max(pair.shape[-1] for pair in selected_pairs)
    track = np.zeros((len(selected_pairs), 2, max_length), dtype=np.float32)
    for state, pair in enumerate(selected_pairs):
        track[state, :, : pair.shape[-1]] = pair

    return track, np.asarray(peak_itd_samples, dtype=np.float64)


def _fft_or_direct_convolve(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    try:
        from scipy.signal import fftconvolve
    except ImportError:
        return np.convolve(x, h, mode="full")
    return fftconvolve(x, h, mode="full")


def _crossfade_trajectory_filter(signal: np.ndarray, rirs: np.ndarray) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    h = np.asarray(rirs, dtype=np.float64)
    if h.ndim != 3 or h.shape[1] != 2:
        raise ValueError(f"rirs must have shape [T,2,L], got {h.shape}")
    if len(x) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    states, channels, rir_length = h.shape
    output = np.zeros((len(x) + rir_length - 1, channels), dtype=np.float64)

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
        positions = np.arange(start, stop, dtype=np.float64)

        if state == 0:
            weights = (centers[1] - positions) / max(centers[1] - centers[0], 1.0)
        elif state == states - 1:
            weights = (positions - centers[-2]) / max(
                centers[-1] - centers[-2], 1.0
            )
        else:
            weights = np.empty_like(positions)
            left = positions <= centers[state]
            weights[left] = (positions[left] - centers[state - 1]) / max(
                centers[state] - centers[state - 1], 1.0
            )
            weights[~left] = (centers[state + 1] - positions[~left]) / max(
                centers[state + 1] - centers[state], 1.0
            )

        weighted = x[start:stop] * np.clip(weights, 0.0, 1.0)
        for channel in range(channels):
            filtered = _fft_or_direct_convolve(weighted, h[state, channel])
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


def _late_room_tail_rirs(
    source_xyz: np.ndarray,
    receiver_xyz: np.ndarray,
    directions: np.ndarray,
    room_size: tuple[float, float, float],
    t60: float,
    fs: int,
    ear_distance: float,
    max_rir_seconds: float,
) -> np.ndarray:
    if _BASE is None:
        raise RuntimeError("The original tmp5 module has not been loaded")

    source = np.asarray(source_xyz, dtype=np.float64)
    receiver = np.asarray(receiver_xyz, dtype=np.float64)
    right_ear = receiver + 0.5 * float(ear_distance) * directions
    left_ear = receiver - 0.5 * float(ear_distance) * directions

    room = np.asarray(room_size, dtype=np.float32)
    beta = _BASE.safe_beta_sabine(room, float(t60))
    tdiff = float(_BASE.gpuRIR.att2t_SabineEstimator(15.0, float(t60)))
    tmax = float(_BASE.gpuRIR.att2t_SabineEstimator(60.0, float(t60)))
    tmax = min(tmax, float(max_rir_seconds))
    tdiff = min(tdiff, tmax)
    try:
        nb_img = _BASE.gpuRIR.t2n(tdiff, room, c=_OPTIONS.sound_speed)
    except TypeError:
        nb_img = _BASE.gpuRIR.t2n(tdiff, room)

    tail_states: list[np.ndarray] = []
    fade_samples = int(round(_OPTIONS.room_tail_fade_ms * fs / 1000.0))
    onset_samples = int(round(_OPTIONS.room_tail_onset_ms * fs / 1000.0))

    for state in range(len(source)):
        ears = np.stack([left_ear[state], right_ear[state]], axis=0)
        rir = _BASE.gpuRIR.simulateRIR(
            room,
            beta,
            source[state].reshape(1, 3),
            ears,
            nb_img,
            tmax,
            int(fs),
            Tdiff=tdiff,
            mic_pattern="omni",
            c=_OPTIONS.sound_speed,
        )
        pair = np.asarray(rir, dtype=np.float32)[0]
        distances = np.linalg.norm(source[state][None, :] - ears, axis=1)
        direct_indices = np.rint(
            distances / _OPTIONS.sound_speed * fs
        ).astype(np.int64)

        for channel in range(2):
            start = int(direct_indices[channel]) + onset_samples
            start = int(np.clip(start, 0, pair.shape[-1]))
            envelope = np.zeros(pair.shape[-1], dtype=np.float32)
            if start < pair.shape[-1]:
                if fade_samples > 0:
                    end = min(pair.shape[-1], start + fade_samples)
                    phase = np.linspace(0.0, 1.0, end - start, endpoint=False)
                    envelope[start:end] = 0.5 - 0.5 * np.cos(np.pi * phase)
                    envelope[end:] = 1.0
                else:
                    envelope[start:] = 1.0
            pair[channel] *= envelope
        tail_states.append(pair)

    return np.stack(tail_states, axis=0).astype(np.float32)


def _geometric_itd_samples(
    source_xyz: np.ndarray,
    receiver_xyz: np.ndarray,
    directions: np.ndarray,
    ear_distance: float,
    fs: int,
) -> np.ndarray:
    right_ear = receiver_xyz + 0.5 * ear_distance * directions
    left_ear = receiver_xyz - 0.5 * ear_distance * directions
    distance_left = np.linalg.norm(source_xyz - left_ear, axis=1)
    distance_right = np.linalg.norm(source_xyz - right_ear, axis=1)
    return (distance_right - distance_left) / _OPTIONS.sound_speed * fs


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
    """Drop-in HRTF replacement for tmp5.simulate_dynamic_spatial_stereo."""
    if _BASE is None:
        raise RuntimeError("The original tmp5 module has not been loaded")
    if fs <= 0:
        raise ValueError("fs must be positive")
    if ear_distance <= 0 or ear_distance > 0.30:
        raise ValueError(
            f"ear_distance={ear_distance} is invalid or implausible for binaural rendering"
        )

    signal = np.asarray(source_signal, dtype=np.float32).reshape(-1)
    source = np.asarray(source_xyz, dtype=np.float64)
    receiver = np.asarray(receiver_xyz, dtype=np.float64)
    directions = _normalise_rows(
        left_to_right_dirs, name="left_to_right_dirs"
    )
    if not (source.shape == receiver.shape == directions.shape):
        raise ValueError(
            "source_xyz, receiver_xyz, and left_to_right_dirs must all have shape [T,3]"
        )
    if source.ndim != 2 or source.shape[1] != 3 or len(source) < 1:
        raise ValueError(f"Invalid source trajectory shape: {source.shape}")

    local_directions, distances = _listener_local_source_vectors(
        source, receiver, directions
    )
    database = _load_hrir_database(int(fs))
    hrir_track, hrir_peak_itd = _select_hrir_track(
        database, local_directions, distances
    )
    stereo = _trajectory_filter(signal, hrir_track)

    if _OPTIONS.room_tail_db > -120.0 and t60 > 0 and max_rir_seconds > 0:
        tail_rirs = _late_room_tail_rirs(
            source,
            receiver,
            directions,
            room_size,
            float(t60),
            int(fs),
            float(ear_distance),
            float(max_rir_seconds),
        )
        tail = _trajectory_filter(signal, tail_rirs)
        stereo = stereo + _db_to_linear(_OPTIONS.room_tail_db) * tail

    if _OPTIONS.print_itd_stats:
        geometric = _geometric_itd_samples(
            source,
            receiver,
            directions,
            float(ear_distance),
            int(fs),
        )
        print(
            "[HRTF renderer] ITD diagnostics in samples: "
            f"geometric(min/median/max)="
            f"{np.min(geometric):+.3f}/"
            f"{np.median(geometric):+.3f}/"
            f"{np.max(geometric):+.3f}; "
            f"selected-HRIR-peak(min/median/max)="
            f"{np.min(hrir_peak_itd):+.3f}/"
            f"{np.median(hrir_peak_itd):+.3f}/"
            f"{np.max(hrir_peak_itd):+.3f}; "
            "geometric_fraction_abs_below_0.5="
            f"{np.mean(np.abs(geometric) < 0.5):.3f}"
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
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["receiver_model"] = (
            "SOFA HRIR/HRTF direct-path binaural renderer with optional "
            "late-only gpuRIR room tail"
        )
        metadata["spatial_renderer"] = {
            "type": "sofa_hrir_dynamic_binaural",
            "sofa_path": str(_OPTIONS.sofa_path),
            "sofa_verify": _OPTIONS.verify,
            "direction_selection": "nearest angular direction with temporal crossfade",
            "reference_distance_m": _OPTIONS.reference_distance_m,
            "min_distance_m": _OPTIONS.min_distance_m,
            "distance_gain_db_limits": [
                _OPTIONS.min_gain_db,
                _OPTIONS.max_gain_db,
            ],
            "room_tail_db": _OPTIONS.room_tail_db,
            "room_tail_onset_ms": _OPTIONS.room_tail_onset_ms,
            "room_tail_fade_ms": _OPTIONS.room_tail_fade_ms,
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
        configured_path = (
            getattr(args, "hrir_sofa_path", None)
            or getattr(args, "hrir_sofa", None)
            or os.environ.get("TMP5_HRIR_SOFA")
        )
        verify_value: str | bool = getattr(args, "hrir_verify", "auto")
        if isinstance(verify_value, str):
            lower = verify_value.strip().lower()
            if lower in {"true", "yes", "1"}:
                verify_value = True
            elif lower in {"false", "no", "0"}:
                verify_value = False
            elif lower != "auto":
                raise ValueError("hrir_verify must be auto, true, or false")

        _OPTIONS = HRTFOptions(
            sofa_path=Path(configured_path) if configured_path else None,
            verify=verify_value,
            reference_distance_m=float(
                getattr(args, "hrir_reference_distance_m", 1.0)
            ),
            min_distance_m=float(
                getattr(args, "hrir_min_distance_m", 0.25)
            ),
            min_gain_db=float(
                getattr(args, "hrir_min_gain_db", -24.0)
            ),
            max_gain_db=float(
                getattr(args, "hrir_max_gain_db", 12.0)
            ),
            room_tail_db=float(
                getattr(args, "hrir_room_tail_db", -12.0)
            ),
            room_tail_onset_ms=float(
                getattr(args, "hrir_room_tail_onset_ms", 20.0)
            ),
            room_tail_fade_ms=float(
                getattr(args, "hrir_room_tail_fade_ms", 10.0)
            ),
            sound_speed=float(
                getattr(args, "hrir_sound_speed", 343.0)
            ),
            print_itd_stats=_as_bool(
                getattr(args, "hrir_print_itd_stats", True)
            ),
            force_crossfade_fallback=_as_bool(
                getattr(args, "hrir_force_crossfade_fallback", False)
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
