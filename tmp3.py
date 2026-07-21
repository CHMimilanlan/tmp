#!/usr/bin/env python3
"""Generate diverse LTX Motion-Track IC-LoRA prompt/track specifications.

This tool creates a controlled synthetic AV dataset manifest.  For each sample it:
  1) samples a visible sound-source / action / environment under quotas;
  2) samples a trajectory primitive plus a speed profile;
  3) writes an LTX-compatible dense sparse-track JSON;
  4) renders an inspection/control video using the sparse_tracks.py high-res
     blue->green->yellow->red trail convention and RGB->BGR channel swap;
  5) optionally calls an OpenAI-compatible Qwen endpoint to write a polished LTX prompt;
  6) writes a JSONL manifest and per-sample metadata.

The exact sparse-track JSON written by this script is:
    [[{"x": 12, "y": 34}, ...]]
where the outer list is the list of tracks and each inner list has one point per frame.

Example:
    python generate_dataset_specs.py --config config.yaml --count 32 --qwen off
    python generate_dataset_specs.py --config config.yaml --count 512 --qwen on
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import random
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import requests
import yaml
from tqdm import tqdm

LOGGER = logging.getLogger("ltx_prompt_factory")

# Mirrors the public ComfyUI-LTXVideo sparse_tracks.py renderer.
_MIN_RADIUS = 2
_MAX_RADIUS = 8
_MAX_TRAIL = 50
_REF_SHORT_SIDE = 1080

SYSTEM_PROMPT = """You are a dataset prompt designer for controlled audio-video generation.

A separate motion-track controller determines the exact screen-space path of the main subject.
Do not specify left, right, top, bottom, start position, end position, or exact screen direction.
Do not mention motion tracks, trajectories, IC-LoRA, control maps, reference videos, or coordinates.

Write prompts for a five-second, single continuous, realistic audio-video clip. The camera must
remain locked off and static. There must be exactly one clearly visible dominant moving sound
source. Its dominant sound must be causally produced by the visible subject. Ambient sound may
exist but must be secondary. Do not use dominant music, cuts, transitions, logos, subtitles,
readable text, or implausible physics.

Return JSON only, following the requested schema exactly.
"""

FORBIDDEN_PROMPT_PATTERNS = [
    r"\bmotion[- ]?track\b",
    r"\btrajectory\b",
    r"\bic[- ]?lora\b",
    r"\bcontrol (?:map|signal)\b",
    r"\breference video\b",
    r"\bfrom left to right\b",
    r"\bfrom right to left\b",
    r"\bleft[- ]to[- ]right\b",
    r"\bright[- ]to[- ]left\b",
    r"\btop[- ]to[- ]bottom\b",
    r"\bbottom[- ]to[- ]top\b",
    r"\bcamera (?:pan|pans|panning|tilt|tilts|tilting|zoom|zooms|zooming|dolly|dollies|track|tracks|follow|follows)\b",
    r"\bhandheld\b",
    r"\bscene cut\b",
    r"\bcut to\b",
    r"\btransition\b",
]
STATIC_CAMERA_HINTS = (
    "camera remains",
    "camera stays",
    "camera is locked",
    "locked-off",
    "locked off",
    "completely static",
    "static camera",
)


@dataclass
class TrajectoryResult:
    primitive: str
    speed_profile: str
    direction: str
    control_points: list[dict[str, float]]
    dense_points: list[dict[str, int]]
    start_zone: str
    end_zone: str
    motion_label: str
    motion_magnitude: str
    displacement_norm: float
    path_length_norm: float


def build_qwen_user_prompt(plans: list[dict[str, Any]]) -> str:
    """Build one strict JSON-writing prompt for a group of planned samples."""
    schema = {
        "items": [
            {
                "sample_id": "same sample_id as input",
                "ltx_prompt": "3 to 5 English sentences only",
            }
        ]
    }
    user_payload = {
        "task": "Write one polished LTX prompt per input plan.",
        "output_schema": schema,
        "plans": [
            {
                "sample_id": p["sample_id"],
                "primary_emitter": p["primary_emitter"],
                "action": p["action"],
                "causal_sounds": p["causal_sounds"],
                "sound_temporal_pattern": p["sound_temporal_pattern"],
                "environment": p["environment"],
                "acoustic_environment": p["acoustic_environment"],
                "lighting": p["lighting"],
                "visual_style": p["visual_style"],
                "ambient_sound": p["ambient_sound"],
                "motion_description": "The subject moves smoothly along a natural route.",
                "camera_requirement": "The camera remains locked off and completely static.",
            }
            for p in plans
        ],
        "quality_rules": [
            "Use 3 to 5 concise English sentences per prompt.",
            "Describe the visible subject, the action, the causal sound, ambient sound, lighting, and a locked-off static camera.",
            "Do not introduce additional dominant sound sources or a moving camera.",
            "Do not mention exact screen-space direction or coordinates.",
            "Do not mention motion tracks, trajectories, IC-LoRA, control maps, reference videos, or coordinates.",
            "Keep the source and environment faithful to each plan.",
        ],
    }
    return json.dumps(user_payload, ensure_ascii=False)


def parse_qwen_items(text: str) -> dict[str, str]:
    parsed = parse_json_loose(text)
    items = parsed.get("items", parsed.get("candidates", [])) if isinstance(parsed, dict) else parsed
    if not isinstance(items, list):
        raise ValueError("Qwen response has no list field named 'items' or 'candidates'.")
    out: dict[str, str] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        sample_id = item.get("sample_id")
        prompt = item.get("ltx_prompt")
        if isinstance(sample_id, str) and isinstance(prompt, str):
            out[sample_id] = normalize_space(prompt)
    return out


class QwenHTTPClient:
    """OpenAI-compatible Chat Completions client.

    Keep this backend for users who prefer running vLLM as an HTTP server.
    The default backend below is the in-process vLLM backend requested for Qwen/Qwen3.5-35B-A3B.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.base_url = str(cfg["base_url"]).rstrip("/")
        self.model = str(cfg.get("model", "Qwen/Qwen3.5-35B-A3B"))
        self.api_key = str(cfg.get("api_key", "EMPTY"))
        self.temperature = float(cfg.get("temperature", 0.0))
        self.top_p = float(cfg.get("top_p", 1.0))
        self.max_tokens = int(cfg.get("max_tokens", 4096))
        self.timeout = int(cfg.get("timeout_seconds", 180))
        self.retries = int(cfg.get("max_retries", 2))
        self.use_json_mode = bool(cfg.get("use_json_mode", True))

    def generate_prompts(self, plans: list[dict[str, Any]]) -> dict[str, str]:
        if not plans:
            return {}
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_qwen_user_prompt(plans)},
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        if self.use_json_mode:
            request["response_format"] = {"type": "json_object"}

        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = requests.post(url, headers=headers, json=request, timeout=self.timeout)
                response.raise_for_status()
                payload = response.json()
                content = payload["choices"][0]["message"]["content"]
                return parse_qwen_items(content)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < self.retries:
                    sleep_s = min(2**attempt, 8)
                    LOGGER.warning("Qwen HTTP request failed (%s). Retrying in %ss...", exc, sleep_s)
                    time.sleep(sleep_s)
        raise RuntimeError(f"Qwen HTTP request failed after {self.retries + 1} attempt(s): {last_error}")


class QwenVLLMClient:
    """In-process vLLM client for Qwen/Qwen3.5-35B-A3B.

    This follows the user's requested pattern:
      1) load AutoTokenizer;
      2) apply Qwen chat template with enable_thinking=False when supported;
      3) load vLLM LLM once;
      4) run batched generation with SamplingParams.

    It does not require starting a separate `vllm serve` HTTP server.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        try:
            from transformers import AutoTokenizer
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise ImportError(
                "The qwen.backend='vllm' mode requires `transformers` and `vllm`. "
                "Install them with: pip install transformers vllm"
            ) from exc

        self.AutoTokenizer = AutoTokenizer
        self.LLM = LLM
        self.SamplingParams = SamplingParams

        self.model = str(cfg.get("model", "Qwen/Qwen3.5-35B-A3B"))
        self.tensor_parallel_size = int(cfg.get("tensor_parallel_size", 1))
        self.dtype = str(cfg.get("dtype", "auto"))
        self.max_model_len = int(cfg.get("max_model_len", 8192))
        self.gpu_memory_utilization = float(cfg.get("gpu_memory_utilization", 0.90))
        self.temperature = float(cfg.get("temperature", 0.0))
        self.top_p = float(cfg.get("top_p", 1.0))
        self.max_tokens = int(cfg.get("max_tokens", 4096))
        self.trust_remote_code = bool(cfg.get("trust_remote_code", False))
        self.enable_thinking = bool(cfg.get("enable_thinking", False))

        LOGGER.info(
            "Loading Qwen with vLLM: model=%s, tensor_parallel_size=%s, dtype=%s, max_model_len=%s",
            self.model,
            self.tensor_parallel_size,
            self.dtype,
            self.max_model_len,
        )
        self.tokenizer = self.AutoTokenizer.from_pretrained(
            self.model,
            trust_remote_code=self.trust_remote_code,
        )
        self.llm = self.LLM(
            model=self.model,
            tensor_parallel_size=self.tensor_parallel_size,
            dtype=self.dtype,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=self.trust_remote_code,
        )
        self.sampling_params = self.SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )

    def _apply_chat_template(self, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            # Older tokenizers may not expose enable_thinking.
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def generate_prompts_for_batches(self, plan_batches: list[list[dict[str, Any]]]) -> dict[str, str]:
        plan_batches = [b for b in plan_batches if b]
        if not plan_batches:
            return {}
        user_prompts = [build_qwen_user_prompt(batch) for batch in plan_batches]
        full_prompts = [self._apply_chat_template(prompt) for prompt in user_prompts]
        outputs = self.llm.generate(full_prompts, self.sampling_params)
        result: dict[str, str] = {}
        # vLLM preserves input order in outputs, but we only need the parsed sample_id keys.
        for output, batch in zip(outputs, plan_batches, strict=False):
            try:
                text = output.outputs[0].text.strip()
                result.update(parse_qwen_items(text))
            except Exception as exc:  # noqa: BLE001
                ids = [p["sample_id"] for p in batch]
                LOGGER.exception("Could not parse Qwen vLLM output for batch %s. Falling back to templates: %s", ids, exc)
        return result

    def generate_prompts(self, plans: list[dict[str, Any]]) -> dict[str, str]:
        return self.generate_prompts_for_batches([plans])


class QwenClient:
    """Backend switcher. Default is local in-process vLLM."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.backend = str(cfg.get("backend", "vllm")).lower()
        if self.backend == "vllm":
            self.client = QwenVLLMClient(cfg)
        elif self.backend in {"http", "openai", "openai_compatible"}:
            self.client = QwenHTTPClient(cfg)
        else:
            raise ValueError("qwen.backend must be 'vllm' or 'http'.")

    def generate_prompts(self, plans: list[dict[str, Any]]) -> dict[str, str]:
        return self.client.generate_prompts(plans)

    def generate_prompts_for_batches(self, plan_batches: list[list[dict[str, Any]]]) -> dict[str, str]:
        if hasattr(self.client, "generate_prompts_for_batches"):
            return self.client.generate_prompts_for_batches(plan_batches)
        result: dict[str, str] = {}
        for batch in plan_batches:
            result.update(self.client.generate_prompts(batch))
        return result

def parse_json_loose(text: str) -> Any:
    """Parse model output while handling Markdown fences and leading reasoning text."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Find the first JSON object/array that can be parsed.
        candidates = [m.start() for m in re.finditer(r"[\[{]", text)]
        for start in candidates:
            fragment = text[start:]
            try:
                return json.loads(fragment)
            except json.JSONDecodeError:
                continue
    raise ValueError("Could not parse JSON from model response.")


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def jaccard_similarity(a: str, b: str) -> float:
    a_tokens, b_tokens = token_set(a), token_set(b)
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON must be an object: {path}")
    return data


def resolve_path(value: str | Path, base_dir: Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (base_dir / p).resolve()


def largest_remainder_counts(total: int, weights: dict[str, float]) -> dict[str, int]:
    positive = {k: float(v) for k, v in weights.items() if float(v) > 0}
    if not positive:
        raise ValueError("At least one positive weight is required.")
    denominator = sum(positive.values())
    raw = {k: total * v / denominator for k, v in positive.items()}
    floor_counts = {k: int(math.floor(v)) for k, v in raw.items()}
    remaining = total - sum(floor_counts.values())
    ranking = sorted(positive, key=lambda k: (raw[k] - floor_counts[k], k), reverse=True)
    for key in ranking[:remaining]:
        floor_counts[key] += 1
    return floor_counts


def shuffled_schedule(total: int, weights: dict[str, float], rng: random.Random) -> list[str]:
    counts = largest_remainder_counts(total, weights)
    values = [name for name, n in counts.items() for _ in range(n)]
    rng.shuffle(values)
    return values


def build_compatible_category_environment_pairs(
    total: int,
    category_weights: dict[str, float],
    environment_weights: dict[str, float],
    catalog: dict[str, Any],
    rng: random.Random,
) -> list[tuple[str, str]]:
    """Create exact category/environment quotas without semantically impossible pairs.

    The routine is a small constrained scheduler rather than independent random sampling.
    It prevents combinations such as a duck whose profile is restricted to an outdoor pond
    being placed in a library merely because the independent environment schedule chose one.
    """
    cat_remaining = largest_remainder_counts(total, category_weights)
    env_remaining = largest_remainder_counts(total, environment_weights)
    profiles_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for profile in catalog["profiles"]:
        profiles_by_category[profile["category"]].append(profile)

    compatibility: dict[str, set[str]] = {}
    for category, amount in cat_remaining.items():
        profiles = profiles_by_category.get(category, [])
        if not profiles:
            raise ValueError(f"No profiles in source catalog for category '{category}'.")
        compatible = set()
        for profile in profiles:
            compatible.update(profile.get("environment_tags", []))
        compatibility[category] = compatible
        if amount > 0 and not compatible:
            raise ValueError(f"Category '{category}' has no compatible environment tags.")

    unknown_env = set(env_remaining) - {e["category"] for e in catalog["environments"]}
    if unknown_env:
        raise ValueError(f"Config environment categories absent from source catalog: {sorted(unknown_env)}")

    def feasible_after_choice(category: str, environment: str) -> bool:
        trial_cat = dict(cat_remaining)
        trial_env = dict(env_remaining)
        trial_cat[category] -= 1
        trial_env[environment] -= 1
        # Every demanded environment must still have enough compatible category slots left.
        for env_name, demand in trial_env.items():
            if demand <= 0:
                continue
            supply = sum(amount for cat, amount in trial_cat.items() if env_name in compatibility[cat])
            if demand > supply:
                return False
        # Every category must still have enough compatible environment slots left.
        for cat, demand in trial_cat.items():
            if demand <= 0:
                continue
            supply = sum(amount for env_name, amount in trial_env.items() if env_name in compatibility[cat])
            if demand > supply:
                return False
        return True

    pairs: list[tuple[str, str]] = []
    for _ in range(total):
        categories = [c for c, n in cat_remaining.items() if n > 0]
        if not categories:
            break
        # Pick the currently most constrained category first.
        categories.sort(key=lambda c: (sum(env_remaining[e] for e in compatibility[c] if env_remaining[e] > 0), -cat_remaining[c], c))
        chosen_category = categories[0]
        candidates = [e for e in compatibility[chosen_category] if env_remaining.get(e, 0) > 0]
        viable = [e for e in candidates if feasible_after_choice(chosen_category, e)]
        if not viable:
            raise RuntimeError(
                f"No feasible environment remains for category '{chosen_category}'. "
                "Adjust category/environment weights or add compatible source profiles."
            )
        # Prioritise an environment that has high unsatisfied demand but few remaining
        # compatible category slots, then randomise exact ties for diversity.
        def env_priority(env_name: str) -> tuple[float, float]:
            compatible_supply = sum(n for c, n in cat_remaining.items() if env_name in compatibility[c])
            scarcity = env_remaining[env_name] / max(compatible_supply, 1)
            return (scarcity, rng.random())

        chosen_environment = max(viable, key=env_priority)
        pairs.append((chosen_category, chosen_environment))
        cat_remaining[chosen_category] -= 1
        env_remaining[chosen_environment] -= 1

    if any(cat_remaining.values()) or any(env_remaining.values()):
        raise RuntimeError("Could not satisfy the requested category/environment quotas.")
    rng.shuffle(pairs)
    return pairs


def zone_x(x_norm: float) -> str:
    if x_norm < 1 / 3:
        return "left"
    if x_norm > 2 / 3:
        return "right"
    return "center"


def zone_y(y_norm: float) -> str:
    return "up" if y_norm < 0.5 else "down"


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def smoothstep(t: np.ndarray | float) -> np.ndarray | float:
    return 3 * np.asarray(t) ** 2 - 2 * np.asarray(t) ** 3


def speed_warp(t: np.ndarray, profile: str) -> np.ndarray:
    """Map evenly spaced output frames to progression along a geometric curve."""
    if profile == "constant":
        return t
    if profile == "ease_in_out":
        return smoothstep(t)
    if profile == "accelerate":
        return t**1.8
    if profile == "decelerate":
        return 1.0 - (1.0 - t) ** 1.8
    if profile == "stop_go":
        out = np.empty_like(t)
        first = t <= 0.42
        hold = (t > 0.42) & (t <= 0.58)
        last = t > 0.58
        out[first] = 0.46 * smoothstep(t[first] / 0.42)
        out[hold] = 0.46
        out[last] = 0.46 + 0.54 * smoothstep((t[last] - 0.58) / 0.42)
        return out
    raise ValueError(f"Unknown speed profile: {profile}")


def catmull_rom(p0: dict[str, float], p1: dict[str, float], p2: dict[str, float], p3: dict[str, float], t: float) -> dict[str, float]:
    """Matches the public ComfyUI-LTXVideo sparse track interpolation."""
    t2 = t * t
    t3 = t2 * t
    return {
        "x": 0.5
        * (
            2 * p1["x"]
            + (-p0["x"] + p2["x"]) * t
            + (2 * p0["x"] - 5 * p1["x"] + 4 * p2["x"] - p3["x"]) * t2
            + (-p0["x"] + 3 * p1["x"] - 3 * p2["x"] + p3["x"]) * t3
        ),
        "y": 0.5
        * (
            2 * p1["y"]
            + (-p0["y"] + p2["y"]) * t
            + (2 * p0["y"] - 5 * p1["y"] + 4 * p2["y"] - p3["y"]) * t2
            + (-p0["y"] + 3 * p1["y"] - 3 * p2["y"] + p3["y"]) * t3
        ),
    }


def interpolate_spline(control_points: list[dict[str, float]], num_samples: int) -> list[dict[str, float]]:
    if num_samples < 2:
        raise ValueError("num_samples must be >= 2")
    if not control_points:
        return []
    if len(control_points) == 1:
        return [dict(control_points[0]) for _ in range(num_samples)]
    if len(control_points) == 2:
        a, b = control_points
        return [
            {
                "x": a["x"] + (b["x"] - a["x"]) * i / (num_samples - 1),
                "y": a["y"] + (b["y"] - a["y"]) * i / (num_samples - 1),
            }
            for i in range(num_samples)
        ]

    points = [control_points[0], *control_points, control_points[-1]]
    n_segments = len(points) - 3
    result: list[dict[str, float]] = []
    for i in range(num_samples):
        global_t = (i / (num_samples - 1)) * n_segments
        segment = min(int(global_t), n_segments - 1)
        local_t = global_t - segment
        result.append(catmull_rom(points[segment], points[segment + 1], points[segment + 2], points[segment + 3], local_t))
    return result


def sample_curve_by_progress(curve: list[dict[str, float]], progress: np.ndarray) -> list[dict[str, float]]:
    max_idx = len(curve) - 1
    out: list[dict[str, float]] = []
    for p in progress:
        position = float(clamp(float(p), 0.0, 1.0)) * max_idx
        low = int(math.floor(position))
        high = min(low + 1, max_idx)
        alpha = position - low
        x = curve[low]["x"] * (1 - alpha) + curve[high]["x"] * alpha
        y = curve[low]["y"] * (1 - alpha) + curve[high]["y"] * alpha
        out.append({"x": x, "y": y})
    return out


def make_control_points(primitive: str, direction: str, rng: random.Random) -> list[dict[str, float]]:
    """Create 3-4 normalized control points while keeping a margin from image borders."""
    x_low, x_high = rng.uniform(0.12, 0.24), rng.uniform(0.76, 0.88)
    y_low, y_high = rng.uniform(0.16, 0.30), rng.uniform(0.70, 0.84)
    base_y = rng.uniform(0.36, 0.66)
    base_x = rng.uniform(0.38, 0.62)
    jitter = lambda scale: rng.uniform(-scale, scale)

    if primitive in {"horizontal_linear", "horizontal_arc", "horizontal_s_curve"}:
        if direction == "right":
            start, end = (x_low, base_y + jitter(0.035)), (x_high, base_y + jitter(0.035))
        else:
            start, end = (x_high, base_y + jitter(0.035)), (x_low, base_y + jitter(0.035))
        x1 = start[0] * (2 / 3) + end[0] * (1 / 3)
        x2 = start[0] * (1 / 3) + end[0] * (2 / 3)
        if primitive == "horizontal_linear":
            y1 = start[1] * (2 / 3) + end[1] * (1 / 3)
            y2 = start[1] * (1 / 3) + end[1] * (2 / 3)
        elif primitive == "horizontal_arc":
            arc = rng.choice([-1.0, 1.0]) * rng.uniform(0.06, 0.13)
            y1 = base_y + arc
            y2 = base_y + arc * 0.85
        else:  # horizontal_s_curve
            bend = rng.choice([-1.0, 1.0]) * rng.uniform(0.055, 0.11)
            y1 = base_y + bend
            y2 = base_y - bend
        return [
            {"x": start[0], "y": clamp(start[1], 0.10, 0.90)},
            {"x": x1, "y": clamp(y1, 0.10, 0.90)},
            {"x": x2, "y": clamp(y2, 0.10, 0.90)},
            {"x": end[0], "y": clamp(end[1], 0.10, 0.90)},
        ]

    if primitive in {"diagonal_linear", "diagonal_arc"}:
        if direction == "down_right":
            start, end = (x_low, y_low), (x_high, y_high)
        elif direction == "up_right":
            start, end = (x_low, y_high), (x_high, y_low)
        elif direction == "down_left":
            start, end = (x_high, y_low), (x_low, y_high)
        else:
            start, end = (x_high, y_high), (x_low, y_low)
        x1 = start[0] * (2 / 3) + end[0] * (1 / 3)
        x2 = start[0] * (1 / 3) + end[0] * (2 / 3)
        y1 = start[1] * (2 / 3) + end[1] * (1 / 3)
        y2 = start[1] * (1 / 3) + end[1] * (2 / 3)
        if primitive == "diagonal_arc":
            lateral = rng.choice([-1.0, 1.0]) * rng.uniform(0.05, 0.10)
            y1 += lateral
            y2 -= lateral
        return [
            {"x": start[0], "y": start[1]},
            {"x": clamp(x1, 0.10, 0.90), "y": clamp(y1, 0.10, 0.90)},
            {"x": clamp(x2, 0.10, 0.90), "y": clamp(y2, 0.10, 0.90)},
            {"x": end[0], "y": end[1]},
        ]

    if primitive == "vertical_short":
        span = rng.uniform(0.28, 0.42)
        if direction == "down":
            start, end = (base_x + jitter(0.03), clamp(base_y - span / 2, 0.13, 0.55)), (base_x + jitter(0.03), clamp(base_y + span / 2, 0.45, 0.87))
        else:
            start, end = (base_x + jitter(0.03), clamp(base_y + span / 2, 0.45, 0.87)), (base_x + jitter(0.03), clamp(base_y - span / 2, 0.13, 0.55))
        return [
            {"x": clamp(start[0], 0.12, 0.88), "y": start[1]},
            {"x": clamp(base_x + jitter(0.045), 0.12, 0.88), "y": start[1] * (2 / 3) + end[1] * (1 / 3)},
            {"x": clamp(base_x + jitter(0.045), 0.12, 0.88), "y": start[1] * (1 / 3) + end[1] * (2 / 3)},
            {"x": clamp(end[0], 0.12, 0.88), "y": end[1]},
        ]

    if primitive == "local_sway":
        amplitude_x = rng.uniform(0.08, 0.16)
        amplitude_y = rng.uniform(0.025, 0.075)
        x0, y0 = base_x - amplitude_x / 2, base_y
        return [
            {"x": clamp(x0, 0.20, 0.80), "y": clamp(y0, 0.18, 0.82)},
            {"x": clamp(base_x + amplitude_x / 2, 0.20, 0.80), "y": clamp(base_y - amplitude_y, 0.18, 0.82)},
            {"x": clamp(base_x - amplitude_x / 3, 0.20, 0.80), "y": clamp(base_y + amplitude_y, 0.18, 0.82)},
            {"x": clamp(base_x + amplitude_x / 3, 0.20, 0.80), "y": clamp(base_y, 0.18, 0.82)},
        ]

    raise ValueError(f"Unsupported primitive: {primitive}")


def infer_direction(primitive: str, rng: random.Random) -> str:
    if primitive.startswith("horizontal"):
        return rng.choice(["right", "left"])
    if primitive.startswith("diagonal"):
        return rng.choice(["down_right", "up_right", "down_left", "up_left"])
    if primitive == "vertical_short":
        return rng.choice(["down", "up"])
    if primitive == "local_sway":
        return "local"
    raise ValueError(f"No direction rule for primitive {primitive}")


def infer_motion_label(start: dict[str, float], end: dict[str, float]) -> str:
    dx = end["x"] - start["x"]
    dy = end["y"] - start["y"]
    if abs(dx) >= abs(dy) and abs(dx) >= 0.06:
        return "left_to_right" if dx > 0 else "right_to_left"
    if abs(dy) >= 0.06:
        return "up_to_down" if dy > 0 else "down_to_up"
    return "local_motion"


def create_trajectory(
    primitive: str,
    speed_profile: str,
    width: int,
    height: int,
    frames: int,
    rng: random.Random,
) -> TrajectoryResult:
    direction = infer_direction(primitive, rng)
    control_points_norm = make_control_points(primitive, direction, rng)
    dense_base_norm = interpolate_spline(control_points_norm, 2001)
    time_grid = np.linspace(0.0, 1.0, frames, dtype=np.float64)
    progress = speed_warp(time_grid, speed_profile)
    dense_norm = sample_curve_by_progress(dense_base_norm, progress)

    dense_points = [
        {
            "x": int(round(clamp(p["x"], 0.0, 1.0) * (width - 1))),
            "y": int(round(clamp(p["y"], 0.0, 1.0) * (height - 1))),
        }
        for p in dense_norm
    ]
    start = dense_norm[0]
    end = dense_norm[-1]
    dx, dy = end["x"] - start["x"], end["y"] - start["y"]
    displacement = math.hypot(dx, dy)
    path_length = sum(
        math.hypot(dense_norm[i]["x"] - dense_norm[i - 1]["x"], dense_norm[i]["y"] - dense_norm[i - 1]["y"])
        for i in range(1, len(dense_norm))
    )
    if abs(dx) >= abs(dy):
        start_zone, end_zone = zone_x(start["x"]), zone_x(end["x"])
    else:
        start_zone, end_zone = zone_y(start["y"]), zone_y(end["y"])

    return TrajectoryResult(
        primitive=primitive,
        speed_profile=speed_profile,
        direction=direction,
        control_points=[{"x": round(p["x"], 4), "y": round(p["y"], 4)} for p in control_points_norm],
        dense_points=dense_points,
        start_zone=start_zone,
        end_zone=end_zone,
        motion_label=infer_motion_label(start, end),
        motion_magnitude="large" if displacement >= 0.38 else "slight",
        displacement_norm=round(displacement, 5),
        path_length_norm=round(path_length, 5),
    )


def age_color_rgb(ratio: float) -> tuple[int, int, int]:
    """Match sparse_tracks.py: old blue -> green -> yellow -> new red, in RGB order."""
    ratio = float(np.clip(ratio, 0.0, 1.0))
    if ratio <= 1 / 3:
        tr = ratio * 3
        r, g, b = 0.0, tr, 1.0 - tr
    elif ratio <= 2 / 3:
        tr = (ratio - 1 / 3) * 3
        r, g, b = tr, 1.0, 0.0
    else:
        tr = (ratio - 2 / 3) * 3
        r, g, b = 1.0, 1.0 - tr, 0.0
    return (
        int(round(r * 255)),
        int(round(g * 255)),
        int(round(b * 255)),
    )


def render_resolution(
    width: int,
    height: int,
    reference_short_side: int = _REF_SHORT_SIDE,
) -> tuple[int, int, float, float]:
    """Match sparse_tracks.py: draw on a high-res canvas whose short side is 1080."""
    if height <= width:
        render_width = int(width * reference_short_side / height)
        render_height = reference_short_side
    else:
        render_width = reference_short_side
        render_height = int(height * reference_short_side / width)

    scale_x = render_width / width
    scale_y = render_height / height
    return render_width, render_height, scale_x, scale_y


def generate_track_frames(
    tracks: list[list[dict[str, int]]],
    width: int,
    height: int,
) -> Iterable[np.ndarray]:
    """Render tracker frames using the same channel convention as sparse_tracks.py.

    Important detail: values are first drawn as RGB on a high-resolution canvas,
    resized, then the numeric channels are swapped RGB -> BGR. The resulting array
    is still written to FFmpeg as raw `rgb24` bytes, matching the Motion Track
    IC-LoRA reference-video training format used by sparse_tracks.py.
    """
    if not tracks:
        raise ValueError("tracks must contain at least one track")

    render_width, render_height, scale_x, scale_y = render_resolution(width, height)
    num_frames = max(len(track) for track in tracks)

    scaled_tracks: list[list[dict[str, float]]] = []
    for track in tracks:
        if not track:
            raise ValueError("each track must contain at least one point")
        scaled_tracks.append(
            [
                {
                    "x": point["x"] * scale_x,
                    "y": point["y"] * scale_y,
                }
                for point in track
            ]
        )

    for frame_idx in range(num_frames):
        # OpenCV only writes numeric arrays here; we intentionally treat this as RGB.
        highres_rgb = np.zeros((render_height, render_width, 3), dtype=np.uint8)
        trail_start = max(0, frame_idx - _MAX_TRAIL)

        # Old trail first, new trail last; newer points cover older points.
        for track in scaled_tracks:
            end_idx = min(frame_idx, len(track) - 1)
            for point_idx in range(trail_start, end_idx + 1):
                point = track[point_idx]
                age = frame_idx - point_idx
                ratio = float(np.clip(1.0 - age / _MAX_TRAIL, 0.0, 1.0))
                radius = _MIN_RADIUS + (_MAX_RADIUS - _MIN_RADIUS) * ratio
                radius = max(1, int(round(radius)))
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

        # sparse_tracks.py compatibility: numeric RGB -> BGR channel swap.
        frame_for_ic_lora = frame_rgb[..., [2, 1, 0]].copy()
        yield frame_for_ic_lora


def save_h264_video(
    frames: Iterable[np.ndarray],
    output_path: str | Path,
    width: int,
    height: int,
    fps: float,
) -> None:
    """Save raw generated frames to H.264 through FFmpeg stdin.

    This avoids OpenCV VideoWriter color conversion ambiguity and follows the
    reference script's rawvideo -> H.264 path.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg was not found. Please install FFmpeg before rendering tracker.mp4.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
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
        "-profile:v",
        "high",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    assert process.stderr is not None

    try:
        for frame in frames:
            if frame.shape != (height, width, 3):
                raise ValueError(f"Frame shape mismatch: expected {(height, width, 3)}, got {frame.shape}")
            if frame.dtype != np.uint8:
                raise ValueError(f"Frame dtype mismatch: expected uint8, got {frame.dtype}")
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        return_code = process.wait()
    except BrokenPipeError as exc:
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"FFmpeg stopped early:\n{stderr}") from exc

    if return_code != 0:
        raise RuntimeError(f"FFmpeg H.264 encoding failed, return_code={return_code}\n{stderr}")


def write_tracker_video(tracks: list[list[dict[str, int]]], width: int, height: int, fps: float, output_path: Path) -> None:
    """Write an official-format Motion Track IC-LoRA tracker.mp4."""
    save_h264_video(
        frames=generate_track_frames(tracks=tracks, width=width, height=height),
        output_path=output_path,
        width=width,
        height=height,
        fps=fps,
    )


def deterministic_prompt(plan: dict[str, Any]) -> str:
    sounds = plan["causal_sounds"]
    if len(sounds) == 1:
        sound_text = sounds[0]
        sound_verb = "is"
    elif len(sounds) == 2:
        sound_text = f"{sounds[0]} and {sounds[1]}"
        sound_verb = "are"
    else:
        sound_text = ", ".join(sounds[:-1]) + f", and {sounds[-1]}"
        sound_verb = "are"
    return normalize_space(
        f"A locked-off medium-wide shot in {plan['environment']} under {plan['lighting']}. "
        f"{capitalize_first(plan['primary_emitter'])} {plan['action']} smoothly along a natural route. "
        f"{capitalize_first(sound_text)} {sound_verb} the dominant audible event, while the ambient background remains soft. "
        f"The scene has a {plan['visual_style']} look in {plan['acoustic_environment']}. "
        "The camera remains completely static in one continuous shot."
    )


def capitalize_first(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def validate_or_repair_prompt(prompt: str | None, plan: dict[str, Any]) -> tuple[str, str]:
    """Return (prompt, source). Invalid Qwen output falls back to a controlled template."""
    fallback = deterministic_prompt(plan)
    if not prompt or not isinstance(prompt, str):
        return fallback, "template_missing_qwen_output"
    candidate = normalize_space(prompt)
    if len(candidate.split()) < 25 or len(candidate.split()) > 180:
        return fallback, "template_bad_length"
    lower = candidate.lower()
    if any(re.search(pattern, lower) for pattern in FORBIDDEN_PROMPT_PATTERNS):
        return fallback, "template_forbidden_content"
    if not any(hint in lower for hint in STATIC_CAMERA_HINTS):
        candidate += " The camera remains locked off and completely static."
    # Still make the static-camera condition explicit after model output.
    if not candidate.endswith((".", "!", "?")):
        candidate += "."
    return candidate, "qwen"


def build_plan(
    sample_id: str,
    profile: dict[str, Any],
    environment: dict[str, Any],
    trajectory: TrajectoryResult,
    width: int,
    height: int,
    fps: float,
    frames: int,
    seed: int,
    rng: random.Random,
) -> dict[str, Any]:
    lighting = rng.choice(environment.get("lighting_options", ["soft natural light"]))
    visual_style = rng.choice(environment.get("visual_styles", ["realistic documentary style"]))
    return {
        "sample_id": sample_id,
        "seed": seed,
        "width": width,
        "height": height,
        "fps": fps,
        "num_frames": frames,
        "duration_seconds": round(frames / fps, 4),
        "source_profile_id": profile["id"],
        "source_category": profile["category"],
        "primary_emitter": profile["primary_emitter"],
        "action": profile["action"],
        "causal_sounds": profile["causal_sounds"],
        "sound_temporal_pattern": profile.get("sound_temporal_pattern", "continuous"),
        "environment_id": environment["id"],
        "environment": environment["environment"],
        "acoustic_environment": environment["acoustic_environment"],
        "ambient_sound": environment["ambient_sound"],
        "lighting": lighting,
        "visual_style": visual_style,
        "camera_policy": "locked_off_static",
        "motion_primitive": trajectory.primitive,
        "speed_profile": trajectory.speed_profile,
        "track_direction_metadata": trajectory.direction,
        "sparse_control_points_normalized": trajectory.control_points,
        "motion_label": trajectory.motion_label,
        "start_zone": trajectory.start_zone,
        "end_zone": trajectory.end_zone,
        "motion_magnitude": trajectory.motion_magnitude,
        "displacement_norm": trajectory.displacement_norm,
        "path_length_norm": trajectory.path_length_norm,
        "expected_validation": {
            "single_visible_dominant_emitter": True,
            "sound_is_causally_linked_to_motion": True,
            "static_camera": True,
            "track_following_required": True,
        },
    }


def sample_profile_and_environment(
    category: str,
    environment_category: str,
    catalog: dict[str, Any],
    rng: random.Random,
) -> tuple[dict[str, Any], dict[str, Any]]:
    profiles = [
        p
        for p in catalog["profiles"]
        if p["category"] == category and environment_category in p.get("environment_tags", [])
    ]
    if not profiles:
        raise ValueError(
            f"No source profile for category '{category}' is compatible with environment category "
            f"'{environment_category}'. The pair scheduler should have prevented this."
        )
    environments = [e for e in catalog["environments"] if e["category"] == environment_category]
    if not environments:
        raise ValueError(f"No environments in catalog for category '{environment_category}'.")
    return rng.choice(profiles), rng.choice(environments)


def choose_trajectory_for_profile(profile: dict[str, Any], cfg: dict[str, Any], rng: random.Random) -> tuple[str, str]:
    global_primitives = set(cfg["trajectory"]["allowed_primitives"])
    allowed = [p for p in profile.get("motion_primitives", []) if p in global_primitives]
    if not allowed:
        raise ValueError(f"Profile {profile['id']} has no primitive enabled in config.")
    primitive = rng.choice(allowed)
    speed_profiles = list(cfg["trajectory"]["speed_profiles"])
    if primitive == "local_sway":
        speed_profiles = [p for p in speed_profiles if p in {"constant", "ease_in_out"}] or ["ease_in_out"]
    return primitive, rng.choice(speed_profiles)


def make_signature(plan: dict[str, Any]) -> str:
    values = [
        plan["source_profile_id"],
        plan["environment_id"],
        plan["motion_primitive"],
        plan["speed_profile"],
        plan["motion_label"],
        plan["lighting"],
        plan["visual_style"],
    ]
    return "|".join(values)


def sample_plans(config: dict[str, Any], catalog: dict[str, Any], count: int, start_index: int) -> list[dict[str, Any]]:
    generation = config["generation"]
    trajectory_cfg = config["trajectory"]
    seed = int(generation["seed"])
    rng = random.Random(seed + start_index * 1009)
    width, height = int(generation["width"]), int(generation["height"])
    fps, frames = float(generation["fps"]), int(generation["num_frames"])

    category_environment_pairs = build_compatible_category_environment_pairs(
        count,
        generation["category_weights"],
        generation["environment_weights"],
        catalog,
        rng,
    )

    plans: list[dict[str, Any]] = []
    seen_signatures: set[str] = set()
    max_attempts = int(generation.get("max_uniqueness_attempts", 80))

    for local_idx in range(count):
        sample_number = start_index + local_idx
        sample_id = f"synthetic_{sample_number:06d}"
        category, env_category = category_environment_pairs[local_idx]
        successful = False
        for attempt in range(max_attempts):
            # Use a deterministic per-sample stream so reruns remain reproducible.
            local_seed = seed + sample_number * 100003 + attempt
            local_rng = random.Random(local_seed)
            profile, environment = sample_profile_and_environment(category, env_category, catalog, local_rng)
            primitive, speed_profile = choose_trajectory_for_profile(profile, config, local_rng)
            trajectory = create_trajectory(primitive, speed_profile, width, height, frames, local_rng)
            plan = build_plan(sample_id, profile, environment, trajectory, width, height, fps, frames, local_seed, local_rng)
            signature = make_signature(plan)
            if signature not in seen_signatures:
                plan["diversity_signature"] = signature
                plan["motion_track"] = [trajectory.dense_points]
                seen_signatures.add(signature)
                plans.append(plan)
                successful = True
                break
        if not successful:
            raise RuntimeError(
                f"Unable to create a unique plan for {sample_id}. Increase source/environment variety "
                "or lower generation.max_uniqueness_attempts."
            )
    return plans


def prompt_all_plans(plans: list[dict[str, Any]], config: dict[str, Any], qwen_mode: str) -> None:
    qwen_cfg = dict(config.get("qwen", {}))
    enabled = bool(qwen_cfg.get("enabled", False))
    use_qwen = (qwen_mode == "on") or (qwen_mode == "auto" and enabled)
    batch_size = int(qwen_cfg.get("batch_size", 12))
    seen_prompts: list[str] = []

    outputs: dict[str, str] = {}
    if use_qwen:
        client = QwenClient(qwen_cfg)
        plan_batches = [plans[start : start + batch_size] for start in range(0, len(plans), batch_size)]
        try:
            outputs.update(client.generate_prompts_for_batches(plan_batches))
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Qwen generation failed. Falling back to templates: %s", exc)

    prompt_similarity_threshold = float(config["generation"].get("max_prompt_jaccard_similarity", 0.86))
    for plan in plans:
        prompt, source = validate_or_repair_prompt(outputs.get(plan["sample_id"]), plan)
        # A lexical guard prevents accidentally repeated Qwen prose from entering a batch.
        if source == "qwen" and any(jaccard_similarity(prompt, old) >= prompt_similarity_threshold for old in seen_prompts):
            prompt, source = deterministic_prompt(plan), "template_prompt_similarity_guard"
        seen_prompts.append(prompt)
        plan["ltx_prompt"] = prompt
        plan["prompt_source"] = source


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def append_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def next_sample_index(root: Path) -> int:
    indices: list[int] = []
    for path in root.glob("synthetic_*"):
        if path.is_dir():
            try:
                indices.append(int(path.name.split("_")[-1]))
            except ValueError:
                continue
    return max(indices) + 1 if indices else 1


def write_plans(plans: list[dict[str, Any]], config: dict[str, Any], output_root: Path, render_tracker: bool, overwrite: bool) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for plan in tqdm(plans, desc="Writing samples", unit="sample"):
        sample_dir = output_root / plan["sample_id"]
        if sample_dir.exists() and not overwrite:
            raise FileExistsError(f"Sample directory exists: {sample_dir}. Use --overwrite or choose another output root.")
        sample_dir.mkdir(parents=True, exist_ok=True)

        track_json = sample_dir / "motion_track.json"
        prompt_txt = sample_dir / "prompt.txt"
        metadata_json = sample_dir / "sample_spec.json"
        tracker_mp4 = sample_dir / "tracker.mp4"
        dump_json(track_json, plan["motion_track"])
        prompt_txt.write_text(plan["ltx_prompt"] + "\n", encoding="utf-8")
        if render_tracker:
            write_tracker_video(plan["motion_track"], plan["width"], plan["height"], plan["fps"], tracker_mp4)

        metadata = dict(plan)
        metadata.pop("motion_track", None)
        metadata["paths"] = {
            "motion_track_json": str(track_json.relative_to(output_root)),
            "tracker_video": str(tracker_mp4.relative_to(output_root)) if render_tracker else None,
            "prompt_txt": str(prompt_txt.relative_to(output_root)),
        }
        dump_json(metadata_json, metadata)

        manifest_record = {
            "sample_id": plan["sample_id"],
            "prompt": plan["ltx_prompt"],
            "motion_track_json": str(track_json.relative_to(output_root)),
            "tracker_video": str(tracker_mp4.relative_to(output_root)) if render_tracker else None,
            "sample_spec": str(metadata_json.relative_to(output_root)),
            "source_category": plan["source_category"],
            "primary_emitter": plan["primary_emitter"],
            "causal_sounds": plan["causal_sounds"],
            "environment": plan["environment"],
            "motion_primitive": plan["motion_primitive"],
            "speed_profile": plan["speed_profile"],
            "motion_label": plan["motion_label"],
            "start_zone": plan["start_zone"],
            "end_zone": plan["end_zone"],
            "prompt_source": plan["prompt_source"],
            "seed": plan["seed"],
        }
        records.append(manifest_record)
    return records


def create_summary(records: list[dict[str, Any]], output_root: Path, config: dict[str, Any]) -> None:
    summary = {
        "num_samples": len(records),
        "source_category_counts": dict(Counter(r["source_category"] for r in records)),
        "environment_counts": dict(Counter(r["environment"] for r in records)),
        "motion_primitive_counts": dict(Counter(r["motion_primitive"] for r in records)),
        "motion_label_counts": dict(Counter(r["motion_label"] for r in records)),
        "prompt_source_counts": dict(Counter(r["prompt_source"] for r in records)),
        "generation_config": config["generation"],
        "trajectory_config": config["trajectory"],
    }
    dump_json(output_root / "summary.json", summary)


def validate_config(config: dict[str, Any]) -> None:
    for top_level in ("generation", "trajectory", "qwen", "catalog"):
        if top_level not in config:
            raise ValueError(f"Missing top-level config field: '{top_level}'")
    g = config["generation"]
    for key in ("width", "height", "fps", "num_frames", "seed", "category_weights", "environment_weights"):
        if key not in g:
            raise ValueError(f"Missing generation.{key}")
    width, height = int(g["width"]), int(g["height"])
    if width < 8 or height < 8:
        raise ValueError("width and height must be >= 8")
    if width % 8 != 0 or height % 8 != 0:
        LOGGER.warning("Width/height are not multiples of 8. LTX normally requires model-friendly dimensions; 64 is recommended.")
    if int(g["num_frames"]) < 2 or float(g["fps"]) <= 0:
        raise ValueError("num_frames must be >= 2 and fps must be > 0")


def make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML configuration.")
    parser.add_argument("--count", type=int, default=None, help="Override generation.count from config.")
    parser.add_argument("--qwen", choices=("auto", "on", "off"), default="auto", help="Use Qwen according to config, force it, or disable it.")
    parser.add_argument("--no-render-tracker", action="store_true", help="Write JSON + metadata only; do not render tracker.mp4.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing sample directory.")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser


def main() -> int:
    args = make_arg_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    config_path = args.config.resolve()
    config = read_yaml(config_path)
    validate_config(config)
    if args.count is not None:
        if args.count <= 0:
            raise ValueError("--count must be positive")
        config["generation"]["count"] = int(args.count)
    count = int(config["generation"]["count"])

    base_dir = config_path.parent
    catalog_value = Path(config["catalog"]["path"])
    catalog_path = resolve_path(catalog_value, base_dir)
    # A copied config.yaml often lives outside this project directory. In that case,
    # keep the documented short catalog filename working by falling back to the script directory.
    if not catalog_path.exists() and not catalog_value.is_absolute():
        catalog_path = (Path(__file__).resolve().parent / catalog_value).resolve()
    catalog = read_json(catalog_path)
    if "profiles" not in catalog or "environments" not in catalog:
        raise ValueError("Catalog needs both 'profiles' and 'environments' arrays.")

    output_root = resolve_path(config["generation"]["output_root"], base_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    start_index = next_sample_index(output_root)
    plans = sample_plans(config, catalog, count, start_index)
    prompt_all_plans(plans, config, args.qwen)
    records = write_plans(
        plans,
        config,
        output_root,
        render_tracker=not args.no_render_tracker,
        overwrite=args.overwrite,
    )
    append_jsonl(output_root / "manifest.jsonl", records)
    create_summary(records, output_root, config)

    LOGGER.info("Created %d samples in %s", len(records), output_root)
    LOGGER.info("Manifest: %s", output_root / "manifest.jsonl")
    LOGGER.info("Summary:  %s", output_root / "summary.json")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        LOGGER.error("Interrupted by user.")
        raise SystemExit(130)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Generation failed: %s", exc)
        raise SystemExit(1)
