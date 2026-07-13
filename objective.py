"""Shared objective-function, geometry, formatting, and reproducibility helpers."""
from __future__ import annotations

import hashlib
import math
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

TORCH_AVAILABLE = True

def truncate_float(value, decimals: int = 2) -> float:
    if value is None:
        return value
    try:
        value = float(value)
    except (TypeError, ValueError):
        return value
    factor = 10 ** decimals
    if value >= 0:
        return math.floor(value * factor) / factor
    return math.ceil(value * factor) / factor

def fmt_trunc(value, decimals: int = 2, signed: bool = False) -> str:
    if value is None:
        return ""
    try:
        v = truncate_float(value, decimals)
    except Exception:
        return ""
    if signed:
        if abs(v) < 1e-12:
            return f"{0:.{decimals}f}"
        return f"{v:+.{decimals}f}"
    return f"{v:.{decimals}f}"

def fmt_trunc2(value) -> str:
    return fmt_trunc(value, decimals=2, signed=False)

def fmt_signed_trunc2(value) -> str:
    return fmt_trunc(value, decimals=2, signed=True)

def get_strawberry_position(
    strawberry_idx: int,
    detection_boxes,
    index_to_mask: Optional[Dict[int, np.ndarray]] = None,
) -> Tuple[float, float]:
    if not (1 <= strawberry_idx <= len(detection_boxes)):
        return 0.0, 0.0
    mask = index_to_mask.get(strawberry_idx) if isinstance(index_to_mask, dict) else None
    if mask is not None and np.any(mask > 0):
        ys, xs = np.where(mask > 0)
        return float(xs.mean()), float(ys.mean())
    x1, y1, x2, y2 = [int(v) for v in detection_boxes[strawberry_idx - 1][:4]]
    return float((x1 + x2) / 2), float((y1 + y2) / 2)

def get_strawberry_position_from_processor(
    processor,
    strawberry_idx: int,
) -> Tuple[float, float]:
    if hasattr(processor, "get_cached_mask_centroid"):
        centroid = processor.get_cached_mask_centroid(strawberry_idx)
        if centroid is not None:
            return float(centroid[0]), float(centroid[1])
    return get_strawberry_position(
        strawberry_idx,
        processor.detection_boxes,
        index_to_mask=None,
    )

def build_strawberry_positions_from_processor(
    processor,
) -> List[Tuple[float, float]]:
    return [
        get_strawberry_position_from_processor(processor, det_idx)
        for det_idx in range(1, len(processor.detection_boxes) + 1)
    ]

def calculate_path_length_pixels(path: List[int], strawberry_positions: List[Tuple[float, float]]) -> float:
    if not path or len(path) < 2 or not strawberry_positions:
        return 0.0
    total_distance = 0.0
    for idx_a, idx_b in zip(path[:-1], path[1:]):
        if not (1 <= idx_a <= len(strawberry_positions) and 1 <= idx_b <= len(strawberry_positions)):
            continue
        x1, y1 = strawberry_positions[idx_a - 1]
        x2, y2 = strawberry_positions[idx_b - 1]
        total_distance += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return float(total_distance)

def seed_everything(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)

    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass

def make_stable_seed_from_image_id(image_id: str, base_seed: int = 42, modulo: int = 1000000) -> int:
    image_id = str(image_id)
    digest = hashlib.blake2b(image_id.encode("utf-8"), digest_size=8).hexdigest()
    stable_offset = int(digest, 16) % int(modulo)
    return int(base_seed) + stable_offset

def make_stable_seed_from_image_round(image_id: str, round_id: int, base_seed: int = 42, modulo: int = 1000000) -> int:
    key = f"{image_id}__round_{int(round_id)}"
    return make_stable_seed_from_image_id(key, base_seed=base_seed, modulo=modulo)

def calculate_weighted_objective(
    avg_difficulty: float,
    normalized_distance: float,
    skip_rate: float,
    config,
) -> float:
    return (
        config.w1_difficulty * avg_difficulty
        + config.w2_distance * normalized_distance
        + config.w3_skip_rate * skip_rate
    )
