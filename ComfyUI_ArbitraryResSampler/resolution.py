from __future__ import annotations

import math
from typing import List, Tuple

from .constants import LATENT_SCALE_FACTOR


def ceil_div(a: int, b: int) -> int:
    return -(-int(a) // int(b))


def round_to_multiple(value: int, multiple: int = LATENT_SCALE_FACTOR) -> int:
    return max(multiple, int(round(int(value) / float(multiple)) * multiple))


def normalize_resolution(width: int, height: int) -> Tuple[int, int]:
    return round_to_multiple(width), round_to_multiple(height)


def latent_size_from_pixels(width: int, height: int) -> Tuple[int, int]:
    return ceil_div(width, LATENT_SCALE_FACTOR), ceil_div(height, LATENT_SCALE_FACTOR)


def pixels_from_latent_size(latent_width: int, latent_height: int) -> Tuple[int, int]:
    return int(latent_width) * LATENT_SCALE_FACTOR, int(latent_height) * LATENT_SCALE_FACTOR


def fit_base_resolution(target_width: int, target_height: int, base_pixels: int) -> Tuple[int, int]:
    """Fit a base SDXL-like resolution to the target aspect ratio.

    `base_pixels` is treated as the approximate square side length, so 1024 means
    an area around 1024 x 1024, but reshaped to the target aspect ratio.
    """
    target_width, target_height = normalize_resolution(target_width, target_height)
    aspect = target_width / max(target_height, 1)
    area = float(base_pixels) * float(base_pixels)

    base_width = int(round(math.sqrt(area * aspect)))
    base_height = int(round(math.sqrt(area / max(aspect, 1e-8))))

    # SDXL tends to behave better on 64-pixel multiples for the first pass.
    base_width = max(64, int(round(base_width / 64.0) * 64))
    base_height = max(64, int(round(base_height / 64.0) * 64))

    return min(base_width, target_width), min(base_height, target_height)


def build_stage_schedule(
    start_width: int,
    start_height: int,
    target_width: int,
    target_height: int,
    max_scale_per_stage: float,
) -> List[Tuple[int, int]]:
    """Build a monotonic pixel-resolution pyramid from start to target.

    Works for arbitrary aspect ratios and arbitrary target sizes. If the source is
    already larger than the target in either dimension, the final target is still
    appended as a resize stage.
    """
    start_width, start_height = normalize_resolution(start_width, start_height)
    target_width, target_height = normalize_resolution(target_width, target_height)

    if max_scale_per_stage <= 1.0:
        max_scale_per_stage = 1.1

    stages: List[Tuple[int, int]] = [(start_width, start_height)]
    cur_w, cur_h = start_width, start_height

    # If downscaling or mixed scaling is needed, do it directly. The hierarchy is
    # designed for upscaling, not repeated downscaling.
    if cur_w >= target_width and cur_h >= target_height:
        if (cur_w, cur_h) != (target_width, target_height):
            stages.append((target_width, target_height))
        return stages

    while cur_w < target_width or cur_h < target_height:
        next_w = min(target_width, int(round(cur_w * max_scale_per_stage)))
        next_h = min(target_height, int(round(cur_h * max_scale_per_stage)))

        next_w = max(cur_w + 8, next_w) if next_w < target_width else target_width
        next_h = max(cur_h + 8, next_h) if next_h < target_height else target_height

        next_w, next_h = normalize_resolution(next_w, next_h)
        next_w = min(next_w, target_width)
        next_h = min(next_h, target_height)

        if (next_w, next_h) == (cur_w, cur_h):
            break

        stages.append((next_w, next_h))
        cur_w, cur_h = next_w, next_h

        if (cur_w, cur_h) == (target_width, target_height):
            break

    if stages[-1] != (target_width, target_height):
        stages.append((target_width, target_height))

    deduped: List[Tuple[int, int]] = []
    for stage in stages:
        if not deduped or deduped[-1] != stage:
            deduped.append(stage)
    return deduped


def format_stage_plan(stages: List[Tuple[int, int]]) -> str:
    return " -> ".join(f"{width}x{height}" for width, height in stages)
