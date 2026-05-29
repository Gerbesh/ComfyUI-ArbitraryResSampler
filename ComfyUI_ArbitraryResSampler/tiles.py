from __future__ import annotations

from typing import List

import torch


def tile_positions(size: int, tile: int, overlap: int) -> List[int]:
    tile = min(max(1, int(tile)), int(size))
    overlap = max(0, min(int(overlap), tile - 1))
    step = max(1, tile - overlap)

    positions = [0]
    cursor = 0
    while cursor + tile < size:
        cursor += step
        if cursor + tile >= size:
            cursor = size - tile
        if positions[-1] == cursor:
            break
        positions.append(cursor)
    return positions


def feather_mask(height: int, width: int, overlap_y: int, overlap_x: int, device, dtype):
    y = torch.ones(height, device=device, dtype=dtype)
    x = torch.ones(width, device=device, dtype=dtype)

    overlap_y = max(0, min(int(overlap_y), max(0, height // 2)))
    overlap_x = max(0, min(int(overlap_x), max(0, width // 2)))

    if overlap_y > 0:
        ramp = torch.linspace(0.0, 1.0, steps=overlap_y + 2, device=device, dtype=dtype)[1:-1]
        y[:overlap_y] = torch.minimum(y[:overlap_y], ramp)
        y[-overlap_y:] = torch.minimum(y[-overlap_y:], torch.flip(ramp, dims=[0]))

    if overlap_x > 0:
        ramp = torch.linspace(0.0, 1.0, steps=overlap_x + 2, device=device, dtype=dtype)[1:-1]
        x[:overlap_x] = torch.minimum(x[:overlap_x], ramp)
        x[-overlap_x:] = torch.minimum(x[-overlap_x:], torch.flip(ramp, dims=[0]))

    return torch.outer(y, x).unsqueeze(0).unsqueeze(0).clamp_min(1e-3)
