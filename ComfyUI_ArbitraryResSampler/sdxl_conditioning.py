from __future__ import annotations

from typing import Any


def _is_conditioning_sequence(value: Any) -> bool:
    if not isinstance(value, list) or len(value) == 0:
        return False
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return False
        if not isinstance(item[1], dict):
            return False
    return True


def _patch_sdxl_meta(
    meta: dict,
    full_width: int,
    full_height: int,
    crop_x: int,
    crop_y: int,
    target_width: int,
    target_height: int,
) -> dict:
    patched = dict(meta)

    if "width" in patched:
        patched["width"] = int(full_width)
    if "height" in patched:
        patched["height"] = int(full_height)
    if "crop_w" in patched:
        patched["crop_w"] = int(crop_x)
    if "crop_h" in patched:
        patched["crop_h"] = int(crop_y)
    if "target_width" in patched:
        patched["target_width"] = int(target_width)
    if "target_height" in patched:
        patched["target_height"] = int(target_height)

    return patched


def build_tile_conditioning(
    conditioning,
    full_width: int,
    full_height: int,
    crop_x: int,
    crop_y: int,
    target_width: int,
    target_height: int,
):
    output = []
    for item in conditioning:
        cond_tensor, cond_meta = item
        patched_meta = apply_tile_conditioning(
            cond_meta,
            full_width=full_width,
            full_height=full_height,
            crop_x=crop_x,
            crop_y=crop_y,
            target_width=target_width,
            target_height=target_height,
            mode="sdxl_tile_crop",
        )
        if isinstance(item, list):
            output.append([cond_tensor, patched_meta])
        else:
            output.append((cond_tensor, patched_meta))
    return output


def apply_tile_conditioning(
    obj: Any,
    full_width: int,
    full_height: int,
    crop_x: int,
    crop_y: int,
    target_width: int,
    target_height: int,
    mode: str = "sdxl_tile_crop",
):
    if mode == "plain":
        return obj

    if _is_conditioning_sequence(obj):
        return build_tile_conditioning(
            obj,
            full_width=full_width,
            full_height=full_height,
            crop_x=crop_x,
            crop_y=crop_y,
            target_width=target_width,
            target_height=target_height,
        )

    if isinstance(obj, dict):
        patched = {}
        for key, value in obj.items():
            if isinstance(value, (dict, list, tuple)):
                patched[key] = apply_tile_conditioning(
                    value,
                    full_width=full_width,
                    full_height=full_height,
                    crop_x=crop_x,
                    crop_y=crop_y,
                    target_width=target_width,
                    target_height=target_height,
                    mode=mode,
                )
            else:
                patched[key] = value

        if any(k in patched for k in ("width", "height", "crop_w", "crop_h", "target_width", "target_height")):
            patched = _patch_sdxl_meta(
                patched,
                full_width=full_width,
                full_height=full_height,
                crop_x=crop_x,
                crop_y=crop_y,
                target_width=target_width,
                target_height=target_height,
            )
        return patched

    if isinstance(obj, list):
        return [
            apply_tile_conditioning(
                value,
                full_width=full_width,
                full_height=full_height,
                crop_x=crop_x,
                crop_y=crop_y,
                target_width=target_width,
                target_height=target_height,
                mode=mode,
            )
            if isinstance(value, (dict, list, tuple))
            else value
            for value in obj
        ]

    if isinstance(obj, tuple):
        return tuple(
            apply_tile_conditioning(
                value,
                full_width=full_width,
                full_height=full_height,
                crop_x=crop_x,
                crop_y=crop_y,
                target_width=target_width,
                target_height=target_height,
                mode=mode,
            )
            if isinstance(value, (dict, list, tuple))
            else value
            for value in obj
        )

    return obj