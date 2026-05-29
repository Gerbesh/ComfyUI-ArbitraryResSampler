from __future__ import annotations

import torch
import torch.nn.functional as F

from .constants import LATENT_SCALE_FACTOR
from .resolution import ceil_div


INTERPOLATE_ALIGN_CORNERS = {"linear", "bilinear", "bicubic", "trilinear"}


def make_empty_latent(width: int, height: int, batch_size: int = 1) -> dict:
    latent_width = ceil_div(width, LATENT_SCALE_FACTOR)
    latent_height = ceil_div(height, LATENT_SCALE_FACTOR)
    samples = torch.zeros((batch_size, 4, latent_height, latent_width), device="cpu")
    return {"samples": samples}


def resize_latent(latent: dict, width: int, height: int, mode: str = "bicubic") -> dict:
    output = latent.copy()
    samples = output["samples"]
    latent_width = ceil_div(width, LATENT_SCALE_FACTOR)
    latent_height = ceil_div(height, LATENT_SCALE_FACTOR)

    kwargs = {"size": (latent_height, latent_width), "mode": mode}
    if mode in INTERPOLATE_ALIGN_CORNERS:
        kwargs["align_corners"] = False

    output["samples"] = F.interpolate(samples, **kwargs)
    return output


def low_frequency_preserve(source_samples, refined_samples, factor: int, strength: float):
    if strength <= 0.0:
        return refined_samples

    _, _, height, width = source_samples.shape
    low_height = max(1, height // max(1, int(factor)))
    low_width = max(1, width // max(1, int(factor)))

    source_low = F.interpolate(source_samples, size=(low_height, low_width), mode="area")
    refined_low = F.interpolate(refined_samples, size=(low_height, low_width), mode="area")

    source_low = F.interpolate(source_low, size=(height, width), mode="bicubic", align_corners=False)
    refined_low = F.interpolate(refined_low, size=(height, width), mode="bicubic", align_corners=False)

    return refined_samples + (source_low - refined_low) * float(strength)
