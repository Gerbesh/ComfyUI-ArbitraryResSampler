from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

import comfy.sample
import comfy.samplers
import comfy.utils
import comfy.model_management

try:
    from comfy.samplers import KSamplerX0Inpaint
except Exception:  # pragma: no cover - Comfy internal fallback guard
    KSamplerX0Inpaint = None

from .constants import LATENT_SCALE_FACTOR
from .latent_ops import low_frequency_preserve
from .tiles import feather_mask, tile_positions


@dataclass(frozen=True)
class ARTiledSamplerConfig:
    tile_pixels: int = 1024
    overlap_pixels: int = 256
    global_context_strength: float = 0.35
    global_context_pixels: int = 1024
    lowfreq_factor: int = 8
    tile_mode: str = "auto"  # auto, always, off
    tile_threshold_pixels: int = 1536
    sigma_blend_start: float = 0.0
    sigma_blend_end: float = 1.0

    @property
    def tile_latents(self) -> int:
        return max(8, int(round(self.tile_pixels / LATENT_SCALE_FACTOR)))

    @property
    def overlap_latents(self) -> int:
        return max(0, int(round(self.overlap_pixels / LATENT_SCALE_FACTOR)))

    @property
    def tile_threshold_latents(self) -> int:
        return max(8, int(round(self.tile_threshold_pixels / LATENT_SCALE_FACTOR)))


def _max_denoise(model_wrap, sigmas: torch.Tensor) -> bool:
    max_sigma = float(model_wrap.inner_model.model_sampling.sigma_max)
    sigma = float(sigmas[0])
    return math.isclose(max_sigma, sigma, rel_tol=1e-5) or sigma > max_sigma


def _sigma_batch(sigma: torch.Tensor, batch_size: int) -> torch.Tensor:
    if sigma.ndim == 0:
        return sigma.repeat(batch_size)
    if sigma.shape[0] == batch_size:
        return sigma
    return sigma[:1].repeat(batch_size)


def _should_tile(x: torch.Tensor, config: ARTiledSamplerConfig) -> bool:
    if config.tile_mode == "off":
        return False
    if config.tile_mode == "always":
        return True
    _, _, height, width = x.shape
    return max(height, width) > config.tile_threshold_latents


def _crop_extra_args(extra_args: dict, y: int, x: int, height: int, width: int) -> dict:
    """Crop mask-like tensors in extra_args when they spatially match x.

    The common no-mask path is untouched. For inpaint/noise masks this avoids
    immediate shape mismatches, but complex inpaint workflows are still treated
    as experimental because Comfy internals also keep latent_image/noise state
    inside KSamplerX0Inpaint.
    """
    cropped = dict(extra_args)
    mask = cropped.get("denoise_mask", None)
    if isinstance(mask, torch.Tensor) and mask.ndim >= 3:
        cropped["denoise_mask"] = mask[..., y:y + height, x:x + width]
    return cropped


def _call_model(model, x: torch.Tensor, sigma: torch.Tensor, extra_args: dict, seed_offset: int = 0) -> torch.Tensor:
    call_args = dict(extra_args)
    if seed_offset:
        call_args["seed"] = int(call_args.get("seed", 0) or 0) + int(seed_offset)
    return model(x, _sigma_batch(sigma, x.shape[0]), **call_args)


def _tiled_model_prediction(
    model,
    x: torch.Tensor,
    sigma: torch.Tensor,
    extra_args: dict,
    config: ARTiledSamplerConfig,
) -> torch.Tensor:
    batch, channels, latent_height, latent_width = x.shape
    if batch != 1:
        raise ValueError("AR tiled sampler currently supports batch_size=1 only.")

    tile_width = min(config.tile_latents, latent_width)
    tile_height = min(config.tile_latents, latent_height)
    overlap_width = max(0, min(config.overlap_latents, tile_width - 1))
    overlap_height = max(0, min(config.overlap_latents, tile_height - 1))

    xs = tile_positions(latent_width, tile_width, overlap_width)
    ys = tile_positions(latent_height, tile_height, overlap_height)

    accumulator = torch.zeros_like(x)
    weights = torch.zeros((1, 1, latent_height, latent_width), device=x.device, dtype=x.dtype)

    tile_index = 0
    for y in ys:
        for px in xs:
            current_height = min(tile_height, latent_height - y)
            current_width = min(tile_width, latent_width - px)
            tile_x = x[:, :, y:y + current_height, px:px + current_width]
            tile_args = _crop_extra_args(extra_args, y, px, current_height, current_width)

            denoised_tile = _call_model(
                model,
                tile_x,
                sigma,
                tile_args,
                seed_offset=tile_index * 9973,
            )

            weight = feather_mask(
                current_height,
                current_width,
                overlap_height,
                overlap_width,
                device=x.device,
                dtype=x.dtype,
            )
            accumulator[:, :, y:y + current_height, px:px + current_width] += denoised_tile * weight
            weights[:, :, y:y + current_height, px:px + current_width] += weight
            tile_index += 1

    return accumulator / weights.clamp_min(1e-6)


def _global_context_prediction(
    model,
    x: torch.Tensor,
    sigma: torch.Tensor,
    extra_args: dict,
    config: ARTiledSamplerConfig,
) -> Optional[torch.Tensor]:
    if config.global_context_strength <= 0.0:
        return None

    _, _, latent_height, latent_width = x.shape
    max_context_latents = max(8, int(round(config.global_context_pixels / LATENT_SCALE_FACTOR)))
    max_side = max(latent_height, latent_width)

    if max_side <= max_context_latents:
        return _call_model(model, x, sigma, extra_args)

    scale = max_context_latents / max_side
    context_height = max(8, int(round(latent_height * scale)))
    context_width = max(8, int(round(latent_width * scale)))

    context_x = F.interpolate(x, size=(context_height, context_width), mode="bilinear", align_corners=False)
    context_prediction = _call_model(model, context_x, sigma, extra_args)
    return F.interpolate(context_prediction, size=(latent_height, latent_width), mode="bilinear", align_corners=False)


def _predict_denoised(
    model,
    x: torch.Tensor,
    sigma: torch.Tensor,
    extra_args: dict,
    config: ARTiledSamplerConfig,
) -> torch.Tensor:
    if not _should_tile(x, config):
        return _call_model(model, x, sigma, extra_args)

    tiled_prediction = _tiled_model_prediction(model, x, sigma, extra_args, config)
    global_prediction = _global_context_prediction(model, x, sigma, extra_args, config)

    if global_prediction is not None and config.global_context_strength > 0.0:
        tiled_prediction = low_frequency_preserve(
            source_samples=global_prediction,
            refined_samples=tiled_prediction,
            factor=config.lowfreq_factor,
            strength=config.global_context_strength,
        )

    return tiled_prediction


class AREulerTiledFusionSampler:
    """Comfy-compatible sampler object using tiled per-step denoised fusion.

    This is intentionally not registered as a global Comfy sampler. The node
    passes an instance directly into comfy.sample.sample_custom(...), so the
    plugin remains isolated and reversible.
    """

    def __init__(self, config: Optional[ARTiledSamplerConfig] = None):
        self.config = config or ARTiledSamplerConfig()

    def sample(
        self,
        model_wrap,
        sigmas,
        extra_args,
        callback,
        noise,
        latent_image=None,
        denoise_mask=None,
        disable_pbar=False,
    ):
        if KSamplerX0Inpaint is None:
            raise RuntimeError("Comfy KSamplerX0Inpaint is not available; update ComfyUI or use the legacy hierarchical sampler.")

        extra_args = dict(extra_args)
        extra_args["denoise_mask"] = denoise_mask

        model = KSamplerX0Inpaint(model_wrap, sigmas)
        model.latent_image = latent_image
        model.noise = noise

        x = model_wrap.inner_model.model_sampling.noise_scaling(
            sigmas[0],
            noise,
            latent_image,
            _max_denoise(model_wrap, sigmas),
        )

        total_steps = max(0, len(sigmas) - 1)
        pbar = None if disable_pbar else comfy.utils.ProgressBar(total_steps)

        for i in range(total_steps):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]

            if float(sigma) <= 0.0:
                continue

            denoised = _predict_denoised(model, x, sigma, extra_args, self.config)
            derivative = (x - denoised) / sigma
            dt = sigma_next - sigma
            x = x + derivative * dt

            if callback is not None:
                callback({"x": x, "i": i, "sigma": sigma, "sigma_hat": sigma, "denoised": denoised})
            if pbar is not None:
                pbar.update(1)

        return model_wrap.inner_model.model_sampling.inverse_noise_scaling(sigmas[-1], x)


def calculate_sigmas(model, steps: int, scheduler: str, denoise: float) -> torch.Tensor:
    sampler = comfy.samplers.KSampler(
        model,
        steps=int(steps),
        device=model.load_device,
        sampler="euler",
        scheduler=scheduler,
        denoise=float(denoise),
        model_options=model.model_options,
    )
    return sampler.sigmas


def sample_latent_ar_fusion(
    model,
    seed: int,
    steps: int,
    cfg: float,
    scheduler: str,
    positive,
    negative,
    latent: dict,
    denoise: float,
    tile_pixels: int,
    overlap_pixels: int,
    lowfreq_preservation: float,
    lowfreq_factor: int,
    global_context_pixels: int = 1024,
    tile_mode: str = "auto",
    tile_threshold_pixels: int = 1536,
    disable_noise: bool = False,
) -> dict:
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)

    if disable_noise:
        noise = torch.zeros(
            latent_image.size(),
            dtype=latent_image.dtype,
            layout=latent_image.layout,
            device="cpu",
        )
    else:
        batch_indices = latent.get("batch_index", None)
        noise = comfy.sample.prepare_noise(latent_image, int(seed), batch_indices)

    sigmas = calculate_sigmas(model, steps=steps, scheduler=scheduler, denoise=denoise)
    sampler = AREulerTiledFusionSampler(
        ARTiledSamplerConfig(
            tile_pixels=int(tile_pixels),
            overlap_pixels=int(overlap_pixels),
            global_context_strength=float(lowfreq_preservation),
            global_context_pixels=int(global_context_pixels),
            lowfreq_factor=int(lowfreq_factor),
            tile_mode=tile_mode,
            tile_threshold_pixels=int(tile_threshold_pixels),
        )
    )

    samples = comfy.sample.sample_custom(
        model=model,
        noise=noise,
        cfg=float(cfg),
        sampler=sampler,
        sigmas=sigmas,
        positive=positive,
        negative=negative,
        latent_image=latent_image,
        noise_mask=latent.get("noise_mask", None),
        callback=None,
        disable_pbar=False,
        seed=int(seed),
    )

    output = latent.copy()
    output["samples"] = samples
    return output
