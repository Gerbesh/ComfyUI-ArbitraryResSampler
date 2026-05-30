from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

import comfy.sample
import comfy.samplers
import comfy.utils

try:
    from comfy.samplers import KSamplerX0Inpaint
except Exception:
    KSamplerX0Inpaint = None

from .constants import LATENT_SCALE_FACTOR
from .latent_ops import low_frequency_preserve
from .sdxl_conditioning import apply_tile_conditioning
from .tiles import feather_mask, tile_positions


@dataclass(frozen=True)
class ARTiledSamplerConfig:
    tile_pixels: int = 1024
    overlap_pixels: int = 256
    halo_pixels: int = 192
    global_context_strength: float = 0.35
    global_context_pixels: int = 1024
    lowfreq_factor: int = 8
    tile_mode: str = "auto"  # auto, always, off
    tile_threshold_pixels: int = 1536
    conditioning_mode: str = "sdxl_tile_crop"  # plain, sdxl_tile_crop

    @property
    def tile_latents(self) -> int:
        return max(8, int(round(self.tile_pixels / LATENT_SCALE_FACTOR)))

    @property
    def overlap_latents(self) -> int:
        return max(0, int(round(self.overlap_pixels / LATENT_SCALE_FACTOR)))

    @property
    def halo_latents(self) -> int:
        return max(0, int(round(self.halo_pixels / LATENT_SCALE_FACTOR)))

    @property
    def tile_threshold_latents(self) -> int:
        return max(8, int(round(self.tile_threshold_pixels / LATENT_SCALE_FACTOR)))


@dataclass(frozen=True)
class ARGuidanceConfig:
    heatmap: Optional[torch.Tensor] = None
    mode: str = "off"  # off, detail, preserve, balanced
    source_anchor_strength: float = 0.0
    heatmap_strength: float = 0.0
    heatmap_gamma: float = 1.0
    source_color_preserve: float = 0.0


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


def _crop_spatial_tensor(tensor: torch.Tensor, full_h: int, full_w: int, y: int, x: int, h: int, w: int):
    if tensor.ndim >= 2 and tensor.shape[-2] == full_h and tensor.shape[-1] == full_w:
        return tensor[..., y:y + h, x:x + w]
    return tensor


def _crop_nested(obj, full_h: int, full_w: int, y: int, x: int, h: int, w: int):
    if isinstance(obj, torch.Tensor):
        return _crop_spatial_tensor(obj, full_h, full_w, y, x, h, w)

    if isinstance(obj, dict):
        return {key: _crop_nested(value, full_h, full_w, y, x, h, w) for key, value in obj.items()}

    if isinstance(obj, list):
        return [_crop_nested(value, full_h, full_w, y, x, h, w) for value in obj]

    if isinstance(obj, tuple):
        return tuple(_crop_nested(value, full_h, full_w, y, x, h, w) for value in obj)

    return obj


def _prepare_guidance_heatmap(
    heatmap: Optional[torch.Tensor],
    latent_height: int,
    latent_width: int,
    device,
    dtype,
) -> Optional[torch.Tensor]:
    if heatmap is None:
        return None

    heat = heatmap
    if heat.ndim == 4:
        if heat.shape[-1] in (1, 3, 4):  # BHWC IMAGE
            heat = heat[..., :1].permute(0, 3, 1, 2).contiguous()
        elif heat.shape[1] in (1, 3, 4):  # BCHW
            heat = heat[:, :1, :, :].contiguous()
        else:
            raise ValueError(f"Unsupported heatmap shape: {tuple(heat.shape)}")
    elif heat.ndim == 3:
        if heat.shape[-1] in (1, 3, 4):  # HWC
            heat = heat[..., :1].unsqueeze(0).permute(0, 3, 1, 2).contiguous()
        else:  # BHW
            heat = heat.unsqueeze(1).contiguous()
    elif heat.ndim == 2:
        heat = heat.unsqueeze(0).unsqueeze(0).contiguous()
    else:
        raise ValueError(f"Unsupported heatmap shape: {tuple(heat.shape)}")

    heat = heat.to(device=device, dtype=dtype)
    if heat.shape[0] != 1:
        heat = heat[:1]

    heat = F.interpolate(
        heat,
        size=(latent_height, latent_width),
        mode="bilinear",
        align_corners=False,
    )
    return heat.clamp(0.0, 1.0)


def _guided_source_weight(
    heat_core: Optional[torch.Tensor],
    guidance: Optional[ARGuidanceConfig],
    reference: torch.Tensor,
) -> Optional[torch.Tensor]:
    if guidance is None or guidance.mode == "off":
        return None

    anchor = float(guidance.source_anchor_strength)
    strength = float(guidance.heatmap_strength)

    if anchor <= 0.0 and strength <= 0.0:
        return None

    if heat_core is None:
        heat = torch.zeros(
            (reference.shape[0], 1, reference.shape[-2], reference.shape[-1]),
            device=reference.device,
            dtype=reference.dtype,
        )
    else:
        gamma = max(0.05, float(guidance.heatmap_gamma))
        heat = heat_core.clamp(0.0, 1.0) ** gamma

    mode = str(guidance.mode)
    if mode == "detail":
        # high heat means this region needs more generated detail, so less source anchoring
        source_weight = anchor - heat * strength
    elif mode == "preserve":
        # high heat means risk/protection mask, so more source anchoring
        source_weight = anchor + heat * strength
    elif mode == "balanced":
        # heat above 0.5 gets more model freedom, heat below 0.5 gets more source lock
        source_weight = anchor + (0.5 - heat) * strength
    else:
        source_weight = torch.full_like(heat, anchor)

    return source_weight.clamp(0.0, 0.95)


def _apply_guided_core_blend(
    denoised_core: torch.Tensor,
    source_core: Optional[torch.Tensor],
    heat_core: Optional[torch.Tensor],
    guidance: Optional[ARGuidanceConfig],
) -> torch.Tensor:
    if source_core is None or guidance is None or guidance.mode == "off":
        return denoised_core

    if source_core.shape != denoised_core.shape:
        return denoised_core

    source_weight = _guided_source_weight(heat_core, guidance, denoised_core)
    if source_weight is None:
        return denoised_core

    return denoised_core * (1.0 - source_weight) + source_core * source_weight


def _prepare_tile_extra_args(
    extra_args: dict,
    full_latent_h: int,
    full_latent_w: int,
    tile_y: int,
    tile_x: int,
    tile_h: int,
    tile_w: int,
    conditioning_mode: str,
) -> dict:
    cropped = _crop_nested(extra_args, full_latent_h, full_latent_w, tile_y, tile_x, tile_h, tile_w)

    if conditioning_mode != "plain":
        cropped = apply_tile_conditioning(
            cropped,
            full_width=int(full_latent_w * LATENT_SCALE_FACTOR),
            full_height=int(full_latent_h * LATENT_SCALE_FACTOR),
            crop_x=int(tile_x * LATENT_SCALE_FACTOR),
            crop_y=int(tile_y * LATENT_SCALE_FACTOR),
            target_width=int(tile_w * LATENT_SCALE_FACTOR),
            target_height=int(tile_h * LATENT_SCALE_FACTOR),
            mode=conditioning_mode,
        )

    return cropped


def _prepare_global_context_args(
    extra_args: dict,
    full_latent_h: int,
    full_latent_w: int,
    context_h: int,
    context_w: int,
    conditioning_mode: str,
) -> dict:
    prepared = dict(extra_args)
    if conditioning_mode != "plain":
        prepared = apply_tile_conditioning(
            prepared,
            full_width=int(full_latent_w * LATENT_SCALE_FACTOR),
            full_height=int(full_latent_h * LATENT_SCALE_FACTOR),
            crop_x=0,
            crop_y=0,
            target_width=int(context_w * LATENT_SCALE_FACTOR),
            target_height=int(context_h * LATENT_SCALE_FACTOR),
            mode=conditioning_mode,
        )
    return prepared


def _call_model(model, x: torch.Tensor, sigma: torch.Tensor, extra_args: dict) -> torch.Tensor:
    return model(x, _sigma_batch(sigma, x.shape[0]), **extra_args)


def _tiled_model_prediction(
    model,
    x: torch.Tensor,
    sigma: torch.Tensor,
    extra_args: dict,
    config: ARTiledSamplerConfig,
    source_samples: Optional[torch.Tensor] = None,
    guidance: Optional[ARGuidanceConfig] = None,
) -> torch.Tensor:
    batch, _, latent_height, latent_width = x.shape
    if batch != 1:
        raise ValueError("AR tiled sampler currently supports batch_size=1 only.")

    core_width = min(config.tile_latents, latent_width)
    core_height = min(config.tile_latents, latent_height)
    overlap_width = max(0, min(config.overlap_latents, core_width - 1))
    overlap_height = max(0, min(config.overlap_latents, core_height - 1))
    halo = config.halo_latents

    xs = tile_positions(latent_width, core_width, overlap_width)
    ys = tile_positions(latent_height, core_height, overlap_height)

    accumulator = torch.zeros_like(x)
    weights = torch.zeros((1, 1, latent_height, latent_width), device=x.device, dtype=x.dtype)

    guidance_heatmap = None
    if guidance is not None and guidance.mode != "off":
        guidance_heatmap = _prepare_guidance_heatmap(
            guidance.heatmap,
            latent_height,
            latent_width,
            device=x.device,
            dtype=x.dtype,
        )

    for core_y in ys:
        for core_x in xs:
            current_core_height = min(core_height, latent_height - core_y)
            current_core_width = min(core_width, latent_width - core_x)

            expanded_y0 = max(0, core_y - halo)
            expanded_x0 = max(0, core_x - halo)
            expanded_y1 = min(latent_height, core_y + current_core_height + halo)
            expanded_x1 = min(latent_width, core_x + current_core_width + halo)

            expanded_height = expanded_y1 - expanded_y0
            expanded_width = expanded_x1 - expanded_x0

            tile_x = x[:, :, expanded_y0:expanded_y1, expanded_x0:expanded_x1]

            tile_args = _prepare_tile_extra_args(
                extra_args=extra_args,
                full_latent_h=latent_height,
                full_latent_w=latent_width,
                tile_y=expanded_y0,
                tile_x=expanded_x0,
                tile_h=expanded_height,
                tile_w=expanded_width,
                conditioning_mode=config.conditioning_mode,
            )

            denoised_expanded = _call_model(model, tile_x, sigma, tile_args)

            core_rel_y = core_y - expanded_y0
            core_rel_x = core_x - expanded_x0
            denoised_core = denoised_expanded[
                :,
                :,
                core_rel_y:core_rel_y + current_core_height,
                core_rel_x:core_rel_x + current_core_width,
            ]

            source_core = None
            if source_samples is not None and source_samples.shape[-2:] == x.shape[-2:]:
                source_core = source_samples[
                    :,
                    :,
                    core_y:core_y + current_core_height,
                    core_x:core_x + current_core_width,
                ]

            heat_core = None
            if guidance_heatmap is not None:
                heat_core = guidance_heatmap[
                    :,
                    :,
                    core_y:core_y + current_core_height,
                    core_x:core_x + current_core_width,
                ]

            denoised_core = _apply_guided_core_blend(
                denoised_core=denoised_core,
                source_core=source_core,
                heat_core=heat_core,
                guidance=guidance,
            )

            weight = feather_mask(
                current_core_height,
                current_core_width,
                overlap_height,
                overlap_width,
                device=x.device,
                dtype=x.dtype,
            )

            accumulator[:, :, core_y:core_y + current_core_height, core_x:core_x + current_core_width] += denoised_core * weight
            weights[:, :, core_y:core_y + current_core_height, core_x:core_x + current_core_width] += weight

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
        context_args = _prepare_global_context_args(
            extra_args=extra_args,
            full_latent_h=latent_height,
            full_latent_w=latent_width,
            context_h=latent_height,
            context_w=latent_width,
            conditioning_mode=config.conditioning_mode,
        )
        return _call_model(model, x, sigma, context_args)

    scale = max_context_latents / max_side
    context_height = max(8, int(round(latent_height * scale)))
    context_width = max(8, int(round(latent_width * scale)))

    context_x = F.interpolate(x, size=(context_height, context_width), mode="bilinear", align_corners=False)
    context_args = _prepare_global_context_args(
        extra_args=extra_args,
        full_latent_h=latent_height,
        full_latent_w=latent_width,
        context_h=context_height,
        context_w=context_width,
        conditioning_mode=config.conditioning_mode,
    )
    context_prediction = _call_model(model, context_x, sigma, context_args)
    return F.interpolate(context_prediction, size=(latent_height, latent_width), mode="bilinear", align_corners=False)


def _predict_denoised(
    model,
    x: torch.Tensor,
    sigma: torch.Tensor,
    extra_args: dict,
    config: ARTiledSamplerConfig,
    source_samples: Optional[torch.Tensor] = None,
    guidance: Optional[ARGuidanceConfig] = None,
) -> torch.Tensor:
    if not _should_tile(x, config):
        direct_args = _prepare_global_context_args(
            extra_args=extra_args,
            full_latent_h=x.shape[-2],
            full_latent_w=x.shape[-1],
            context_h=x.shape[-2],
            context_w=x.shape[-1],
            conditioning_mode=config.conditioning_mode,
        )
        direct_prediction = _call_model(model, x, sigma, direct_args)
        if guidance is not None and guidance.mode != "off" and source_samples is not None:
            heat = _prepare_guidance_heatmap(
                guidance.heatmap,
                x.shape[-2],
                x.shape[-1],
                device=x.device,
                dtype=x.dtype,
            )
            direct_prediction = _apply_guided_core_blend(
                denoised_core=direct_prediction,
                source_core=source_samples if source_samples.shape == direct_prediction.shape else None,
                heat_core=heat,
                guidance=guidance,
            )
        return direct_prediction

    tiled_prediction = _tiled_model_prediction(
        model,
        x,
        sigma,
        extra_args,
        config,
        source_samples=source_samples,
        guidance=guidance,
    )
    global_prediction = _global_context_prediction(model, x, sigma, extra_args, config)

    if global_prediction is not None and config.global_context_strength > 0.0:
        tiled_prediction = low_frequency_preserve(
            source_samples=global_prediction,
            refined_samples=tiled_prediction,
            factor=config.lowfreq_factor,
            strength=config.global_context_strength,
        )

    if (
        guidance is not None
        and source_samples is not None
        and float(guidance.source_color_preserve) > 0.0
        and source_samples.shape == tiled_prediction.shape
    ):
        tiled_prediction = low_frequency_preserve(
            source_samples=source_samples,
            refined_samples=tiled_prediction,
            factor=config.lowfreq_factor,
            strength=float(guidance.source_color_preserve),
        )

    return tiled_prediction


class AREulerTiledFusionSampler:
    def __init__(
        self,
        config: Optional[ARTiledSamplerConfig] = None,
        guidance: Optional[ARGuidanceConfig] = None,
    ):
        self.config = config or ARTiledSamplerConfig()
        self.guidance = guidance

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
            raise RuntimeError("Comfy KSamplerX0Inpaint is not available; update ComfyUI or switch back to the legacy sampler.")

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

            denoised = _predict_denoised(
                model,
                x,
                sigma,
                extra_args,
                self.config,
                source_samples=latent_image,
                guidance=self.guidance,
            )
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
    halo_pixels: int,
    lowfreq_preservation: float,
    lowfreq_factor: int,
    conditioning_mode: str = "sdxl_tile_crop",
    global_context_pixels: int = 1024,
    tile_mode: str = "auto",
    tile_threshold_pixels: int = 1536,
    disable_noise: bool = False,
    guidance_heatmap: Optional[torch.Tensor] = None,
    guidance_mode: str = "off",
    source_anchor_strength: float = 0.0,
    heatmap_strength: float = 0.0,
    heatmap_gamma: float = 1.0,
    source_color_preserve: float = 0.0,
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
    guidance = None
    if str(guidance_mode) != "off" and (
        guidance_heatmap is not None
        or float(source_anchor_strength) > 0.0
        or float(source_color_preserve) > 0.0
    ):
        guidance = ARGuidanceConfig(
            heatmap=guidance_heatmap,
            mode=str(guidance_mode),
            source_anchor_strength=float(source_anchor_strength),
            heatmap_strength=float(heatmap_strength),
            heatmap_gamma=float(heatmap_gamma),
            source_color_preserve=float(source_color_preserve),
        )

    sampler = AREulerTiledFusionSampler(
        ARTiledSamplerConfig(
            tile_pixels=int(tile_pixels),
            overlap_pixels=int(overlap_pixels),
            halo_pixels=int(halo_pixels),
            global_context_strength=float(lowfreq_preservation),
            global_context_pixels=int(global_context_pixels),
            lowfreq_factor=int(lowfreq_factor),
            tile_mode=tile_mode,
            tile_threshold_pixels=int(tile_threshold_pixels),
            conditioning_mode=conditioning_mode,
        ),
        guidance=guidance,
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
        disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
        seed=int(seed),
    )

    output = latent.copy()
    output["samples"] = samples
    return output