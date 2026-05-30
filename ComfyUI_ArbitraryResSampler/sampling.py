from __future__ import annotations

from typing import Optional

import torch

import comfy.sample
import comfy.utils

try:
    import latent_preview
except Exception:
    latent_preview = None

from .constants import LATENT_SCALE_FACTOR
from .latent_ops import low_frequency_preserve, make_empty_latent, resize_latent
from .noise import inject_fractal_noise
from .resolution import (
    build_stage_schedule,
    fit_base_resolution,
    format_stage_plan,
    normalize_resolution,
    pixels_from_latent_size,
)
from .tiles import feather_mask, tile_positions
from .ar_sampler import sample_latent_ar_fusion


def sample_latent(
    model,
    seed: int,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    positive,
    negative,
    latent: dict,
    denoise: float = 1.0,
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

    noise_mask = latent.get("noise_mask", None)
    callback = latent_preview.prepare_callback(model, steps) if latent_preview is not None else None
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    samples = comfy.sample.sample(
        model,
        noise,
        int(steps),
        float(cfg),
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise=float(denoise),
        disable_noise=disable_noise,
        start_step=None,
        last_step=None,
        force_full_denoise=False,
        noise_mask=noise_mask,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=int(seed),
    )

    output = latent.copy()
    output["samples"] = samples
    return output


def tiled_refine(
    model,
    seed: int,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    positive,
    negative,
    latent: dict,
    tile_pixels: int,
    overlap_pixels: int,
    halo_pixels: int,
    local_denoise: float,
    fractal_strength: float,
    octaves: int,
    persistence: float,
    lowfreq_preservation: float,
    lowfreq_factor: int,
    tile_seed_mode: str,
    conditioning_mode: str,
    local_sampler: str = "legacy",
    guidance_heatmap: Optional[torch.Tensor] = None,
    guidance_mode: str = "off",
    source_anchor_strength: float = 0.0,
    heatmap_strength: float = 0.0,
    heatmap_gamma: float = 1.0,
    source_color_preserve: float = 0.0,
) -> dict:
    if local_sampler == "ar_fusion":
        return sample_latent_ar_fusion(
            model=model,
            seed=seed,
            steps=steps,
            cfg=cfg,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent=latent,
            denoise=local_denoise,
            tile_pixels=tile_pixels,
            overlap_pixels=overlap_pixels,
            halo_pixels=halo_pixels,
            lowfreq_preservation=lowfreq_preservation,
            lowfreq_factor=lowfreq_factor,
            conditioning_mode=conditioning_mode,
            tile_mode="always",
            tile_threshold_pixels=tile_pixels,
            disable_noise=False,
            guidance_heatmap=guidance_heatmap,
            guidance_mode=guidance_mode,
            source_anchor_strength=source_anchor_strength,
            heatmap_strength=heatmap_strength,
            heatmap_gamma=heatmap_gamma,
            source_color_preserve=source_color_preserve,
        )

    samples = latent["samples"]
    batch, _, latent_height, latent_width = samples.shape
    if batch != 1:
        raise ValueError("ArbitraryRes sampler currently supports batch_size=1 only.")

    tile_width = max(8, int(round(tile_pixels / LATENT_SCALE_FACTOR)))
    tile_height = max(8, int(round(tile_pixels / LATENT_SCALE_FACTOR)))
    overlap_width = max(0, int(round(overlap_pixels / LATENT_SCALE_FACTOR)))
    overlap_height = max(0, int(round(overlap_pixels / LATENT_SCALE_FACTOR)))

    xs = tile_positions(latent_width, tile_width, overlap_width)
    ys = tile_positions(latent_height, tile_height, overlap_height)

    accumulator = torch.zeros_like(samples)
    weights = torch.zeros((1, 1, latent_height, latent_width), device=samples.device, dtype=samples.dtype)
    source_samples = samples.clone()

    tile_index = 0
    for y in ys:
        for x in xs:
            current_height = min(tile_height, latent_height - y)
            current_width = min(tile_width, latent_width - x)

            tile_samples = samples[:, :, y:y + current_height, x:x + current_width].clone()
            tile_latent = {"samples": tile_samples}

            if tile_seed_mode == "shared":
                tile_seed = int(seed)
            else:
                tile_seed = int(seed) + tile_index * 9973

            if fractal_strength > 0.0:
                tile_latent = inject_fractal_noise(
                    tile_latent,
                    seed=tile_seed,
                    strength=fractal_strength,
                    octaves=octaves,
                    persistence=persistence,
                )

            sampled_tile = sample_latent(
                model=model,
                seed=tile_seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=positive,
                negative=negative,
                latent=tile_latent,
                denoise=local_denoise,
                disable_noise=False,
            )

            weight = feather_mask(
                current_height,
                current_width,
                overlap_height,
                overlap_width,
                sampled_tile["samples"].device,
                sampled_tile["samples"].dtype,
            )

            accumulator[:, :, y:y + current_height, x:x + current_width] += sampled_tile["samples"] * weight
            weights[:, :, y:y + current_height, x:x + current_width] += weight
            tile_index += 1

    refined = accumulator / weights.clamp_min(1e-6)
    refined = low_frequency_preserve(source_samples, refined, lowfreq_factor, lowfreq_preservation)

    output = latent.copy()
    output["samples"] = refined
    return output


def hierarchical_sample(
    model,
    positive,
    negative,
    seed: int,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    target_width: int,
    target_height: int,
    base_pixels: int,
    max_scale_per_stage: float,
    global_denoise: float,
    local_denoise: float,
    global_max_megapixels: float,
    tile_pixels: int,
    overlap_pixels: int,
    halo_pixels: int,
    fractal_strength: float,
    octaves: int,
    persistence: float,
    lowfreq_preservation: float,
    lowfreq_factor: int,
    upscale_mode: str,
    tile_seed_mode: str,
    conditioning_mode: str,
    local_sampler: str = "legacy",
    same_size_refine: bool = True,
    source_latent: Optional[dict] = None,
    guidance_heatmap: Optional[torch.Tensor] = None,
    guidance_mode: str = "off",
    source_anchor_strength: float = 0.0,
    heatmap_strength: float = 0.0,
    heatmap_gamma: float = 1.0,
    source_color_preserve: float = 0.0,
) -> tuple[dict, str]:
    target_width, target_height = normalize_resolution(target_width, target_height)

    if source_latent is None:
        start_width, start_height = fit_base_resolution(target_width, target_height, base_pixels)
        stages = build_stage_schedule(start_width, start_height, target_width, target_height, max_scale_per_stage)
        latent = make_empty_latent(start_width, start_height)
        latent = sample_latent(
            model=model,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent=latent,
            denoise=1.0,
            disable_noise=False,
        )
    else:
        latent = source_latent.copy()
        _, _, latent_h, latent_w = latent["samples"].shape
        start_width, start_height = pixels_from_latent_size(latent_w, latent_h)
        stages = build_stage_schedule(start_width, start_height, target_width, target_height, max_scale_per_stage)
        if (start_width, start_height) != stages[0]:
            latent = resize_latent(latent, stages[0][0], stages[0][1], mode=upscale_mode)

    if source_latent is not None and len(stages) == 1 and bool(same_size_refine):
        stage_index = 0
        stage_width, stage_height = stages[0]

        if fractal_strength > 0.0:
            latent = inject_fractal_noise(
                latent,
                seed=int(seed) + 99991,
                strength=fractal_strength,
                octaves=octaves,
                persistence=persistence,
            )

        megapixels = (stage_width * stage_height) / 1_000_000.0
        if global_denoise > 0.0 and megapixels <= float(global_max_megapixels):
            latent = sample_latent(
                model=model,
                seed=int(seed) + 31337,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=positive,
                negative=negative,
                latent=latent,
                denoise=global_denoise,
                disable_noise=False,
            )

        if local_denoise > 0.0:
            latent = tiled_refine(
                model=model,
                seed=int(seed) + 65537,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=positive,
                negative=negative,
                latent=latent,
                tile_pixels=tile_pixels,
                overlap_pixels=overlap_pixels,
                halo_pixels=halo_pixels,
                local_denoise=local_denoise,
                fractal_strength=0.0,
                octaves=octaves,
                persistence=persistence,
                lowfreq_preservation=lowfreq_preservation,
                lowfreq_factor=lowfreq_factor,
                tile_seed_mode=tile_seed_mode,
                conditioning_mode=conditioning_mode,
                local_sampler=local_sampler,
                guidance_heatmap=guidance_heatmap,
                guidance_mode=guidance_mode,
                source_anchor_strength=source_anchor_strength,
                heatmap_strength=heatmap_strength,
                heatmap_gamma=heatmap_gamma,
                source_color_preserve=source_color_preserve,
            )
    for stage_index, (stage_width, stage_height) in enumerate(stages[1:], start=1):
        latent = resize_latent(latent, stage_width, stage_height, mode=upscale_mode)

        if fractal_strength > 0.0:
            latent = inject_fractal_noise(
                latent,
                seed=int(seed) + stage_index * 104729,
                strength=fractal_strength,
                octaves=octaves,
                persistence=persistence,
            )

        megapixels = (stage_width * stage_height) / 1_000_000.0
        if global_denoise > 0.0 and megapixels <= float(global_max_megapixels):
            latent = sample_latent(
                model=model,
                seed=int(seed) + stage_index * 271,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=positive,
                negative=negative,
                latent=latent,
                denoise=global_denoise,
                disable_noise=False,
            )

        if local_denoise > 0.0:
            latent = tiled_refine(
                model=model,
                seed=int(seed) + stage_index * 65537,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=positive,
                negative=negative,
                latent=latent,
                tile_pixels=tile_pixels,
                overlap_pixels=overlap_pixels,
                halo_pixels=halo_pixels,
                local_denoise=local_denoise,
                fractal_strength=fractal_strength,
                octaves=octaves,
                persistence=persistence,
                lowfreq_preservation=lowfreq_preservation,
                lowfreq_factor=lowfreq_factor,
                tile_seed_mode=tile_seed_mode,
                conditioning_mode=conditioning_mode,
                local_sampler=local_sampler,
                guidance_heatmap=guidance_heatmap,
                guidance_mode=guidance_mode,
                source_anchor_strength=source_anchor_strength,
                heatmap_strength=heatmap_strength,
                heatmap_gamma=heatmap_gamma,
                source_color_preserve=source_color_preserve,
            )

    return latent, format_stage_plan(stages)

