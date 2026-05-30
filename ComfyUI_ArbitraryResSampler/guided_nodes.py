from __future__ import annotations

import comfy.samplers

from .sampling import hierarchical_sample


def _preset_values(
    preset: str,
    strength: float,
    color_preserve: float,
    structure_preserve: float,
    detail_bias: float,
) -> dict:
    strength = float(strength)
    color_preserve = float(color_preserve)
    structure_preserve = float(structure_preserve)
    detail_bias = float(detail_bias)

    presets = {
        "clean_x2": {
            "global_denoise": 0.12,
            "local_denoise": 0.18,
            "lowfreq_preservation": 0.55,
            "fractal_strength": 0.0,
            "guidance_mode": "detail",
            "source_anchor_strength": 0.22,
            "heatmap_strength": 0.22,
            "source_color_preserve": 0.60,
            "max_scale_per_stage": 1.40,
            "global_max_megapixels": 8.0,
        },
        "strong_x2": {
            "global_denoise": 0.16,
            "local_denoise": 0.24,
            "lowfreq_preservation": 0.48,
            "fractal_strength": 0.01,
            "guidance_mode": "detail",
            "source_anchor_strength": 0.18,
            "heatmap_strength": 0.34,
            "source_color_preserve": 0.52,
            "max_scale_per_stage": 1.45,
            "global_max_megapixels": 8.0,
        },
        "light_rerender": {
            "global_denoise": 0.20,
            "local_denoise": 0.28,
            "lowfreq_preservation": 0.40,
            "fractal_strength": 0.015,
            "guidance_mode": "balanced",
            "source_anchor_strength": 0.12,
            "heatmap_strength": 0.42,
            "source_color_preserve": 0.45,
            "max_scale_per_stage": 1.45,
            "global_max_megapixels": 8.0,
        },
        "anti_blotch": {
            "global_denoise": 0.10,
            "local_denoise": 0.16,
            "lowfreq_preservation": 0.65,
            "fractal_strength": 0.0,
            "guidance_mode": "preserve",
            "source_anchor_strength": 0.28,
            "heatmap_strength": 0.42,
            "source_color_preserve": 0.78,
            "max_scale_per_stage": 1.35,
            "global_max_megapixels": 8.0,
        },
        "detail_recovery": {
            "global_denoise": 0.12,
            "local_denoise": 0.22,
            "lowfreq_preservation": 0.50,
            "fractal_strength": 0.005,
            "guidance_mode": "detail",
            "source_anchor_strength": 0.16,
            "heatmap_strength": 0.48,
            "source_color_preserve": 0.56,
            "max_scale_per_stage": 1.40,
            "global_max_megapixels": 8.0,
        },
    }

    cfg = dict(presets.get(preset, presets["clean_x2"]))

    # One UX slider should scale the amount of intervention, not silently destroy fidelity.
    cfg["global_denoise"] = max(0.0, min(1.0, cfg["global_denoise"] * (0.65 + 0.70 * strength)))
    cfg["local_denoise"] = max(0.0, min(1.0, cfg["local_denoise"] * (0.65 + 0.70 * strength)))
    cfg["heatmap_strength"] = max(0.0, min(1.0, cfg["heatmap_strength"] * max(0.0, detail_bias) * max(0.0, strength)))
    cfg["source_anchor_strength"] = max(0.0, min(0.95, cfg["source_anchor_strength"] * max(0.0, structure_preserve)))
    cfg["source_color_preserve"] = max(0.0, min(1.0, cfg["source_color_preserve"] * max(0.0, color_preserve)))
    cfg["lowfreq_preservation"] = max(0.0, min(1.0, cfg["lowfreq_preservation"] * (0.75 + 0.50 * color_preserve)))

    return cfg


class ARImageGuidedHierarchicalSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "source_latent": ("LATENT",),
                "heatmap": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
                "steps": ("INT", {"default": 34, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 5.8, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "target_width": ("INT", {"default": 5000, "min": 64, "max": 65535}),
                "target_height": ("INT", {"default": 2800, "min": 64, "max": 65535}),
                "preset": (["clean_x2", "strong_x2", "light_rerender", "anti_blotch", "detail_recovery"], {"default": "clean_x2"}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "color_preserve": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "structure_preserve": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "detail_bias": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "tile_pixels": ("INT", {"default": 1536, "min": 256, "max": 4096, "step": 64}),
                "overlap_pixels": ("INT", {"default": 320, "min": 0, "max": 2048, "step": 32}),
                "halo_pixels": ("INT", {"default": 256, "min": 0, "max": 2048, "step": 32}),
                "heatmap_gamma": ("FLOAT", {"default": 1.15, "min": 0.1, "max": 4.0, "step": 0.05}),
                "upscale_mode": (["bicubic", "bilinear", "nearest"], {"default": "bicubic"}),
                "tile_seed_mode": (["offset", "shared"], {"default": "offset"}),
            },
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "plan")
    FUNCTION = "run"
    CATEGORY = "sampling/arbitrary_res"

    def run(
        self,
        model,
        positive,
        negative,
        source_latent,
        heatmap,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        target_width,
        target_height,
        preset,
        strength,
        color_preserve,
        structure_preserve,
        detail_bias,
        tile_pixels,
        overlap_pixels,
        halo_pixels,
        heatmap_gamma,
        upscale_mode,
        tile_seed_mode,
    ):
        p = _preset_values(
            preset=preset,
            strength=strength,
            color_preserve=color_preserve,
            structure_preserve=structure_preserve,
            detail_bias=detail_bias,
        )

        latent, plan = hierarchical_sample(
            model=model,
            positive=positive,
            negative=negative,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            target_width=target_width,
            target_height=target_height,
            base_pixels=1024,
            max_scale_per_stage=p["max_scale_per_stage"],
            global_denoise=p["global_denoise"],
            local_denoise=p["local_denoise"],
            global_max_megapixels=p["global_max_megapixels"],
            tile_pixels=tile_pixels,
            overlap_pixels=overlap_pixels,
            halo_pixels=halo_pixels,
            fractal_strength=p["fractal_strength"],
            octaves=4,
            persistence=0.5,
            lowfreq_preservation=p["lowfreq_preservation"],
            lowfreq_factor=8,
            upscale_mode=upscale_mode,
            tile_seed_mode=tile_seed_mode,
            conditioning_mode="sdxl_tile_crop",
            local_sampler="ar_fusion",
            same_size_refine=True,
            source_latent=source_latent,
            guidance_heatmap=heatmap,
            guidance_mode=p["guidance_mode"],
            source_anchor_strength=p["source_anchor_strength"],
            heatmap_strength=p["heatmap_strength"],
            heatmap_gamma=heatmap_gamma,
            source_color_preserve=p["source_color_preserve"],
        )

        report = (
            f"{plan}\\n"
            f"AR Image Guided Hierarchical Sampler | preset={preset} | "
            f"global_denoise={p['global_denoise']:.3f} | "
            f"local_denoise={p['local_denoise']:.3f} | "
            f"guidance_mode={p['guidance_mode']} | "
            f"source_anchor={p['source_anchor_strength']:.3f} | "
            f"heatmap_strength={p['heatmap_strength']:.3f} | "
            f"source_color_preserve={p['source_color_preserve']:.3f}"
        )

        return latent, report
