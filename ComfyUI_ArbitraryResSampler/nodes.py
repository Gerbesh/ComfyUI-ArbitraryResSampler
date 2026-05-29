from __future__ import annotations

import comfy.samplers

from .ar_sampler import sample_latent_ar_fusion
from .noise import inject_fractal_noise
from .resolution import build_stage_schedule, fit_base_resolution, format_stage_plan, normalize_resolution
from .sampling import hierarchical_sample


class FractalLatentNoise:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "strength": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 5.0, "step": 0.01}),
                "octaves": ("INT", {"default": 4, "min": 1, "max": 8}),
                "persistence": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "apply"
    CATEGORY = "latent/arbitrary_res"

    def apply(self, latent, seed, strength, octaves, persistence):
        return (inject_fractal_noise(latent, seed, strength, octaves, persistence),)


class ArbitraryResolutionPlan:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_width": ("INT", {"default": 5000, "min": 64, "max": 65535}),
                "target_height": ("INT", {"default": 2800, "min": 64, "max": 65535}),
                "base_pixels": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "max_scale_per_stage": ("FLOAT", {"default": 1.6, "min": 1.1, "max": 3.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "INT")
    RETURN_NAMES = ("plan", "base_width", "base_height")
    FUNCTION = "plan"
    CATEGORY = "latent/arbitrary_res"

    def plan(self, target_width, target_height, base_pixels, max_scale_per_stage):
        target_width, target_height = normalize_resolution(target_width, target_height)
        base_width, base_height = fit_base_resolution(target_width, target_height, base_pixels)
        stages = build_stage_schedule(base_width, base_height, target_width, target_height, max_scale_per_stage)
        return (format_stage_plan(stages), base_width, base_height)


class ARTiledFusionSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
                "steps": ("INT", {"default": 24, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 0.28, "min": 0.0, "max": 1.0, "step": 0.01}),
                "tile_pixels": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "overlap_pixels": ("INT", {"default": 256, "min": 0, "max": 2048, "step": 32}),
                "halo_pixels": ("INT", {"default": 192, "min": 0, "max": 2048, "step": 32}),
                "tile_mode": (["auto", "always", "off"], {"default": "auto"}),
                "tile_threshold_pixels": ("INT", {"default": 1536, "min": 256, "max": 8192, "step": 64}),
                "conditioning_mode": (["plain", "sdxl_tile_crop"], {"default": "sdxl_tile_crop"}),
                "lowfreq_preservation": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "lowfreq_factor": ("INT", {"default": 8, "min": 2, "max": 64}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "run"
    CATEGORY = "sampling/arbitrary_res"

    def run(
        self,
        model,
        positive,
        negative,
        latent,
        seed,
        steps,
        cfg,
        scheduler,
        denoise,
        tile_pixels,
        overlap_pixels,
        halo_pixels,
        tile_mode,
        tile_threshold_pixels,
        conditioning_mode,
        lowfreq_preservation,
        lowfreq_factor,
    ):
        return (
            sample_latent_ar_fusion(
                model=model,
                seed=seed,
                steps=steps,
                cfg=cfg,
                scheduler=scheduler,
                positive=positive,
                negative=negative,
                latent=latent,
                denoise=denoise,
                tile_pixels=tile_pixels,
                overlap_pixels=overlap_pixels,
                halo_pixels=halo_pixels,
                lowfreq_preservation=lowfreq_preservation,
                lowfreq_factor=lowfreq_factor,
                conditioning_mode=conditioning_mode,
                tile_mode=tile_mode,
                tile_threshold_pixels=tile_threshold_pixels,
                disable_noise=False,
            ),
        )


class ArbitraryResolutionHierarchicalSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
                "steps": ("INT", {"default": 28, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "target_width": ("INT", {"default": 5000, "min": 64, "max": 65535}),
                "target_height": ("INT", {"default": 2800, "min": 64, "max": 65535}),
                "base_pixels": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "max_scale_per_stage": ("FLOAT", {"default": 1.55, "min": 1.1, "max": 3.0, "step": 0.05}),
                "global_denoise": ("FLOAT", {"default": 0.18, "min": 0.0, "max": 1.0, "step": 0.01}),
                "local_denoise": ("FLOAT", {"default": 0.24, "min": 0.0, "max": 1.0, "step": 0.01}),
                "local_sampler": (["legacy", "ar_fusion"], {"default": "ar_fusion"}),
                "same_size_refine": ("BOOLEAN", {"default": True}),
                "global_max_megapixels": ("FLOAT", {"default": 3.0, "min": 0.25, "max": 64.0, "step": 0.05}),
                "tile_pixels": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "overlap_pixels": ("INT", {"default": 256, "min": 0, "max": 2048, "step": 32}),
                "halo_pixels": ("INT", {"default": 192, "min": 0, "max": 2048, "step": 32}),
                "conditioning_mode": (["plain", "sdxl_tile_crop"], {"default": "sdxl_tile_crop"}),
                "fractal_strength": ("FLOAT", {"default": 0.075, "min": 0.0, "max": 2.0, "step": 0.005}),
                "octaves": ("INT", {"default": 4, "min": 1, "max": 8}),
                "persistence": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 1.0, "step": 0.01}),
                "lowfreq_preservation": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "lowfreq_factor": ("INT", {"default": 8, "min": 2, "max": 64}),
                "upscale_mode": (["bicubic", "bilinear", "nearest"], {"default": "bicubic"}),
                "tile_seed_mode": (["offset", "shared"], {"default": "offset"}),
            },
            "optional": {
                "source_latent": ("LATENT",),
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
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        target_width,
        target_height,
        base_pixels,
        max_scale_per_stage,
        global_denoise,
        local_denoise,
        local_sampler,
        global_max_megapixels,
        tile_pixels,
        overlap_pixels,
        halo_pixels,
        conditioning_mode,
        fractal_strength,
        octaves,
        persistence,
        lowfreq_preservation,
        lowfreq_factor,
        upscale_mode,
        tile_seed_mode,
        source_latent=None,
    ):
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
            base_pixels=base_pixels,
            max_scale_per_stage=max_scale_per_stage,
            global_denoise=global_denoise,
            local_denoise=local_denoise,
            global_max_megapixels=global_max_megapixels,
            tile_pixels=tile_pixels,
            overlap_pixels=overlap_pixels,
            halo_pixels=halo_pixels,
            fractal_strength=fractal_strength,
            octaves=octaves,
            persistence=persistence,
            lowfreq_preservation=lowfreq_preservation,
            lowfreq_factor=lowfreq_factor,
            upscale_mode=upscale_mode,
            tile_seed_mode=tile_seed_mode,
            conditioning_mode=conditioning_mode,
            local_sampler=local_sampler,
            same_size_refine=same_size_refine,
            source_latent=source_latent,
        )
        return (latent, plan)


NODE_CLASS_MAPPINGS = {
    "FractalLatentNoise": FractalLatentNoise,
    "ArbitraryResolutionPlan": ArbitraryResolutionPlan,
    "ARTiledFusionSampler": ARTiledFusionSampler,
    "ArbitraryResolutionHierarchicalSampler": ArbitraryResolutionHierarchicalSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FractalLatentNoise": "Fractal Latent Noise",
    "ArbitraryResolutionPlan": "Arbitrary Resolution Plan",
    "ARTiledFusionSampler": "AR Tiled Fusion Sampler",
    "ArbitraryResolutionHierarchicalSampler": "Arbitrary Resolution Hierarchical Sampler",
}


