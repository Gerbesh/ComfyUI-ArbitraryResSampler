from __future__ import annotations

import comfy.samplers

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
                "local_denoise": ("FLOAT", {"default": 0.26, "min": 0.0, "max": 1.0, "step": 0.01}),
                "global_max_megapixels": ("FLOAT", {"default": 3.0, "min": 0.25, "max": 64.0, "step": 0.05}),
                "tile_pixels": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "overlap_pixels": ("INT", {"default": 256, "min": 0, "max": 2048, "step": 32}),
                "fractal_strength": ("FLOAT", {"default": 0.075, "min": 0.0, "max": 2.0, "step": 0.005}),
                "octaves": ("INT", {"default": 4, "min": 1, "max": 8}),
                "persistence": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 1.0, "step": 0.01}),
                "lowfreq_preservation": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01}),
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
        global_max_megapixels,
        tile_pixels,
        overlap_pixels,
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
            fractal_strength=fractal_strength,
            octaves=octaves,
            persistence=persistence,
            lowfreq_preservation=lowfreq_preservation,
            lowfreq_factor=lowfreq_factor,
            upscale_mode=upscale_mode,
            tile_seed_mode=tile_seed_mode,
            source_latent=source_latent,
        )
        return (latent, plan)


NODE_CLASS_MAPPINGS = {
    "FractalLatentNoise": FractalLatentNoise,
    "ArbitraryResolutionPlan": ArbitraryResolutionPlan,
    "ArbitraryResolutionHierarchicalSampler": ArbitraryResolutionHierarchicalSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FractalLatentNoise": "Fractal Latent Noise",
    "ArbitraryResolutionPlan": "Arbitrary Resolution Plan",
    "ArbitraryResolutionHierarchicalSampler": "Arbitrary Resolution Hierarchical Sampler",
}
