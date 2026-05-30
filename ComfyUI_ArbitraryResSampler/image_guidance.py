from __future__ import annotations

import torch
import torch.nn.functional as F


def _luminance(image: torch.Tensor) -> torch.Tensor:
    r = image[..., 0:1]
    g = image[..., 1:2]
    b = image[..., 2:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _to_nchw_gray(gray_bhwc: torch.Tensor) -> torch.Tensor:
    return gray_bhwc.permute(0, 3, 1, 2).contiguous()


def _to_bhwc(gray_nchw: torch.Tensor) -> torch.Tensor:
    return gray_nchw.permute(0, 2, 3, 1).contiguous()


def _normalize_map(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dims = tuple(range(1, x.ndim))
    mn = x.amin(dim=dims, keepdim=True)
    mx = x.amax(dim=dims, keepdim=True)
    return ((x - mn) / (mx - mn + eps)).clamp(0.0, 1.0)


def _box_blur_nchw(x: torch.Tensor, radius: int) -> torch.Tensor:
    radius = max(0, int(radius))
    if radius <= 0:
        return x
    kernel = radius * 2 + 1
    return F.avg_pool2d(x, kernel_size=kernel, stride=1, padding=radius)


def _sobel_edges(gray_nchw: torch.Tensor) -> torch.Tensor:
    device = gray_nchw.device
    dtype = gray_nchw.dtype

    kx = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)

    ky = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)

    gx = F.conv2d(gray_nchw, kx, padding=1)
    gy = F.conv2d(gray_nchw, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-8)


def build_image_heatmap(
    image: torch.Tensor,
    detail_weight: float = 0.55,
    softness_weight: float = 0.35,
    contrast_weight: float = 0.10,
    blur_radius: int = 5,
    smooth_radius: int = 9,
    gamma: float = 1.0,
    invert: bool = False,
) -> torch.Tensor:
    if image.ndim != 4:
        raise ValueError("Expected IMAGE tensor in BHWC format.")

    image = image[..., :3].clamp(0.0, 1.0)
    gray = _to_nchw_gray(_luminance(image))

    blur_radius = max(1, int(blur_radius))
    smooth_radius = max(0, int(smooth_radius))

    blurred = _box_blur_nchw(gray, blur_radius)
    high = (gray - blurred).abs()

    edges = _sobel_edges(gray)
    local_mean = _box_blur_nchw(gray, blur_radius)
    local_var = _box_blur_nchw((gray - local_mean) ** 2, blur_radius)

    detail = _normalize_map(edges + high)
    softness = 1.0 - _normalize_map(local_var)
    contrast = _normalize_map((gray - local_mean).abs())

    heat = (
        float(detail_weight) * detail
        + float(softness_weight) * softness
        + float(contrast_weight) * contrast
    )

    heat = _normalize_map(heat)

    if smooth_radius > 0:
        heat = _box_blur_nchw(heat, smooth_radius)
        heat = _normalize_map(heat)

    gamma = max(0.05, float(gamma))
    heat = heat.clamp(0.0, 1.0) ** gamma

    if invert:
        heat = 1.0 - heat

    return _to_bhwc(heat).clamp(0.0, 1.0)


def heatmap_to_preview(heatmap: torch.Tensor) -> torch.Tensor:
    heat = heatmap[..., :1].clamp(0.0, 1.0)

    # Simple black -> blue -> yellow -> red ramp, no external deps.
    r = torch.clamp((heat - 0.45) / 0.55, 0.0, 1.0)
    g = torch.clamp(1.0 - torch.abs(heat - 0.55) / 0.45, 0.0, 1.0)
    b = torch.clamp((0.55 - heat) / 0.55, 0.0, 1.0)

    return torch.cat([r, g, b], dim=-1).clamp(0.0, 1.0)


def resize_heatmap_to_latent(heatmap: torch.Tensor, latent_samples: torch.Tensor) -> torch.Tensor:
    if heatmap.ndim != 4:
        raise ValueError("Expected heatmap IMAGE tensor in BHWC format.")

    heat = heatmap[..., :1].permute(0, 3, 1, 2).contiguous()
    _, _, latent_h, latent_w = latent_samples.shape
    heat = F.interpolate(heat, size=(latent_h, latent_w), mode="bilinear", align_corners=False)
    return heat.clamp(0.0, 1.0)


def guided_latent_blend(
    source_latent: dict,
    refined_latent: dict,
    heatmap: torch.Tensor,
    base_strength: float = 0.12,
    heat_strength: float = 0.35,
    color_preserve: float = 0.65,
    gamma: float = 1.0,
) -> tuple[dict, str]:
    source = source_latent["samples"]
    refined = refined_latent["samples"]

    if source.shape != refined.shape:
        raise ValueError(f"Latent shapes do not match: source={tuple(source.shape)} refined={tuple(refined.shape)}")

    heat = resize_heatmap_to_latent(heatmap, source)
    gamma = max(0.05, float(gamma))
    heat = heat.clamp(0.0, 1.0) ** gamma

    strength = (float(base_strength) + heat * float(heat_strength)).clamp(0.0, 1.0)

    blended = source * (1.0 - strength) + refined * strength

    if color_preserve > 0.0:
        # Latent-space low-frequency color/tonal lock.
        factor = 8
        low_h = max(1, source.shape[-2] // factor)
        low_w = max(1, source.shape[-1] // factor)

        source_low = F.interpolate(source, size=(low_h, low_w), mode="area")
        blended_low = F.interpolate(blended, size=(low_h, low_w), mode="area")
        source_low = F.interpolate(source_low, size=source.shape[-2:], mode="bilinear", align_corners=False)
        blended_low = F.interpolate(blended_low, size=source.shape[-2:], mode="bilinear", align_corners=False)

        blended = blended + (source_low - blended_low) * float(color_preserve)

    out = refined_latent.copy()
    out["samples"] = blended

    info = (
        f"AR Guided Latent Blend | base_strength={float(base_strength):.3f} | "
        f"heat_strength={float(heat_strength):.3f} | color_preserve={float(color_preserve):.3f} | "
        f"heat_mean={float(heat.mean().item()):.4f}"
    )

    return out, info


class ARImageHeatmap:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "preset": (["balanced", "detail_recovery", "anti_blotch", "soft_regions", "edges_only"], {"default": "balanced"}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 4.0, "step": 0.05}),
                "blur_radius": ("INT", {"default": 5, "min": 1, "max": 31, "step": 2}),
                "smooth_radius": ("INT", {"default": 9, "min": 0, "max": 63, "step": 2}),
                "invert": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("heatmap", "preview", "report")
    FUNCTION = "run"
    CATEGORY = "guidance/arbitrary_res"

    def run(self, image, preset, strength, gamma, blur_radius, smooth_radius, invert):
        if preset == "detail_recovery":
            weights = (0.70, 0.20, 0.10)
        elif preset == "anti_blotch":
            weights = (0.25, 0.65, 0.10)
        elif preset == "soft_regions":
            weights = (0.15, 0.80, 0.05)
        elif preset == "edges_only":
            weights = (1.0, 0.0, 0.0)
        else:
            weights = (0.55, 0.35, 0.10)

        heat = build_image_heatmap(
            image=image,
            detail_weight=weights[0] * float(strength),
            softness_weight=weights[1] * float(strength),
            contrast_weight=weights[2] * float(strength),
            blur_radius=blur_radius,
            smooth_radius=smooth_radius,
            gamma=gamma,
            invert=bool(invert),
        )
        preview = heatmap_to_preview(heat)
        report = (
            f"AR Image Heatmap | preset={preset} | mean={float(heat.mean().item()):.4f} | "
            f"min={float(heat.min().item()):.4f} | max={float(heat.max().item()):.4f}"
        )
        return heat, preview, report


class ARHeatmapPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "heatmap": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("preview",)
    FUNCTION = "run"
    CATEGORY = "guidance/arbitrary_res"

    def run(self, heatmap):
        return (heatmap_to_preview(heatmap),)


class ARGuidedLatentBlend:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_latent": ("LATENT",),
                "refined_latent": ("LATENT",),
                "heatmap": ("IMAGE",),
                "base_strength": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 1.0, "step": 0.01}),
                "heat_strength": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "color_preserve": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 4.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "report")
    FUNCTION = "run"
    CATEGORY = "guidance/arbitrary_res"

    def run(self, source_latent, refined_latent, heatmap, base_strength, heat_strength, color_preserve, gamma):
        return guided_latent_blend(
            source_latent=source_latent,
            refined_latent=refined_latent,
            heatmap=heatmap,
            base_strength=base_strength,
            heat_strength=heat_strength,
            color_preserve=color_preserve,
            gamma=gamma,
        )