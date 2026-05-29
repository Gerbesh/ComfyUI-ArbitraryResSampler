from __future__ import annotations

from typing import Tuple

import torch

from .constants import LATENT_SCALE_FACTOR
from .tiles import feather_mask, tile_positions


def _round_to_multiple(value: int, multiple: int = 8) -> int:
    value = int(value)
    multiple = max(1, int(multiple))
    return max(multiple, int(round(value / multiple)) * multiple)


def _floor_to_multiple(value: int, multiple: int = 8) -> int:
    value = int(value)
    multiple = max(1, int(multiple))
    return max(multiple, (value // multiple) * multiple)


def _normalize_image_params(tile_pixels: int, overlap_pixels: int, halo_pixels: int) -> Tuple[int, int, int]:
    tile_pixels = _round_to_multiple(tile_pixels, LATENT_SCALE_FACTOR)
    overlap_pixels = max(0, _round_to_multiple(overlap_pixels, LATENT_SCALE_FACTOR))
    halo_pixels = max(0, _round_to_multiple(halo_pixels, LATENT_SCALE_FACTOR))
    return tile_pixels, overlap_pixels, halo_pixels


def _crop_pixels_to_multiple_of_8(pixels: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    _, height, width, _ = pixels.shape
    cropped_height = (height // LATENT_SCALE_FACTOR) * LATENT_SCALE_FACTOR
    cropped_width = (width // LATENT_SCALE_FACTOR) * LATENT_SCALE_FACTOR

    cropped_height = max(LATENT_SCALE_FACTOR, cropped_height)
    cropped_width = max(LATENT_SCALE_FACTOR, cropped_width)

    if cropped_height == height and cropped_width == width:
        return pixels, height, width

    cropped = pixels[:, :cropped_height, :cropped_width, :]
    return cropped, cropped_height, cropped_width


def _image_feather_mask(height: int, width: int, overlap_y: int, overlap_x: int, device, dtype):
    y = torch.ones(height, device=device, dtype=dtype)
    x = torch.ones(width, device=device, dtype=dtype)

    if overlap_y > 0:
        ramp = torch.linspace(0.0, 1.0, steps=overlap_y + 2, device=device, dtype=dtype)[1:-1]
        y[:overlap_y] = torch.minimum(y[:overlap_y], ramp)
        y[-overlap_y:] = torch.minimum(y[-overlap_y:], torch.flip(ramp, dims=[0]))

    if overlap_x > 0:
        ramp = torch.linspace(0.0, 1.0, steps=overlap_x + 2, device=device, dtype=dtype)[1:-1]
        x[:overlap_x] = torch.minimum(x[:overlap_x], ramp)
        x[-overlap_x:] = torch.minimum(x[-overlap_x:], torch.flip(ramp, dims=[0]))

    mask = torch.outer(y, x).unsqueeze(0).unsqueeze(-1)
    return mask.clamp_min(1e-3)


def encode_tiled_pixels(
    vae,
    pixels: torch.Tensor,
    tile_pixels: int = 1536,
    overlap_pixels: int = 128,
    halo_pixels: int = 64,
    crop_to_multiple_of_8: bool = True,
):
    if pixels.shape[0] != 1:
        raise ValueError("AR VAE Encode Tiled currently supports batch_size=1 only.")

    if crop_to_multiple_of_8:
        pixels, _, _ = _crop_pixels_to_multiple_of_8(pixels)

    batch, full_height, full_width, _ = pixels.shape
    if full_height < LATENT_SCALE_FACTOR or full_width < LATENT_SCALE_FACTOR:
        raise ValueError("Input image is too small for VAE encoding.")

    tile_pixels, overlap_pixels, halo_pixels = _normalize_image_params(tile_pixels, overlap_pixels, halo_pixels)

    core_width = min(tile_pixels, full_width)
    core_height = min(tile_pixels, full_height)
    overlap_width = max(0, min(overlap_pixels, max(0, core_width - LATENT_SCALE_FACTOR)))
    overlap_height = max(0, min(overlap_pixels, max(0, core_height - LATENT_SCALE_FACTOR)))

    xs = tile_positions(full_width, core_width, overlap_width)
    ys = tile_positions(full_height, core_height, overlap_height)

    latent_height = full_height // LATENT_SCALE_FACTOR
    latent_width = full_width // LATENT_SCALE_FACTOR

    accumulator = None
    weights = None

    for core_y in ys:
        for core_x in xs:
            current_core_height = min(core_height, full_height - core_y)
            current_core_width = min(core_width, full_width - core_x)

            expanded_y0 = max(0, core_y - halo_pixels)
            expanded_x0 = max(0, core_x - halo_pixels)
            expanded_y1 = min(full_height, core_y + current_core_height + halo_pixels)
            expanded_x1 = min(full_width, core_x + current_core_width + halo_pixels)

            expanded_y0 = (expanded_y0 // LATENT_SCALE_FACTOR) * LATENT_SCALE_FACTOR
            expanded_x0 = (expanded_x0 // LATENT_SCALE_FACTOR) * LATENT_SCALE_FACTOR
            expanded_y1 = (expanded_y1 // LATENT_SCALE_FACTOR) * LATENT_SCALE_FACTOR
            expanded_x1 = (expanded_x1 // LATENT_SCALE_FACTOR) * LATENT_SCALE_FACTOR

            expanded_y1 = max(expanded_y0 + LATENT_SCALE_FACTOR, expanded_y1)
            expanded_x1 = max(expanded_x0 + LATENT_SCALE_FACTOR, expanded_x1)

            tile_pixels_tensor = pixels[:, expanded_y0:expanded_y1, expanded_x0:expanded_x1, :3].contiguous()

            with torch.no_grad():
                encoded = vae.encode(tile_pixels_tensor)

            if accumulator is None:
                accumulator = torch.zeros(
                    (batch, encoded.shape[1], latent_height, latent_width),
                    device=encoded.device,
                    dtype=encoded.dtype,
                )
                weights = torch.zeros(
                    (1, 1, latent_height, latent_width),
                    device=encoded.device,
                    dtype=encoded.dtype,
                )

            core_rel_y = (core_y - expanded_y0) // LATENT_SCALE_FACTOR
            core_rel_x = (core_x - expanded_x0) // LATENT_SCALE_FACTOR
            current_core_latent_height = current_core_height // LATENT_SCALE_FACTOR
            current_core_latent_width = current_core_width // LATENT_SCALE_FACTOR

            encoded_core = encoded[
                :,
                :,
                core_rel_y:core_rel_y + current_core_latent_height,
                core_rel_x:core_rel_x + current_core_latent_width,
            ]

            overlap_latent_height = min(overlap_height // LATENT_SCALE_FACTOR, max(0, current_core_latent_height - 1))
            overlap_latent_width = min(overlap_width // LATENT_SCALE_FACTOR, max(0, current_core_latent_width - 1))

            weight = feather_mask(
                current_core_latent_height,
                current_core_latent_width,
                overlap_latent_height,
                overlap_latent_width,
                device=encoded_core.device,
                dtype=encoded_core.dtype,
            )

            latent_y0 = core_y // LATENT_SCALE_FACTOR
            latent_x0 = core_x // LATENT_SCALE_FACTOR
            latent_y1 = latent_y0 + current_core_latent_height
            latent_x1 = latent_x0 + current_core_latent_width

            accumulator[:, :, latent_y0:latent_y1, latent_x0:latent_x1] += encoded_core * weight
            weights[:, :, latent_y0:latent_y1, latent_x0:latent_x1] += weight

    samples = accumulator / weights.clamp_min(1e-6)

    info = (
        f"AR VAE Encode Tiled | image={full_width}x{full_height} | "
        f"latent={latent_width}x{latent_height} | tile={tile_pixels} | "
        f"overlap={overlap_pixels} | halo={halo_pixels}"
    )
    return {"samples": samples}, info


def decode_tiled_latent(
    vae,
    latent: dict,
    tile_pixels: int = 1536,
    overlap_pixels: int = 128,
    halo_pixels: int = 64,
):
    samples = latent["samples"]
    if samples.shape[0] != 1:
        raise ValueError("AR VAE Decode Tiled currently supports batch_size=1 only.")

    batch, _, latent_height, latent_width = samples.shape
    full_height = latent_height * LATENT_SCALE_FACTOR
    full_width = latent_width * LATENT_SCALE_FACTOR

    tile_pixels, overlap_pixels, halo_pixels = _normalize_image_params(tile_pixels, overlap_pixels, halo_pixels)

    core_latent_width = min(tile_pixels // LATENT_SCALE_FACTOR, latent_width)
    core_latent_height = min(tile_pixels // LATENT_SCALE_FACTOR, latent_height)
    overlap_latent_width = max(0, min(overlap_pixels // LATENT_SCALE_FACTOR, max(0, core_latent_width - 1)))
    overlap_latent_height = max(0, min(overlap_pixels // LATENT_SCALE_FACTOR, max(0, core_latent_height - 1)))
    halo_latent = max(0, halo_pixels // LATENT_SCALE_FACTOR)

    xs = tile_positions(latent_width, core_latent_width, overlap_latent_width)
    ys = tile_positions(latent_height, core_latent_height, overlap_latent_height)

    accumulator = None
    weights = None

    for core_y in ys:
        for core_x in xs:
            current_core_latent_height = min(core_latent_height, latent_height - core_y)
            current_core_latent_width = min(core_latent_width, latent_width - core_x)

            expanded_y0 = max(0, core_y - halo_latent)
            expanded_x0 = max(0, core_x - halo_latent)
            expanded_y1 = min(latent_height, core_y + current_core_latent_height + halo_latent)
            expanded_x1 = min(latent_width, core_x + current_core_latent_width + halo_latent)

            tile_latent = samples[:, :, expanded_y0:expanded_y1, expanded_x0:expanded_x1].contiguous()

            with torch.no_grad():
                decoded = vae.decode(tile_latent)

            if accumulator is None:
                accumulator = torch.zeros(
                    (batch, full_height, full_width, decoded.shape[-1]),
                    device=decoded.device,
                    dtype=decoded.dtype,
                )
                weights = torch.zeros(
                    (1, full_height, full_width, 1),
                    device=decoded.device,
                    dtype=decoded.dtype,
                )

            core_rel_y_px = (core_y - expanded_y0) * LATENT_SCALE_FACTOR
            core_rel_x_px = (core_x - expanded_x0) * LATENT_SCALE_FACTOR
            current_core_height_px = current_core_latent_height * LATENT_SCALE_FACTOR
            current_core_width_px = current_core_latent_width * LATENT_SCALE_FACTOR

            decoded_core = decoded[
                :,
                core_rel_y_px:core_rel_y_px + current_core_height_px,
                core_rel_x_px:core_rel_x_px + current_core_width_px,
                :
            ]

            overlap_height_px = min(overlap_pixels, max(0, current_core_height_px - LATENT_SCALE_FACTOR))
            overlap_width_px = min(overlap_pixels, max(0, current_core_width_px - LATENT_SCALE_FACTOR))

            weight = _image_feather_mask(
                current_core_height_px,
                current_core_width_px,
                overlap_height_px,
                overlap_width_px,
                device=decoded_core.device,
                dtype=decoded_core.dtype,
            )

            image_y0 = core_y * LATENT_SCALE_FACTOR
            image_x0 = core_x * LATENT_SCALE_FACTOR
            image_y1 = image_y0 + current_core_height_px
            image_x1 = image_x0 + current_core_width_px

            accumulator[:, image_y0:image_y1, image_x0:image_x1, :] += decoded_core * weight
            weights[:, image_y0:image_y1, image_x0:image_x1, :] += weight

    pixels = accumulator / weights.clamp_min(1e-6)

    info = (
        f"AR VAE Decode Tiled | image={full_width}x{full_height} | "
        f"latent={latent_width}x{latent_height} | tile={tile_pixels} | "
        f"overlap={overlap_pixels} | halo={halo_pixels}"
    )
    return pixels, info


class ARVAEEncodeTiled:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pixels": ("IMAGE",),
                "vae": ("VAE",),
                "tile_pixels": ("INT", {"default": 1536, "min": 256, "max": 4096, "step": 64}),
                "overlap_pixels": ("INT", {"default": 128, "min": 0, "max": 2048, "step": 32}),
                "halo_pixels": ("INT", {"default": 64, "min": 0, "max": 1024, "step": 32}),
                "crop_to_multiple_of_8": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "info")
    FUNCTION = "run"
    CATEGORY = "vae/arbitrary_res"

    def run(self, pixels, vae, tile_pixels, overlap_pixels, halo_pixels, crop_to_multiple_of_8):
        latent, info = encode_tiled_pixels(
            vae=vae,
            pixels=pixels,
            tile_pixels=tile_pixels,
            overlap_pixels=overlap_pixels,
            halo_pixels=halo_pixels,
            crop_to_multiple_of_8=bool(crop_to_multiple_of_8),
        )
        return latent, info


class ARVAEDecodeTiled:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "vae": ("VAE",),
                "tile_pixels": ("INT", {"default": 1536, "min": 256, "max": 4096, "step": 64}),
                "overlap_pixels": ("INT", {"default": 128, "min": 0, "max": 2048, "step": 32}),
                "halo_pixels": ("INT", {"default": 64, "min": 0, "max": 1024, "step": 32}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("pixels", "info")
    FUNCTION = "run"
    CATEGORY = "vae/arbitrary_res"

    def run(self, latent, vae, tile_pixels, overlap_pixels, halo_pixels):
        pixels, info = decode_tiled_latent(
            vae=vae,
            latent=latent,
            tile_pixels=tile_pixels,
            overlap_pixels=overlap_pixels,
            halo_pixels=halo_pixels,
        )
        return pixels, info


class ARVAERoundtrip:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pixels": ("IMAGE",),
                "vae": ("VAE",),
                "tile_pixels": ("INT", {"default": 1536, "min": 256, "max": 4096, "step": 64}),
                "overlap_pixels": ("INT", {"default": 128, "min": 0, "max": 2048, "step": 32}),
                "halo_pixels": ("INT", {"default": 64, "min": 0, "max": 1024, "step": 32}),
                "crop_to_multiple_of_8": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "LATENT", "STRING")
    RETURN_NAMES = ("pixels", "latent", "report")
    FUNCTION = "run"
    CATEGORY = "vae/arbitrary_res"

    def run(self, pixels, vae, tile_pixels, overlap_pixels, halo_pixels, crop_to_multiple_of_8):
        input_pixels = pixels
        if crop_to_multiple_of_8:
            input_pixels, _, _ = _crop_pixels_to_multiple_of_8(input_pixels)

        latent, encode_info = encode_tiled_pixels(
            vae=vae,
            pixels=input_pixels,
            tile_pixels=tile_pixels,
            overlap_pixels=overlap_pixels,
            halo_pixels=halo_pixels,
            crop_to_multiple_of_8=False,
        )
        decoded, decode_info = decode_tiled_latent(
            vae=vae,
            latent=latent,
            tile_pixels=tile_pixels,
            overlap_pixels=overlap_pixels,
            halo_pixels=halo_pixels,
        )

        diff = (input_pixels[..., :decoded.shape[-1]] - decoded).abs()
        mae = float(diff.mean().item())
        max_err = float(diff.max().item())

        report = (
            f"{encode_info} | {decode_info} | "
            f"roundtrip_mae={mae:.6f} | roundtrip_max={max_err:.6f}"
        )
        return decoded, latent, report