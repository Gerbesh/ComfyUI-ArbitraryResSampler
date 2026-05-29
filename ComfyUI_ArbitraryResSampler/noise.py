from __future__ import annotations

import torch
import torch.nn.functional as F


def make_fractal_noise_like(
    shape,
    seed: int,
    octaves: int,
    persistence: float,
    device,
    dtype,
):
    batch, channels, height, width = shape
    device = torch.device(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))

    noise = torch.zeros(shape, device=device, dtype=dtype)
    amp_sum = 0.0
    octaves = max(1, int(octaves))

    for octave in range(octaves):
        scale = 2 ** (octaves - octave - 1)
        noise_h = max(1, height // scale)
        noise_w = max(1, width // scale)

        octave_noise = torch.randn(
            (batch, channels, noise_h, noise_w),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        octave_noise = F.interpolate(
            octave_noise,
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        )

        amp = float(persistence) ** octave
        noise = noise + octave_noise * amp
        amp_sum += amp

    noise = noise / max(amp_sum, 1e-8)
    mean = noise.mean(dim=(2, 3), keepdim=True)
    std = noise.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    return (noise - mean) / std


def inject_fractal_noise(latent: dict, seed: int, strength: float, octaves: int, persistence: float) -> dict:
    if strength <= 0.0:
        return latent

    output = latent.copy()
    samples = output["samples"].clone()
    noise = make_fractal_noise_like(
        samples.shape,
        seed=seed,
        octaves=octaves,
        persistence=persistence,
        device=samples.device,
        dtype=samples.dtype,
    )
    latent_std = samples.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    output["samples"] = samples + noise * latent_std * float(strength)
    return output
