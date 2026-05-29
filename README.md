# ComfyUI Arbitrary Resolution Sampler

Experimental ComfyUI custom nodes for SDXL-style arbitrary-resolution generation and refinement.

The goal is not to force SDXL to render huge canvases directly. The node builds an image through a resolution pyramid: base render, latent upscale stages, fractal latent noise injection, optional global refinement, tiled local refinement, overlap blending, and low-frequency preservation.

## Nodes

- `Arbitrary Resolution Hierarchical Sampler`
  - main node for text-to-image and latent-to-latent arbitrary-resolution generation/refinement
- `Arbitrary Resolution Plan`
  - shows the planned resolution pyramid
- `Fractal Latent Noise`
  - standalone multiscale latent noise injection node

## Install

Clone or copy this folder into ComfyUI custom nodes:

```bash
git clone https://github.com/Gerbesh/ComfyUI-ArbitraryResSampler.git ComfyUI/custom_nodes/ComfyUI-ArbitraryResSampler
```

Restart ComfyUI.

## Basic workflow

Text-to-image:

```text
CheckpointLoader
CLIP Text Encode positive
CLIP Text Encode negative
Arbitrary Resolution Hierarchical Sampler
VAE Decode
Save Image
```

Image/latent refinement:

```text
Load Image
VAE Encode / VAE Encode Tiled
Arbitrary Resolution Hierarchical Sampler.source_latent
VAE Decode / VAE Decode Tiled
Save Image
```

## Recommended starting settings for 5000x2800

```text
target_width: 5000
target_height: 2800
base_pixels: 1024
max_scale_per_stage: 1.55
steps: 24-30
cfg: 5.5-6.5
global_denoise: 0.14-0.20
local_denoise: 0.22-0.30
global_max_megapixels: 2.5-3.0
tile_pixels: 1024
overlap_pixels: 224-256
fractal_strength: 0.05-0.09
octaves: 4
persistence: 0.45-0.55
lowfreq_preservation: 0.35-0.50
lowfreq_factor: 8
upscale_mode: bicubic
tile_seed_mode: offset
```

For low VRAM:

```text
tile_pixels: 768
overlap_pixels: 160-192
global_denoise: 0.10-0.16
local_denoise: 0.18-0.24
global_max_megapixels: 1.5-2.0
```

## How it works

1. Fits a base SDXL-like resolution to the target aspect ratio.
2. Builds a monotonic stage schedule toward the target size.
3. Renders or accepts a source latent.
4. Upscales latent stage by stage.
5. Injects multiscale fractal latent noise.
6. Runs a global refine only while the stage is below the configured megapixel limit.
7. Runs tiled local refinement with overlap blending.
8. Restores low-frequency structure to reduce form drift.

## Current limitations

- Batch size is currently limited to 1.
- The tiled refinement is tile-level sampling, not true per-step UNet prediction fusion yet.
- Very high target resolutions can be slow because each stage may contain many tile passes.
- This is experimental code. Keep denoise values conservative if preserving composition matters.

## Roadmap

- True per-step tiled UNet fusion sampler.
- Global low-resolution guide pass on every diffusion step.
- Adaptive tile sizing based on target resolution and VRAM profile.
- Region-aware CFG and prompt weighting.
- Better handling for masks and inpaint-style workflows.
- Optional tile diagnostics output.

## License

No open-source license is granted at this stage. All rights reserved unless a LICENSE file is added later.
