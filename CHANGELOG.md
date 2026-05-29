# Changelog

## 0.3.2

- Fixed same_size_refine not being passed into the hierarchical sampler function.
- Fixed ComfyUI runtime NameError when source latent already matches target resolution.

## 0.3.1

- Added same-size source latent refinement path.
- Arbitrary Resolution Hierarchical Sampler now runs local refine even when source_latent already matches 	arget_width / 	arget_height.
- Added same_size_refine toggle, enabled by default.

## 0.3.0

- Added halo-aware tiled fusion:
  - tiles are expanded with surrounding context before model inference
  - only the core area is merged back into the final prediction
- Added SDXL tile-aware conditioning helper:
  - full-frame width/height are preserved
  - crop_w / crop_h track the tile position inside the large frame
  - target_width / target_height track the tile viewport size
- Added `conditioning_mode`:
  - `plain`
  - `sdxl_tile_crop`
- Added `halo_pixels` control to both:
  - `AR Tiled Fusion Sampler`
  - `Arbitrary Resolution Hierarchical Sampler`
- Wired the custom `ar_fusion` path to use halo context + core-only merge + tile-aware conditioning.

## 0.2.0

- Added `AR Tiled Fusion Sampler`, a standalone custom sampler node.
- Added `AREulerTiledFusionSampler`, a Comfy-compatible sampler object passed through `comfy.sample.sample_custom(...)`.
- Added per-step tiled denoised prediction fusion with overlap feathering.
- Added low-resolution global context prediction for low-frequency structure guidance.
- Added `local_sampler` selector to `Arbitrary Resolution Hierarchical Sampler`:
  - `legacy` - previous tile-level KSampler refinement.
  - `ar_fusion` - new custom per-step tiled Euler fusion sampler.
- Kept the legacy path available as a fallback.

## 0.1.0

- Initial public package structure.
- Added arbitrary-resolution hierarchical sampler.
- Added optional `source_latent` input for latent/image refinement workflows.
- Added fractal latent noise injection.
- Added tiled local refinement with overlap feather blending.
- Added low-frequency preservation after tiled refinement.
- Added stage plan helper node.
- Added Python syntax-check GitHub Action.

