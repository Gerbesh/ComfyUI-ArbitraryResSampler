# Changelog

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
