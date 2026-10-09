# Raytracer

A Taichi-based GPU path tracer written in Python.

This repository is a compact but capable ray tracing playground built around Taichi and NumPy. It includes a BVH-accelerated path tracer, analytic and mesh-based primitives, several benchmark scenes, next-event estimation for area lights, and both offline and interactive rendering modes.

## What this repository contains

- `main.py` — primary application entrypoint. Builds scenes, creates a Taichi `Camera`, and renders images with a path tracer implemented in Taichi kernels.
- `run_tests.py` — batch renderer for the full test scene collection. It reuses a single Taichi session to avoid repeated JIT compilation.
- `scene_tests.py` — registry of all named test scenes, aliases, and helper listing logic.
- `scene_cornell.py` — reference Cornell box scene builder with walls, area lights, rounded boxes, glass, metal, and emissive surfaces.
- `scene_presets.py` — additional benchmark scenes such as a studio setup and a spheres gallery.
- `scene_complex.py` — high-detail complex scene builder.
- `scene_accel.py` — GPU-compatible primitive representations, BVH traversal, and analytic intersection routines.
- `materials.py` — material factory functions and ray scattering behavior for diffuse, metal, dielectric, and emissive materials.
- `mesh_io.py` — host-side OBJ/mesh loader and normalization utilities for triangle meshes.
- `Texture.py`, `Color.py`, `AABB.py`, `BVH.py`, `Objects.py`, `Ray.py`, `Rand.py`, `interval.py` — low-level renderer and math utilities.
- `models/` — optional asset folder for imported meshes.
- `output/` — default output folder for rendered PNG images.
- `tests/` — unit tests for BVH, math utilities, scene presets, and render correctness.

## Highlights

- GPU path tracing implemented with Taichi.
- BVH acceleration structure with multiple splitting strategies: `sah`, `median`, `gpu`, and `auto`.
- Next-event estimation (NEE) for area lights, including MIS weighting.
- Support for analytic primitives: spheres, quads, boxes, ellipsoids, triangles, and signed-distance rounded boxes.
- Mesh import support via `mesh_io.py` and `scene_cornell.py` mesh-based Cornell scenes.
- Interactive live rendering mode through Taichi UI.
- Offline high-quality rendering to PNG.

## Dependencies

This project expects a working Python environment with at least:

- Python 3.x
- `taichi`
- `numpy`

Install dependencies with pip:

```bash
pip install taichi numpy
```

> Note: `main.py` currently initializes Taichi with `arch=ti.cuda` by default. If you do not have CUDA available, change the `ti.init(...)` call in `main.py` to use another supported backend such as `ti.cpu`.

## Usage

### Render a single scene

```bash
python main.py --scene cornell --spp 256 --output output/cornell.png
```

### List all available scene names

```bash
python main.py --list-scenes
```

### Render all registered test scenes

```bash
python main.py --render-all-testscenes --width 1024 --spp 256
```

### Interactive live rendering

```bash
python main.py --scene studio --live --width 1280 --live-spp 8
```

### Sampling and render reports

```bash
python main.py --scene cornell_model --width 720 --spp 1024 --quality normal --seed 0 --output output/dragon.png --stats output/dragon.json
```

Explicit `--spp` values are respected in every quality mode. In `normal` and
`quick`, this is the maximum per pixel: adaptive sampling can finish quiet pixels
earlier. `final` and `manual` use the requested count for every pixel. If `--spp`
is omitted, the scene and quality preset choose the budget.

Scrambled Sobol sampling is the default. `--sampler random` selects a reproducible
random baseline, and `--seed` changes the sample sequence. The Sobol direction
numbers use the first 32 dimensions of the Joe-Kuo data, with no SciPy dependency.

Adaptive rendering starts with tiles and switches to a compact queue when the
unfinished pixels become sparse. `--adaptive-scheduler pixels` and
`--adaptive-scheduler tiles` keep either scheduler throughout for comparison.
Two consecutive noise checks must pass before a pixel finishes early. Reaching
the sample limit does not mean that the noise target was met.

The built-in filter uses stable surface guides and measured sample variance.
`normal` and `quick` enable it; use `--denoise` with `final` or `manual`. Filtering
is an approximation, so keep an unfiltered reference when evaluating fine detail.
The normal preset takes one light sample at each eligible bounce, through the
path limit, to reduce noise from late light hits. Explicit light and depth flags
override this preset.

`--stats` writes setup, tracing, filtering, copy and PNG-save timings with actual
sample counts. Setup and tracing can include Taichi kernel compilation or cache
loading. The renderer report excludes Python startup; the comparison tool measures
the full process separately. `--linear-output output/dragon.npz` optionally saves
raw radiance, filtered radiance and per-pixel sample counts.

```bash
python tools/compare_quality.py --scene cornell_model --width 128 --spp 128 --reference-spp 1024 --repeats 2 --display-exposure 1
```

This creates an independent high-sample reference and compares random sampling,
Sobol sampling and both adaptive schedulers. Results, images and logs are written
under `output/quality_comparison/`. Linear HDR and display errors are reported
separately. Compare repeated runs and matching error levels before drawing speed
conclusions; compilation and startup matter especially for small images.

Use `--benchmark-runs 3 --stats output/timing.json` on an offline render to repeat
it in the same session. The report separates each render, so later runs show the
cost after kernels are prepared. The first run and full process remain part of
the total time.

Sampling references: [PBRT](https://pbr-book.org/4ed/Sampling_and_Reconstruction/Sampling_and_Integration)
and [Joe-Kuo direction numbers](https://web.maths.unsw.edu.au/~fkuo/sobol/).

### Batch render test scenes with `run_tests.py`

```bash
python run_tests.py --width 1920 --spp 5000
```

## Common command-line options

- `--scene` — choose a named scene.
- `--list-scenes` — print the scene table and exit.
- `--render-all-testscenes` — export all registered scenes to `output/testscenes/`.
- `--output` — explicit PNG filepath.
- `--width` — image width in pixels.
- `--spp` — samples per pixel for offline rendering.
- `--no-nee` — disable next-event estimation.
- `--bvh-split` — BVH builder mode: `auto`, `sah`, `median`, or `gpu`.
- `--live` — run an interactive Taichi UI render.
- `--live-spp` — samples per pixel per frame in live mode.

## Available scene names

The repository exposes a broad scene registry. Some of the most useful names are:

- `cornell`
- `cornell_model`
- `classic_boxes`
- `glass_only`
- `metal_only`
- `checker_sphere`
- `material_spheres`
- `material_quads`
- `material_boxes`
- `reflection_refraction_boxes`
- `studio`
- `spheres`
- `complex`
- `model_showcase`

Other aliases and debug variants include:

- `wide_fov`, `narrow_fov`, `many_bounces`, `low_spp`, `high_spp`
- `glass_stress`, `metal_stress`, `checker_stress`
- `bounce_deep`, `reflection_refraction`, `default`, `default_ref`
- `studio_ref`, `spheres_ref`, `complex_ref`

If you want the exact list for this repository, use `python main.py --list-scenes`.

## Scene and rendering behavior

- `scene_cornell.py` builds a physically-inspired Cornell box with measured wall reflectances, area lighting, rounded blocks, and glass/metal objects.
- `scene_presets.py` provides a studio interior and a sphere gallery with procedural textures.
- `scene_complex.py` builds heavier scenes that can use the GPU BVH builder.
- `main.py` can render both offline and interactive outputs, and includes a diagnostic path tracer for special debug scenes.

## Output

- Standard renders are written to `output/<scene>.png` by default.
- Batch test renders are written to `output/testscenes/<scene>.png`.
- Live mode saves a final image as `output/live_final.png` when the window closes.

## Notes

- `models/` contains optional imported meshes for model-based scenes. The scene builders can load OBJ and other mesh formats when present.
- `scene_accel.py` implements low-level GPU primitive intersection and BVH traversal in Taichi.
- `materials.py` defines diffuse, specular, dielectric, and emissive materials, with checkerboard and procedural textures.
- The renderer uses ACES-style filmic tonemapping from `render_core.py`.

## Testing

Unit tests live under `tests/` and cover BVH, math helpers, scene presets, and smoke rendering. Run them with your Python test runner.

```bash
python -m unittest discover tests
```

## How to extend this project

If you want to add new scenes, create a builder that returns:

```python
(mats_list, bvh, root, cam)
```

and then add it to `scene_tests.py` or register it in `main.py`.

If you want to add new materials or textures, update `materials.py` and `Texture.py`.

Enjoy exploring GPU path tracing with Taichi!
