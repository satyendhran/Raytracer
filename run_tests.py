"""
Render all 30 test scenes in a SINGLE process to avoid repeated Taichi JIT
compilation (saves ~70s per scene after the first one).

Usage:  python run_tests.py
        python run_tests.py --width 800 --spp 200     # faster preview
        python run_tests.py --width 1920 --spp 5000    # production quality
"""

import argparse
import os
import sys
from time import perf_counter as pf

def main():
    parser = argparse.ArgumentParser(description="Batch render all test scenes.")
    parser.add_argument("--width", type=int, default=1920, help="Image width")
    parser.add_argument("--spp", type=int, default=5000, help="Samples per pixel")
    parser.add_argument(
        "--scene",
        default="all",
        help="Render one named test scene, or 'all' for the full batch.",
    )
    parser.add_argument(
        "--bvh-split",
        choices=("auto", "sah", "median", "gpu"),
        default="auto",
        help="BVH split strategy used by scene builders.",
    )
    args = parser.parse_args()

    import taichi as ti

    import numpy as np
    import taichi.math as tm
    from scene_tests import ALL_TEST_SCENE_NAMES, TEST_SCENE_BUILDERS

    from Color import float_to_rgb8
    from interval import Interval
    from materials import Material
    from Objects import hit_record
    from Rand import random_in_unit_disk
    from Ray import Ray
    from render_core import sanitize_rgb, tonemap_filmic

    vec3 = tm.vec3
    point3 = tm.vec3

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "testscenes")
    os.makedirs(out_dir, exist_ok=True)

    # Import Camera class (it's defined in main.py, so we need to grab it)
    # Instead we'll import main and use its Camera + helpers
    import main as M

    total_t0 = pf()
    scene_names = ALL_TEST_SCENE_NAMES
    if args.scene != "all":
        if args.scene not in TEST_SCENE_BUILDERS:
            print(f"Unknown scene '{args.scene}'.")
            return
        scene_names = [args.scene]

    built_scenes = []
    for idx, name in enumerate(scene_names):
        out_path = os.path.join(out_dir, f"{name}.png")
        print(f"\n[{idx+1}/{len(scene_names)}] Building '{name}' ...")

        builder = TEST_SCENE_BUILDERS.get(name)
        if builder is None:
            print(f"  SKIP — no builder for '{name}'")
            continue

        try:
            split = args.bvh_split
            if split == "auto":
                split = "gpu" if name in ("complex", "cornell_model", "model_showcase") else "sah"
            mats_list, bvh, root, cam = builder(split)
        except Exception as e:
            print(f"  ERROR building scene: {e}")
            continue

        built_scenes.append((idx, name, out_path, mats_list, bvh, root, cam))

    if not built_scenes:
        print("No scenes built.")
        return

    for idx, name, out_path, mats_list, bvh, root, cam in built_scenes:
        print(f"\n[{idx+1}/{len(scene_names)}] Rendering '{name}' @ {args.width}x{args.spp}spp ...")

        M.materials = Material.field(shape=len(mats_list))
        for i, m in enumerate(mats_list):
            M.materials[i] = m

        lookfrom = cam["lookfrom"]
        lookat = cam["lookat"]
        vup = cam["vup"]
        vfov = cam["vfov"]
        aspect_ratio = float(cam["aspect_ratio"])
        dark_bg = int(cam.get("dark_background", 0))
        max_bounces = int(cam.get("max_bounces", 12))

        lf = np.asarray([lookfrom[0], lookfrom[1], lookfrom[2]], dtype=np.float64)
        la = np.asarray([lookat[0], lookat[1], lookat[2]], dtype=np.float64)
        focus_distance = float(np.linalg.norm(lf - la))

        camera = M.Camera(
            bvh,
            image_width=args.width,
            aspect_ratio=aspect_ratio,
            vfov=vfov,
            lookfrom=lookfrom,
            lookat=lookat,
            vup=vup,
            defocus_angle=0.0,
            focus_distance=focus_distance,
            use_hdr_accum=False,
            dark_background=dark_bg,
            max_bounces=max_bounces,
        )
        camera.configure_nee(cam.get("nee_lights", []), True)
        camera.samples_per_pixel = args.spp
        camera.pixel_sample_scale = 1.0 / args.spp
        camera.exposure[None] = float(cam.get("exposure", 0.85))
        camera.light_samples[None] = int(cam.get("light_samples", 4))

        t0 = pf()
        camera.render()
        ti.sync()
        t1 = pf()

        ti.tools.imwrite(camera.img.to_numpy(), out_path)
        print(f"  Saved {out_path}  ({t1-t0:.1f}s)")

    total_t1 = pf()
    print(f"\nRendered {len(built_scenes)} scene(s) in {total_t1-total_t0:.1f}s total.")

if __name__ == "__main__":
    main()
