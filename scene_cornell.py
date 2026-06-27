"""
Reference Cornell box: measured wall reflectances, rounded blocks, grounded objects.
"""

import os

import taichi as ti
import taichi.math as tm

import debug_tools
from materials import (
    create_dielectric,
    create_diffuse_light,
    create_lambertian,
    create_metal,
    create_solid_texture,
    create_uv_checker_texture,
)
from mesh_io import load_mesh_auto, select_best_model
from scene_presets import quad_light
from scene_accel import (
    PrimBVH,
    ScenePrimitive,
    scene_primitive_quad,
    scene_primitive_sphere,
    scene_primitive_triangle,
    scene_primitive_sdf_rounded_box,
)

vec3 = tm.vec3
point3 = tm.vec3

# Standard Cornell reflectances (RGB, 0–1)
_CORNELL_WHITE = vec3(0.730, 0.733, 0.737)
_CORNELL_RED = vec3(0.630, 0.065, 0.050)
_CORNELL_GREEN = vec3(0.140, 0.450, 0.091)
_CORNELL_BLUE = vec3(0.090, 0.210, 0.720)


def build_cornell_scene(bvh_split: str = "sah"):
    white = create_lambertian(create_solid_texture(_CORNELL_WHITE))
    red = create_lambertian(create_solid_texture(_CORNELL_RED))
    green = create_lambertian(create_solid_texture(_CORNELL_GREEN))
    blue = create_lambertian(create_solid_texture(_CORNELL_BLUE))
    tall_mat = create_lambertian(
        create_uv_checker_texture(vec3(0.78, 0.78, 0.82), vec3(0.22, 0.28, 0.55), 5)
    )
    short_mat = create_lambertian(
        create_uv_checker_texture(vec3(0.75, 0.72, 0.68), vec3(0.28, 0.32, 0.38), 4)
    )
    light = create_diffuse_light(vec3(15.0, 15.0, 15.0))
    glass = create_dielectric(1.5)
    metal = create_metal(vec3(0.92, 0.90, 0.88), 0.03)
    checker_ball = create_lambertian(
        create_uv_checker_texture(vec3(0.82, 0.82, 0.84), vec3(0.28, 0.30, 0.35), 6)
    )

    MAT_WHITE, MAT_RED, MAT_GREEN = 0, 1, 2
    MAT_TALL, MAT_SHORT, MAT_LIGHT = 3, 4, 5
    MAT_GLASS, MAT_METAL, MAT_CHECKER, MAT_BLUE = 6, 7, 8, 9

    mats_list = [
        white,
        red,
        green,
        tall_mat,
        short_mat,
        light,
        glass,
        metal,
        checker_ball,
        blue,
    ]

    prims: list[ScenePrimitive] = []
    prims.append(
        scene_primitive_quad(point3(-1, 0, -1), vec3(2, 0, 0), vec3(0, 0, 2), MAT_WHITE)
    )
    prims.append(
        scene_primitive_quad(point3(-1, 2, -1), vec3(2, 0, 0), vec3(0, 0, 2), MAT_WHITE)
    )
    prims.append(
        scene_primitive_quad(point3(-1, 0, -1), vec3(2, 0, 0), vec3(0, 2, 0), MAT_WHITE)
    )
    prims.append(
        scene_primitive_quad(point3(-1, 0, -1), vec3(0, 2, 0), vec3(0, 0, 2), MAT_RED)
    )
    prims.append(
        scene_primitive_quad(point3(1, 0, -1), vec3(0, 0, 2), vec3(0, 2, 0), MAT_GREEN)
    )
    ly = 1.985
    lw, lh = 0.48, 0.48
    lq = (-0.5 * lw, ly, -0.5 * lh)
    lu, lv = (lw, 0, 0), (0, 0, lh)
    prims.append(
        scene_primitive_quad(
            point3(lq[0], lq[1], lq[2]),
            vec3(lu[0], lu[1], lu[2]),
            vec3(lv[0], lv[1], lv[2]),
            MAT_LIGHT,
        )
    )
    prims.append(
        scene_primitive_quad(
            point3(-1, 0, 1),
            vec3(2, 0, 0),
            vec3(0, 2, 0),
            MAT_BLUE,
            invisible_primary=True,
        )
    )

    # Rounded blocks (beveled edges via SDF rounded box)
    prims.append(
        scene_primitive_sdf_rounded_box(
            point3(-0.48, 0.79, -0.52),
            vec3(0.20, 0.79, 0.20),
            0.045,
            MAT_TALL,
        )
    )
    prims.append(
        scene_primitive_sdf_rounded_box(
            point3(0.48, 0.51, 0.38),
            vec3(0.20, 0.51, 0.20),
            0.04,
            MAT_SHORT,
        )
    )

    # Grounded spheres (center y = radius on floor y=0)
    prims.append(
        scene_primitive_sphere(
            point3(-0.36, 0.30, 0.32), point3(-0.36, 0.30, 0.32), 0.30, MAT_CHECKER
        )
    )
    prims.append(
        scene_primitive_sphere(
            point3(0.12, 0.34, -0.42), point3(0.12, 0.34, -0.42), 0.34, MAT_GLASS
        )
    )
    # Metal sphere resting on short block top (y ≈ 1.02 + radius)
    prims.append(
        scene_primitive_sphere(
            point3(0.58, 1.14, 0.38), point3(0.58, 1.14, 0.38), 0.12, MAT_METAL
        )
    )
    # Small sphere on tall block
    prims.append(
        scene_primitive_sphere(
            point3(-0.48, 1.74, -0.52), point3(-0.48, 1.74, -0.52), 0.16, MAT_WHITE
        )
    )

    prim_field = ScenePrimitive.field(shape=len(prims), layout=ti.Layout.SOA)
    for i, p in enumerate(prims):
        prim_field[i] = p

    bvh = PrimBVH(prim_field, len(prims), split=bvh_split, host_primitives=prims)
    debug_tools.completed(f"Cornell BVH built ({bvh.memory_summary()})")
    root = os.path.dirname(os.path.abspath(__file__))
    cam = dict(
        lookfrom=point3(0.0, 1.0, 3.45),
        lookat=point3(0.0, 1.0, 0.0),
        vup=vec3(0.0, 1.0, 0.0),
        vfov=34.0,
        aspect_ratio=1.0,
        dark_background=1,
        nee_lights=[quad_light(lq, lu, lv, (15.0, 15.0, 15.0))],
        default_spp=512,
        light_samples=4,
        exposure=1.0,
        max_bounces=12,
    )
    return mats_list, bvh, root, cam


def build_cornell_model_scene(bvh_split: str = "sah"):
    """Closed Cornell box with an imported model and a primary-invisible front wall."""
    white = create_lambertian(create_solid_texture(_CORNELL_WHITE))
    red = create_lambertian(create_solid_texture(_CORNELL_RED))
    green = create_lambertian(create_solid_texture(_CORNELL_GREEN))
    blue = create_lambertian(create_solid_texture(_CORNELL_BLUE))
    light = create_diffuse_light(vec3(15.0, 15.0, 15.0))

    root = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(root, "models")
    model_path = select_best_model(models_dir, prefer_obj=True)
    mesh = load_mesh_auto(model_path, fit_to_cornell=True, target_height=1.20)
    print(f"Loaded OBJ model: {model_path}", flush=True)
    debug_tools.completed(f"model loaded ({len(mesh.triangles):,} triangles from {model_path})")

    material_base = 5
    mesh_mats = []
    for color_tuple in mesh.material_colors:
        c = vec3(
            max(0.02, min(float(color_tuple[0]), 0.95)),
            max(0.02, min(float(color_tuple[1]), 0.95)),
            max(0.02, min(float(color_tuple[2]), 0.95)),
        )
        mesh_mats.append(create_lambertian(create_solid_texture(c)))
    if not mesh_mats:
        mesh_mats.append(create_lambertian(create_solid_texture(vec3(0.72, 0.70, 0.66))))

    MAT_WHITE, MAT_RED, MAT_GREEN, MAT_BLUE, MAT_LIGHT = 0, 1, 2, 3, 4
    mats_list = [white, red, green, blue, light] + mesh_mats

    prims: list[ScenePrimitive] = []
    prims.append(scene_primitive_quad(point3(-1, 0, -1), vec3(2, 0, 0), vec3(0, 0, 2), MAT_WHITE))
    prims.append(scene_primitive_quad(point3(-1, 2, -1), vec3(2, 0, 0), vec3(0, 0, 2), MAT_WHITE))
    prims.append(scene_primitive_quad(point3(-1, 0, -1), vec3(2, 0, 0), vec3(0, 2, 0), MAT_WHITE))
    prims.append(scene_primitive_quad(point3(-1, 0, -1), vec3(0, 2, 0), vec3(0, 0, 2), MAT_RED))
    prims.append(scene_primitive_quad(point3(1, 0, -1), vec3(0, 0, 2), vec3(0, 2, 0), MAT_GREEN))
    prims.append(
        scene_primitive_quad(
            point3(-1, 0, 1),
            vec3(2, 0, 0),
            vec3(0, 2, 0),
            MAT_BLUE,
            invisible_primary=True,
        )
    )

    ly = 1.985
    lw, lh = 0.48, 0.48
    lq = (-0.5 * lw, ly, -0.5 * lh)
    lu, lv = (lw, 0, 0), (0, 0, lh)
    prims.append(scene_primitive_quad(point3(lq[0], lq[1], lq[2]), vec3(lu[0], lu[1], lu[2]), vec3(lv[0], lv[1], lv[2]), MAT_LIGHT))

    triangles = mesh.triangles

    prim_field = ScenePrimitive.field(shape=len(prims), layout=ti.Layout.SOA)
    for i, p in enumerate(prims):
        prim_field[i] = p

    bvh = PrimBVH(
        prim_field,
        len(prims),
        split=bvh_split,
        host_primitives=prims,
        host_triangles=triangles,
        triangle_mat_id=material_base,
        host_materials=mats_list,
    )
    print(f"Cornell model BVH memory: {bvh.memory_summary()}", flush=True)
    debug_tools.completed(f"Cornell model BVH built ({bvh.memory_summary()})")
    cam = dict(
        lookfrom=point3(0.0, 1.08, 3.55),
        lookat=point3(0.0, 0.92, 0.02),
        vup=vec3(0.0, 1.0, 0.0),
        vfov=31.0,
        aspect_ratio=1.0,
        dark_background=1,
        nee_lights=[quad_light(lq, lu, lv, (15.0, 15.0, 15.0))],
        default_spp=256,
        light_samples=4,
        nee_max_depth=4,
        exposure=1.0,
        max_bounces=12,
    )
    return mats_list, bvh, root, cam
