import math
import os

import taichi as ti
import taichi.math as tm

from src.accel.scene_accel import (PrimBVH, ScenePrimitive,
                                   scene_primitive_box, scene_primitive_quad,
                                   scene_primitive_sphere)
from src.shading.materials import (create_apple_peel_texture,
                                   create_dielectric, create_diffuse_light,
                                   create_lambertian, create_metal,
                                   create_solid_texture,
                                   create_uv_checker_texture)

vec3 = tm.vec3
point3 = tm.vec3


def quad_light(Q, u, v, Le):
    uu = (float(u[0]), float(u[1]), float(u[2]))
    vv = (float(v[0]), float(v[1]), float(v[2]))
    cx = uu[1] * vv[2] - uu[2] * vv[1]
    cy = uu[2] * vv[0] - uu[0] * vv[2]
    cz = uu[0] * vv[1] - uu[1] * vv[0]
    area = math.sqrt(cx * cx + cy * cy + cz * cz)
    if area < 1e-20:
        area = 1e-20
    n = (cx / area, cy / area, cz / area)
    return {
        "Q": (float(Q[0]), float(Q[1]), float(Q[2])),
        "u": uu,
        "v": vv,
        "n": n,
        "area": float(area),
        "Le": (float(Le[0]), float(Le[1]), float(Le[2])),
    }


def build_studio_scene(bvh_split: str = "sah"):
    W, H, D = (2.2, 2.45, 2.4)
    white = create_lambertian(create_solid_texture(vec3(0.88, 0.88, 0.9)))
    light_mat = create_diffuse_light(vec3(14.0, 14.0, 14.0))
    apple = create_lambertian(create_apple_peel_texture(4.2))
    MAT_W = 0
    MAT_L = 1
    MAT_APPLE = 2
    mats_list = [white, light_mat, apple]
    prims: list[ScenePrimitive] = []
    prims.append(
        scene_primitive_quad(
            point3(-W, 0.0, -D), vec3(0.0, 0.0, D), vec3(2.0 * W, 0.0, 0.0), MAT_W
        )
    )
    prims.append(
        scene_primitive_quad(
            point3(-W, H, -D), vec3(2.0 * W, 0.0, 0.0), vec3(0.0, 0.0, D), MAT_W
        )
    )
    prims.append(
        scene_primitive_quad(
            point3(-W, 0.0, -D), vec3(2.0 * W, 0.0, 0.0), vec3(0.0, H, 0.0), MAT_W
        )
    )
    prims.append(
        scene_primitive_quad(
            point3(-W, 0.0, -D), vec3(0.0, H, 0.0), vec3(0.0, 0.0, D), MAT_W
        )
    )
    prims.append(
        scene_primitive_quad(
            point3(W, 0.0, -D), vec3(0.0, 0.0, D), vec3(0.0, H, 0.0), MAT_W
        )
    )
    ly = H - 0.01
    lw, lz = (1.1, 0.95)
    lq = (-0.5 * lw, ly, -0.55 * D)
    lu = (lw, 0.0, 0.0)
    lv = (0.0, 0.0, lz)
    prims.append(
        scene_primitive_quad(
            point3(lq[0], lq[1], lq[2]),
            vec3(lu[0], lu[1], lu[2]),
            vec3(lv[0], lv[1], lv[2]),
            MAT_L,
        )
    )
    prims.append(
        scene_primitive_box(
            point3(-1.05, 0.0, -1.55), point3(-0.35, 1.95, -0.85), MAT_W
        )
    )
    prims.append(
        scene_primitive_box(point3(0.15, 0.0, -1.65), point3(1.05, 1.05, -0.75), MAT_W)
    )
    prims.append(
        scene_primitive_sphere(
            point3(-0.15, 0.42, -1.2), point3(-0.15, 0.42, -1.2), 0.38, MAT_APPLE
        )
    )
    prim_field = ScenePrimitive.field(shape=len(prims), layout=ti.Layout.SOA)
    for i, p in enumerate(prims):
        prim_field[i] = p
    bvh = PrimBVH(prim_field, len(prims), split=bvh_split, host_primitives=prims)
    root = os.path.dirname(os.path.abspath(__file__))
    nee = [quad_light(lq, lu, lv, (14.0, 14.0, 14.0))]
    cam = dict(
        lookfrom=point3(0.0, 1.15, 1.85),
        lookat=point3(0.0, 0.85, -1.0),
        vup=vec3(0.0, 1.0, 0.0),
        vfov=42.0,
        aspect_ratio=16.0 / 9.0,
        dark_background=1,
        nee_lights=nee,
    )
    print(f"Studio scene: {len(prims)} primitives, PrimBVH ({bvh_split})")
    print(f"BVH max node index ~{bvh.node_count}, ordered prims {bvh.prim_count}")
    return (mats_list, bvh, root, cam)


def build_spheres_gallery_scene(bvh_split: str = "sah"):
    ground = create_lambertian(create_solid_texture(vec3(0.5, 0.5, 0.55)))
    diff = create_lambertian(
        create_uv_checker_texture(vec3(0.9, 0.2, 0.2), vec3(0.15, 0.5, 0.2), 8)
    )
    metal = create_metal(vec3(0.9, 0.85, 0.5), 0.15)
    glass = create_dielectric(1.5)
    light_mat = create_diffuse_light(vec3(10.0, 10.0, 10.0))
    MAT_G, MAT_D, MAT_M, MAT_GL, MAT_L = (0, 1, 2, 3, 4)
    mats_list = [ground, diff, metal, glass, light_mat]
    prims: list[ScenePrimitive] = []
    prims.append(
        scene_primitive_sphere(
            point3(0, -100.5, -1), point3(0, -100.5, -1), 100.0, MAT_G
        )
    )
    prims.append(
        scene_primitive_sphere(
            point3(-1.05, 0.2, -1.0), point3(-1.05, 0.2, -1.0), 0.45, MAT_D
        )
    )
    prims.append(
        scene_primitive_sphere(
            point3(0.0, 0.2, -1.0), point3(0.0, 0.2, -1.0), 0.45, MAT_M
        )
    )
    prims.append(
        scene_primitive_sphere(
            point3(1.05, 0.2, -1.0), point3(1.05, 0.2, -1.0), 0.45, MAT_GL
        )
    )
    lw, lz = (2.5, 1.2)
    lq = (-1.25, 2.35, -1.8)
    lu = (lw, 0.0, 0.0)
    lv = (0.0, 0.0, lz)
    prims.append(
        scene_primitive_quad(
            point3(lq[0], lq[1], lq[2]),
            vec3(lu[0], lu[1], lu[2]),
            vec3(lv[0], lv[1], lv[2]),
            MAT_L,
        )
    )
    prim_field = ScenePrimitive.field(shape=len(prims), layout=ti.Layout.SOA)
    for i, p in enumerate(prims):
        prim_field[i] = p
    bvh = PrimBVH(prim_field, len(prims), split=bvh_split, host_primitives=prims)
    root = os.path.dirname(os.path.abspath(__file__))
    nee = [quad_light(lq, lu, lv, (10.0, 10.0, 10.0))]
    cam = dict(
        lookfrom=point3(-0.2, 0.9, 1.6),
        lookat=point3(0.0, 0.2, -1.0),
        vup=vec3(0.0, 1.0, 0.0),
        vfov=48.0,
        aspect_ratio=16.0 / 9.0,
        dark_background=0,
        nee_lights=nee,
    )
    print(f"Spheres gallery: {len(prims)} primitives, PrimBVH ({bvh_split})")
    print(f"BVH max node index ~{bvh.node_count}, ordered prims {bvh.prim_count}")
    return (mats_list, bvh, root, cam)
