import os
import taichi as ti
import taichi.math as tm
from src.shading.materials import create_checker_texture, create_dielectric, create_diffuse_light, create_lambertian, create_marble_texture, create_metal, create_solid_texture, create_uv_checker_texture
from src.scenes.scene_presets import quad_light
from src.accel.scene_accel import PrimBVH, ScenePrimitive, load_mesh_triangles, scene_primitive_box, scene_primitive_ellipsoid, scene_primitive_quad, scene_primitive_sdf_capsule, scene_primitive_sdf_rounded_box, scene_primitive_sdf_torus, scene_primitive_sphere
vec3 = tm.vec3
point3 = tm.vec3

def build_complex_scene(bvh_split: str='gpu'):
    ground_tex = create_checker_texture(vec3(0.15, 0.18, 0.22), vec3(0.55, 0.58, 0.62), 1.2)
    ground = create_lambertian(ground_tex)
    diff = create_lambertian(create_uv_checker_texture(vec3(0.9, 0.25, 0.2), vec3(0.1, 0.45, 0.75), 7))
    marble = create_lambertian(create_marble_texture(vec3(0.92, 0.9, 0.88), vec3(0.35, 0.38, 0.55), 2.5))
    metal = create_metal(vec3(0.85, 0.78, 0.55), 0.12)
    glass = create_dielectric(1.5)
    mesh_mat = create_lambertian(create_uv_checker_texture(vec3(0.95, 0.55, 0.1), vec3(0.08, 0.12, 0.45), 8))
    ellipsoid_mat = create_lambertian(create_marble_texture(vec3(0.7, 0.2, 0.5), vec3(0.15, 0.05, 0.2), 4.0))
    sdf_mat = create_lambertian(create_uv_checker_texture(vec3(0.2, 0.85, 0.4), vec3(0.85, 0.15, 0.5), 5))
    light_mat = create_diffuse_light(vec3(16.0, 16.0, 16.0))
    MAT_G, MAT_D, MAT_MARBLE, MAT_METAL, MAT_GL, MAT_MESH, MAT_ELL, MAT_SDF, MAT_L = range(9)
    mats_list = [ground, diff, marble, metal, glass, mesh_mat, ellipsoid_mat, sdf_mat, light_mat]
    prims: list[ScenePrimitive] = []
    prims.append(scene_primitive_sphere(point3(0, -1000, -1), point3(0, -1000, -1), 1000.0, MAT_G))
    prims.append(scene_primitive_sphere(point3(-1.4, 0.35, -1.2), point3(-1.4, 0.35, -1.2), 0.38, MAT_D))
    prims.append(scene_primitive_sphere(point3(0.0, 0.35, -1.15), point3(0.0, 0.35, -1.15), 0.38, MAT_MARBLE))
    prims.append(scene_primitive_sphere(point3(0.72, 0.28, -0.48), point3(0.72, 0.28, -0.48), 0.24, MAT_METAL))
    prims.append(scene_primitive_sphere(point3(1.35, 0.35, -1.2), point3(1.35, 0.35, -1.2), 0.38, MAT_GL))
    prims.append(scene_primitive_box(point3(-0.55, 0.0, -1.75), point3(0.55, 0.55, -0.65), MAT_D))
    prims.append(scene_primitive_ellipsoid(point3(-0.15, 0.95, -0.95), vec3(0.28, 0.14, 0.2), MAT_ELL))
    prims.append(scene_primitive_sdf_torus(point3(1.05, 0.55, -0.75), 0.32, 0.12, MAT_SDF))
    prims.append(scene_primitive_sdf_capsule(point3(-1.05, 0.45, -0.55), 0.18, 0.28, MAT_SDF))
    prims.append(scene_primitive_sdf_rounded_box(point3(0.55, 0.22, -0.45), vec3(0.22, 0.14, 0.18), 0.06, MAT_SDF))
    root = os.path.dirname(os.path.abspath(__file__))
    mesh_triangles = []
    cube_path = os.path.join(root, 'models', 'cube.obj')
    if os.path.isfile(cube_path):
        _, mesh_triangles = load_mesh_triangles(cube_path, scale=0.22, translate=(0.0, 0.42, -0.88), max_triangles=16000)
        print(f'Complex scene mesh: {len(mesh_triangles)} triangles')
    lw, lz = (2.8, 1.4)
    lq = (-1.4, 2.65, -2.2)
    lu = (lw, 0.0, 0.0)
    lv = (0.0, 0.0, lz)
    prims.append(scene_primitive_quad(point3(lq[0], lq[1], lq[2]), vec3(lu[0], lu[1], lu[2]), vec3(lv[0], lv[1], lv[2]), MAT_L))
    prim_field = ScenePrimitive.field(shape=len(prims), layout=ti.Layout.SOA)
    for i, p in enumerate(prims):
        prim_field[i] = p
    bvh = PrimBVH(prim_field, len(prims), split=bvh_split, host_primitives=prims, host_triangles=mesh_triangles, triangle_mat_id=MAT_MESH)
    print(f'Complex scene: {len(prims)} analytic primitives, {len(mesh_triangles)} triangles, PrimBVH ({bvh_split})')
    print(f'BVH max node index ~{bvh.node_count}, ordered prims {bvh.prim_count}')
    print(f'BVH memory: {bvh.memory_summary()}')
    cam = dict(lookfrom=point3(-0.3, 1.05, 2.4), lookat=point3(0.0, 0.45, -1.0), vup=vec3(0.0, 1.0, 0.0), vfov=44.0, aspect_ratio=16.0 / 9.0, dark_background=0, nee_lights=[quad_light(lq, lu, lv, (16.0, 16.0, 16.0))])
    return (mats_list, bvh, root, cam)