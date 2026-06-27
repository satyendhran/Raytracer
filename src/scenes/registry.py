import os
from typing import Callable
import taichi as ti
import taichi.math as tm
from src.shading.materials import create_dielectric, create_diffuse_light, create_lambertian, create_metal, create_solid_texture, create_uv_checker_texture
from src.scenes.scene_cornell import build_cornell_model_scene, build_cornell_scene
from src.scenes.scene_presets import build_spheres_gallery_scene, build_studio_scene, quad_light
from src.accel.scene_accel import PrimBVH, ScenePrimitive, scene_primitive_box, scene_primitive_quad, scene_primitive_sphere
from src.scenes.scene_complex import build_complex_scene
vec3 = tm.vec3
point3 = tm.vec3
_W = vec3(0.73, 0.733, 0.737)
_R = vec3(0.63, 0.065, 0.05)
_G = vec3(0.14, 0.45, 0.091)
_B = vec3(0.09, 0.21, 0.72)

def _finish(prims, mats, bvh_split, cam, name: str):
    pf = ScenePrimitive.field(shape=len(prims), layout=ti.Layout.SOA)
    for i, p in enumerate(prims):
        pf[i] = p
    bvh = PrimBVH(pf, len(prims), split=bvh_split, host_primitives=prims)
    root = os.path.dirname(os.path.abspath(__file__))
    return (mats, bvh, root, cam)

def _room(prims, mi_floor, mi_white, mi_red, mi_green, mi_front=None):
    if mi_front is None:
        mi_front = mi_white
    prims.append(scene_primitive_quad(point3(-1, 0, -1), vec3(2, 0, 0), vec3(0, 0, 2), mi_floor))
    prims.append(scene_primitive_quad(point3(-1, 2, -1), vec3(2, 0, 0), vec3(0, 0, 2), mi_white))
    prims.append(scene_primitive_quad(point3(-1, 0, -1), vec3(2, 0, 0), vec3(0, 2, 0), mi_white))
    prims.append(scene_primitive_quad(point3(-1, 0, -1), vec3(0, 2, 0), vec3(0, 0, 2), mi_red))
    prims.append(scene_primitive_quad(point3(1, 0, -1), vec3(0, 0, 2), vec3(0, 2, 0), mi_green))
    prims.append(scene_primitive_quad(point3(-1, 0, 1), vec3(2, 0, 0), vec3(0, 2, 0), mi_front, invisible_primary=True))

def _add_front_blue_material(mats):
    mats.append(create_lambertian(create_solid_texture(_B)))
    return len(mats) - 1

def _add_ceiling_light(prims, mi_light, intensity=15.0):
    ly, lw, lh = (1.985, 0.48, 0.48)
    lq = (-0.5 * lw, ly, -0.5 * lh)
    lu, lv = ((lw, 0, 0), (0, 0, lh))
    prims.append(scene_primitive_quad(point3(lq[0], lq[1], lq[2]), vec3(lu[0], lu[1], lu[2]), vec3(lv[0], lv[1], lv[2]), mi_light))
    return (lq, lu, lv, quad_light(lq, lu, lv, (intensity, intensity, intensity)))

def _cam(lq, lu, lv, le, spp=256, vfov=34.0, look_z=3.45, bounces=12):
    return dict(lookfrom=point3(0, 1, look_z), lookat=point3(0, 1, 0), vup=vec3(0, 1, 0), vfov=vfov, aspect_ratio=1.0, dark_background=1, nee_lights=[quad_light(lq, lu, lv, (le, le, le))], default_spp=spp, light_samples=4, exposure=1.0, max_bounces=bounces)

def build_test_cornell(bvh_split: str='sah'):
    return build_cornell_scene(bvh_split)

def build_test_classic_boxes(bvh_split: str='sah'):
    mats = [create_lambertian(create_solid_texture(_W)), create_lambertian(create_solid_texture(_R)), create_lambertian(create_solid_texture(_G)), create_diffuse_light(vec3(15, 15, 15))]
    prims = []
    _room(prims, 0, 0, 1, 2, _add_front_blue_material(mats))
    lq, lu, lv, _ = _add_ceiling_light(prims, 3)
    prims.append(scene_primitive_box(point3(-0.68, 0, -0.72), point3(-0.28, 1.58, -0.32), 0))
    prims.append(scene_primitive_box(point3(0.28, 0, 0.18), point3(0.68, 1.02, 0.58), 0))
    return _finish(prims, mats, bvh_split, _cam(lq, lu, lv, 15), 'classic_boxes')

def build_test_glass_only(bvh_split: str='sah'):
    mats = [create_lambertian(create_solid_texture(_W)), create_lambertian(create_solid_texture(_R)), create_lambertian(create_solid_texture(_G)), create_diffuse_light(vec3(15, 15, 15)), create_dielectric(1.5)]
    prims = []
    _room(prims, 0, 0, 1, 2, _add_front_blue_material(mats))
    lq, lu, lv, _ = _add_ceiling_light(prims, 3)
    prims.append(scene_primitive_sphere(point3(0, 0.35, 0), point3(0, 0.35, 0), 0.35, 4))
    return _finish(prims, mats, bvh_split, _cam(lq, lu, lv, 15, spp=384), 'glass_only')

def build_test_metal_only(bvh_split: str='sah'):
    mats = [create_lambertian(create_solid_texture(_W)), create_lambertian(create_solid_texture(_R)), create_lambertian(create_solid_texture(_G)), create_diffuse_light(vec3(15, 15, 15)), create_metal(vec3(0.92, 0.9, 0.88), 0.02)]
    prims = []
    _room(prims, 0, 0, 1, 2, _add_front_blue_material(mats))
    lq, lu, lv, _ = _add_ceiling_light(prims, 3)
    prims.append(scene_primitive_sphere(point3(0, 0.35, 0), point3(0, 0.35, 0), 0.35, 4))
    return _finish(prims, mats, bvh_split, _cam(lq, lu, lv, 15), 'metal_only')

def build_test_checker_sphere(bvh_split: str='sah'):
    chk = create_lambertian(create_uv_checker_texture(vec3(0.8, 0.8, 0.82), vec3(0.25, 0.28, 0.32), 8))
    mats = [create_lambertian(create_solid_texture(_W)), create_lambertian(create_solid_texture(_R)), create_lambertian(create_solid_texture(_G)), create_diffuse_light(vec3(15, 15, 15)), chk]
    prims = []
    _room(prims, 0, 0, 1, 2, _add_front_blue_material(mats))
    lq, lu, lv, _ = _add_ceiling_light(prims, 3)
    prims.append(scene_primitive_sphere(point3(0, 0.32, 0), point3(0, 0.32, 0), 0.32, 4))
    return _finish(prims, mats, bvh_split, _cam(lq, lu, lv, 15), 'checker_sphere')

def build_test_material_spheres_cornell(bvh_split: str='sah'):
    mats = [create_lambertian(create_solid_texture(_W)), create_lambertian(create_solid_texture(_R)), create_lambertian(create_solid_texture(_G)), create_diffuse_light(vec3(15, 15, 15)), create_lambertian(create_uv_checker_texture(vec3(0.92, 0.88, 0.22), vec3(0.12, 0.22, 0.85), 7)), create_metal(vec3(0.95, 0.92, 0.86), 0.0), create_metal(vec3(0.8, 0.55, 0.35), 0.25), create_dielectric(1.5), create_diffuse_light(vec3(3.0, 0.7, 0.35))]
    prims = []
    _room(prims, 0, 0, 1, 2, _add_front_blue_material(mats))
    lq, lu, lv, _ = _add_ceiling_light(prims, 3)
    prims.append(scene_primitive_sphere(point3(-0.55, 0.22, 0.35), point3(-0.55, 0.22, 0.35), 0.22, 4))
    prims.append(scene_primitive_sphere(point3(0.05, 0.28, 0.15), point3(0.05, 0.28, 0.15), 0.28, 5))
    prims.append(scene_primitive_sphere(point3(0.58, 0.2, 0.42), point3(0.58, 0.2, 0.42), 0.2, 6))
    prims.append(scene_primitive_sphere(point3(-0.22, 0.34, -0.35), point3(-0.22, 0.34, -0.35), 0.34, 7))
    prims.append(scene_primitive_sphere(point3(0.52, 0.14, -0.35), point3(0.52, 0.14, -0.35), 0.14, 8))
    cam = _cam(lq, lu, lv, 15, spp=384, vfov=36.0, bounces=14)
    cam['lookfrom'] = point3(0, 1.0, 3.6)
    cam['lookat'] = point3(0, 0.75, 0.0)
    return _finish(prims, mats, bvh_split, cam, 'material_spheres_cornell')

def build_test_material_quads_cornell(bvh_split: str='sah'):
    mats = [create_lambertian(create_solid_texture(_W)), create_lambertian(create_solid_texture(_R)), create_lambertian(create_solid_texture(_G)), create_diffuse_light(vec3(18, 18, 18)), create_lambertian(create_uv_checker_texture(vec3(0.95, 0.82, 0.18), vec3(0.08, 0.14, 0.75), 8)), create_lambertian(create_solid_texture(vec3(0.88, 0.3, 0.16))), create_lambertian(create_solid_texture(vec3(0.16, 0.55, 0.82))), create_metal(vec3(0.98, 0.96, 0.92), 0.0), create_metal(vec3(0.8, 0.86, 0.92), 0.22), create_diffuse_light(vec3(0.45, 1.7, 3.2))]
    prims = []
    _room(prims, 0, 0, 1, 2, _add_front_blue_material(mats))
    lq, lu, lv, _ = _add_ceiling_light(prims, 3, intensity=18.0)
    prims.append(scene_primitive_quad(point3(-0.84, 0.34, -0.985), vec3(0.46, 0.0, 0.0), vec3(0.0, 0.58, 0.0), 4))
    prims.append(scene_primitive_quad(point3(-0.27, 0.34, -0.985), vec3(0.46, 0.0, 0.0), vec3(0.0, 0.58, 0.0), 5))
    prims.append(scene_primitive_quad(point3(0.3, 0.34, -0.985), vec3(0.46, 0.0, 0.0), vec3(0.0, 0.58, 0.0), 6))
    prims.append(scene_primitive_quad(point3(-0.84, 1.08, -0.985), vec3(0.46, 0.0, 0.0), vec3(0.0, 0.36, 0.0), 9))
    prims.append(scene_primitive_quad(point3(0.3, 1.08, -0.985), vec3(0.46, 0.0, 0.0), vec3(0.0, 0.36, 0.0), 9))
    prims.append(scene_primitive_quad(point3(-0.72, 0.003, 0.03), vec3(0.54, 0.0, 0.0), vec3(0.0, 0.0, 0.54), 7))
    prims.append(scene_primitive_quad(point3(0.2, 0.003, 0.03), vec3(0.54, 0.0, 0.0), vec3(0.0, 0.0, 0.54), 8))
    cam = _cam(lq, lu, lv, 18, spp=384, vfov=35.0, look_z=3.65, bounces=14)
    cam['lookat'] = point3(0.0, 0.78, -0.2)
    return _finish(prims, mats, bvh_split, cam, 'material_quads_cornell')

def build_test_material_boxes_cornell(bvh_split: str='sah'):
    mats = [create_lambertian(create_solid_texture(_W)), create_lambertian(create_solid_texture(_R)), create_lambertian(create_solid_texture(_G)), create_diffuse_light(vec3(16, 16, 16)), create_lambertian(create_uv_checker_texture(vec3(0.95, 0.82, 0.2), vec3(0.1, 0.18, 0.75), 5)), create_metal(vec3(0.96, 0.93, 0.86), 0.0), create_metal(vec3(0.75, 0.82, 0.9), 0.2), create_dielectric(1.5), create_diffuse_light(vec3(2.5, 0.9, 0.35))]
    prims = []
    _room(prims, 0, 0, 1, 2, _add_front_blue_material(mats))
    lq, lu, lv, _ = _add_ceiling_light(prims, 3, intensity=16.0)
    prims.append(scene_primitive_box(point3(-0.82, 0.0, -0.64), point3(-0.46, 0.5, -0.28), 4))
    prims.append(scene_primitive_box(point3(-0.3, 0.0, -0.72), point3(0.02, 0.86, -0.4), 5))
    prims.append(scene_primitive_box(point3(0.18, 0.0, -0.58), point3(0.54, 0.46, -0.22), 6))
    prims.append(scene_primitive_box(point3(0.56, 0.0, 0.1), point3(0.82, 0.62, 0.36), 7))
    prims.append(scene_primitive_box(point3(-0.08, 0.0, 0.2), point3(0.2, 0.22, 0.48), 8))
    cam = _cam(lq, lu, lv, 16, spp=512, vfov=34.0, look_z=3.8, bounces=18)
    cam['lookat'] = point3(0.0, 0.7, -0.12)
    return _finish(prims, mats, bvh_split, cam, 'material_boxes_cornell')

def build_test_reflection_refraction_boxes(bvh_split: str='sah'):
    mats = [create_lambertian(create_solid_texture(_W)), create_lambertian(create_solid_texture(_R)), create_lambertian(create_solid_texture(_G)), create_diffuse_light(vec3(15, 15, 15)), create_lambertian(create_uv_checker_texture(vec3(0.96, 0.96, 0.96), vec3(0.02, 0.04, 0.1), 12)), create_metal(vec3(0.98, 0.98, 1.0), 0.0), create_dielectric(1.5), create_lambertian(create_solid_texture(vec3(0.9, 0.16, 0.08)))]
    prims = []
    _room(prims, 0, 0, 1, 2, _add_front_blue_material(mats))
    lq, lu, lv, _ = _add_ceiling_light(prims, 3, intensity=15.0)
    prims.append(scene_primitive_quad(point3(-0.78, 0.2, -0.985), vec3(1.56, 0, 0), vec3(0, 0.72, 0), 4))
    prims.append(scene_primitive_box(point3(-0.82, 0.0, 0.1), point3(-0.38, 0.92, 0.34), 5))
    prims.append(scene_primitive_box(point3(0.22, 0.0, 0.12), point3(0.62, 0.92, 0.38), 6))
    prims.append(scene_primitive_sphere(point3(0.34, 0.2, -0.38), point3(0.34, 0.2, -0.38), 0.2, 7))
    cam = _cam(lq, lu, lv, 15, spp=512, vfov=33.0, look_z=3.9, bounces=20)
    cam['lookat'] = point3(0.0, 0.78, -0.08)
    return _finish(prims, mats, bvh_split, cam, 'reflection_refraction_boxes')

def _cornell_variant(bvh_split, le=15.0, spp=512, vfov=34.0, bounces=12, exposure=1.0):
    mats, bvh, root, cam = build_cornell_scene(bvh_split)
    ly, lw, lh = (1.985, 0.48, 0.48)
    lq = (-0.5 * lw, ly, -0.5 * lh)
    cam['nee_lights'] = [quad_light(lq, (lw, 0, 0), (0, 0, lh), (le, le, le))]
    cam['default_spp'] = spp
    cam['vfov'] = vfov
    cam['max_bounces'] = bounces
    cam['exposure'] = exposure
    return (mats, bvh, root, cam)

def build_test_dim_light(bvh_split: str='sah'):
    return _cornell_variant(bvh_split, le=4.0, exposure=1.2)

def build_test_bright_light(bvh_split: str='sah'):
    return _cornell_variant(bvh_split, le=22.0, exposure=0.82)

def build_test_wide_fov(bvh_split: str='sah'):
    return _cornell_variant(bvh_split, vfov=48.0)

def build_test_narrow_fov(bvh_split: str='sah'):
    return _cornell_variant(bvh_split, vfov=28.0)

def build_test_many_bounces(bvh_split: str='sah'):
    return _cornell_variant(bvh_split, bounces=16, spp=384)

def build_test_low_spp(bvh_split: str='sah'):
    return _cornell_variant(bvh_split, spp=32)

def build_test_high_spp(bvh_split: str='sah'):
    return _cornell_variant(bvh_split, spp=1024)

def build_test_default(bvh_split: str='sah'):
    import main as main_mod
    return main_mod.build_scene_and_bvh(bvh_split)
TEST_SCENE_BUILDERS: dict[str, Callable] = {'cornell': build_test_cornell, 'cornell_model': build_cornell_model_scene, 'classic_boxes': build_test_classic_boxes, 'glass_only': build_test_glass_only, 'metal_only': build_test_metal_only, 'checker_sphere': build_test_checker_sphere, 'material_spheres': build_test_material_spheres_cornell, 'material_quads': build_test_material_quads_cornell, 'material_boxes': build_test_material_boxes_cornell, 'reflection_refraction_boxes': build_test_reflection_refraction_boxes, 'wide_fov': build_test_wide_fov, 'narrow_fov': build_test_narrow_fov, 'many_bounces': build_test_many_bounces, 'low_spp': build_test_low_spp, 'high_spp': build_test_high_spp, 'default': build_test_default, 'studio': build_studio_scene, 'spheres': build_spheres_gallery_scene, 'complex': build_complex_scene, 'model_showcase': build_cornell_model_scene, 'cornell_material_spheres': build_test_material_spheres_cornell, 'cornell_material_quads': build_test_material_quads_cornell, 'cornell_material_boxes': build_test_material_boxes_cornell, 'glass_stress': build_test_glass_only, 'metal_stress': build_test_metal_only, 'checker_stress': build_test_checker_sphere, 'fov_wide': build_test_wide_fov, 'fov_narrow': build_test_narrow_fov, 'bounce_deep': build_test_many_bounces, 'spp_low': build_test_low_spp, 'spp_high': build_test_high_spp, 'studio_ref': build_studio_scene, 'spheres_ref': build_spheres_gallery_scene, 'complex_ref': build_test_material_quads_cornell, 'default_ref': build_test_default, 'reflection_refraction': build_test_reflection_refraction_boxes}
ALL_TEST_SCENE_NAMES = tuple(sorted(TEST_SCENE_BUILDERS.keys()))

def print_scene_table():
    print('\n| # | Scene | Notes |')
    print('|---|-------|-------|')
    for i, name in enumerate(ALL_TEST_SCENE_NAMES, 1):
        print(f'| {i} | `{name}` | debug render |')