import argparse
import math
import numbers
import os
from time import perf_counter as pf
from typing import Any, Sequence

import numpy as np
import taichi as ti
import taichi.math as tm

from src.accel.scene_accel import (KIND_QUAD, PrimBVH, ScenePrimitive,
                                   load_mesh_triangles,
                                   scene_primitive_ellipsoid,
                                   scene_primitive_quad,
                                   scene_primitive_sphere)
from src.core.Color import float_to_rgb8
from src.core.interval import Interval
from src.core.Rand import (random_cosine_direction, random_double,
                           random_in_unit_disk, random_unit_vec3, reflect,
                           refract)
from src.core.Ray import Ray
from src.core.render_core import sanitize_rgb, tonemap_filmic
from src.geometry.Objects import hit_record
from src.scenes.registry import (ALL_TEST_SCENE_NAMES, TEST_SCENE_BUILDERS,
                                 print_scene_table)
from src.shading.materials import (Material, create_checker_texture,
                                   create_dielectric, create_lambertian,
                                   create_metal, create_solid_texture,
                                   create_uv_checker_texture)
from tools import debug_tools

_TAICHI_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".taichi_cache"
)
os.makedirs(_TAICHI_CACHE_DIR, exist_ok=True)
ti.init(
    arch=ti.cuda,
    default_ip=ti.i32,
    default_fp=ti.f32,
    fast_math=True,
    offline_cache=True,
    offline_cache_file_path=_TAICHI_CACHE_DIR,
)
vec3 = tm.vec3
point3 = tm.vec3
RAY_PRIMARY = 0
RAY_SECONDARY = 1
RAY_SHADOW = 2
DEFAULT_IMAGE_WIDTH = 1920
_GPU_PROBE = ti.field(dtype=ti.i32, shape=())
materials = None


@ti.kernel
def _gpu_probe_kernel():
    _GPU_PROBE[None] = 1


def _probe_taichi_backend() -> None:
    _GPU_PROBE[None] = 0
    _gpu_probe_kernel()
    ti.sync()
    if int(_GPU_PROBE[None]) != 1:
        raise RuntimeError("Taichi GPU backend probe failed before rendering.")
    debug_tools.completed(f"Taichi backend probe (arch={ti.cfg.arch})")


def _host_normalize(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if n < 1e-20:
        return np.zeros(3, dtype=np.float64)
    return v / n


@ti.data_oriented
class Camera:
    NEE_LIGHT_CAPACITY = 4
    TRACE_BOUNCE_CAP = 12

    def __init__(
        self,
        bvh,
        image_width,
        aspect_ratio,
        vfov,
        lookfrom,
        lookat,
        vup,
        defocus_angle,
        focus_distance,
        world=None,
        use_hdr_accum: bool = False,
        dark_background: int = 0,
        use_nee: int = 1,
        max_bounces: int = 12,
    ):
        self.samples_per_pixel = 64
        self.pixel_sample_scale = 1.0 / self.samples_per_pixel
        self.offline_batch_size = 8
        self.debug_mode = False
        self.image_width = image_width
        self.image_height = max(1, int(image_width / aspect_ratio))
        self.img = ti.Vector.field(
            3, ti.u8, shape=(self.image_width, self.image_height), layout=ti.Layout.SOA
        )
        self.color_accum = ti.Vector.field(
            3, ti.f32, shape=(self.image_width, self.image_height), layout=ti.Layout.SOA
        )
        self.max_bounces = ti.field(dtype=ti.i32, shape=())
        self.max_bounces[None] = max_bounces
        self.exposure = ti.field(dtype=ti.f32, shape=())
        self.exposure[None] = 0.85
        self.light_samples = ti.field(dtype=ti.i32, shape=())
        self.light_samples[None] = 4
        self.ray_epsilon = ti.field(dtype=ti.f32, shape=())
        self.ray_epsilon[None] = 0.0001
        self.shadow_epsilon = ti.field(dtype=ti.f32, shape=())
        self.shadow_epsilon[None] = 0.0002
        self.material_count = ti.field(dtype=ti.i32, shape=())
        self.material_count[None] = (
            int(materials.shape[0]) if materials is not None else 1
        )
        self.use_hdr_accum = use_hdr_accum
        if use_hdr_accum:
            self.hdr_accum = ti.Vector.field(
                4,
                ti.f32,
                shape=(self.image_width, self.image_height),
                layout=ti.Layout.SOA,
            )
        self.bvh = bvh
        self.hit_rec = hit_record.field(
            shape=(self.image_width, self.image_height), layout=ti.Layout.SOA
        )
        self.camera_center = ti.Vector.field(3, ti.f32, shape=())
        self.lookat_f = ti.Vector.field(3, ti.f32, shape=())
        self.vup_f = ti.Vector.field(3, ti.f32, shape=())
        self.vfov = ti.field(ti.f32, shape=())
        self.defocus_angle = ti.field(ti.f32, shape=())
        self.focal_distance = ti.field(ti.f32, shape=())
        self.pixel00_loc = ti.Vector.field(3, ti.f32, shape=())
        self.pixel_delta_u = ti.Vector.field(3, ti.f32, shape=())
        self.pixel_delta_v = ti.Vector.field(3, ti.f32, shape=())
        self.defocus_disk_u = ti.Vector.field(3, ti.f32, shape=())
        self.defocus_disk_v = ti.Vector.field(3, ti.f32, shape=())
        self.atten = ti.Vector.field(
            3,
            dtype=ti.f32,
            shape=(self.image_width, self.image_height),
            layout=ti.Layout.SOA,
        )
        self.scattered = Ray.field(
            shape=(self.image_width, self.image_height), layout=ti.Layout.SOA
        )
        self.live_batch = ti.field(dtype=ti.i32, shape=())
        self.live_batch[None] = 1
        self.dark_background = ti.field(dtype=ti.i32, shape=())
        self.dark_background[None] = dark_background
        self.use_nee = ti.field(dtype=ti.i32, shape=())
        self.use_nee[None] = use_nee
        self.nee_max_depth = ti.field(dtype=ti.i32, shape=())
        self.nee_max_depth[None] = 4
        self.nee_count = ti.field(dtype=ti.i32, shape=())
        self.nee_count[None] = 0
        self.nee_light_indices = ti.field(
            dtype=ti.i32, shape=(self.NEE_LIGHT_CAPACITY,)
        )
        self.nee_Q = ti.Vector.field(3, ti.f32, shape=(self.NEE_LIGHT_CAPACITY,))
        self.nee_u = ti.Vector.field(3, ti.f32, shape=(self.NEE_LIGHT_CAPACITY,))
        self.nee_v = ti.Vector.field(3, ti.f32, shape=(self.NEE_LIGHT_CAPACITY,))
        self.nee_n = ti.Vector.field(3, ti.f32, shape=(self.NEE_LIGHT_CAPACITY,))
        self.nee_area = ti.field(dtype=ti.f32, shape=(self.NEE_LIGHT_CAPACITY,))
        self.nee_Le = ti.Vector.field(3, ti.f32, shape=(self.NEE_LIGHT_CAPACITY,))
        self.use_restir = ti.field(dtype=ti.i32, shape=())
        self.use_restir[None] = 0
        self.restir_temporal = ti.field(dtype=ti.f32, shape=())
        self.restir_temporal[None] = 0.75
        self.restir_curr_direct = ti.Vector.field(
            3, ti.f32, shape=(self.image_width, self.image_height), layout=ti.Layout.SOA
        )
        self.restir_reuse_direct = ti.Vector.field(
            3, ti.f32, shape=(self.image_width, self.image_height), layout=ti.Layout.SOA
        )
        self.denoise_enabled = ti.field(dtype=ti.i32, shape=())
        self.denoise_enabled[None] = 0
        self.denoise_iterations = ti.field(dtype=ti.i32, shape=())
        self.denoise_iterations[None] = 4
        self.denoise_phi_color = ti.field(dtype=ti.f32, shape=())
        self.denoise_phi_color[None] = 0.12
        self.denoise_ping = ti.Vector.field(
            3, ti.f32, shape=(self.image_width, self.image_height), layout=ti.Layout.SOA
        )
        self.denoise_pong = ti.Vector.field(
            3, ti.f32, shape=(self.image_width, self.image_height), layout=ti.Layout.SOA
        )
        self.camera_center[None] = lookfrom
        self.lookat_f[None] = lookat
        self.vup_f[None] = vup
        self.vfov[None] = vfov
        self.defocus_angle[None] = defocus_angle
        self.focal_distance[None] = focus_distance
        self.update_camera_rays()

    def set_lights(self, nee_list: Sequence[Any], use_nee: int):
        self.configure_nee(nee_list, bool(use_nee))

    def configure_nee(self, light_indices: Sequence[Any], enabled: bool = True) -> None:
        if not enabled or not light_indices:
            self.use_nee[None] = 0
            self.nee_count[None] = 0
            return
        if len(light_indices) > self.NEE_LIGHT_CAPACITY:
            raise ValueError(
                f"NEE supports at most {self.NEE_LIGHT_CAPACITY} lights, got {len(light_indices)}."
            )
        self.use_nee[None] = 1
        n = len(light_indices)
        self.nee_count[None] = n
        for k in range(n):
            light = light_indices[k]
            if isinstance(light, numbers.Integral):
                self._upload_indexed_nee_light(k, light)
            else:
                self._upload_nee_metadata(k, light)

    def _upload_nee_metadata(self, slot: int, L: Any) -> None:
        required = ("Q", "u", "v", "n", "area", "Le")
        missing = [key for key in required if key not in L]
        if missing:
            raise ValueError(f"NEE light metadata is missing keys: {missing}")
        self.nee_light_indices[slot] = -1
        self.nee_Q[slot] = tm.vec3(float(L["Q"][0]), float(L["Q"][1]), float(L["Q"][2]))
        self.nee_u[slot] = tm.vec3(float(L["u"][0]), float(L["u"][1]), float(L["u"][2]))
        self.nee_v[slot] = tm.vec3(float(L["v"][0]), float(L["v"][1]), float(L["v"][2]))
        self.nee_n[slot] = tm.vec3(float(L["n"][0]), float(L["n"][1]), float(L["n"][2]))
        self.nee_area[slot] = float(L["area"])
        self.nee_Le[slot] = tm.vec3(
            float(L["Le"][0]), float(L["Le"][1]), float(L["Le"][2])
        )

    def _upload_indexed_nee_light(self, slot: int, prim_index: int) -> None:
        if prim_index < 0 or prim_index >= self.bvh.prim_count:
            raise ValueError(f"NEE light primitive index {prim_index} is out of range.")
        prim = self.bvh.primitives[prim_index]
        if int(prim.kind) != KIND_QUAD:
            raise ValueError(f"NEE light primitive {prim_index} is not a quad.")
        mat_id = max(0, min(int(prim.mat_id), int(materials.shape[0]) - 1))
        mat = materials[mat_id]
        if int(mat.material_type) != 3:
            raise ValueError(
                f"NEE light primitive {prim_index} does not use an emissive material."
            )
        u = np.asarray(prim.center_vel, dtype=np.float64)
        v = np.asarray(prim.box_min, dtype=np.float64)
        n = np.asarray(prim.box_max, dtype=np.float64)
        area = float(np.linalg.norm(np.cross(u, v)))
        if area <= 1e-20:
            raise ValueError(f"NEE light primitive {prim_index} has near-zero area.")
        self.nee_light_indices[slot] = int(prim_index)
        self.nee_Q[slot] = prim.center0
        self.nee_u[slot] = prim.center_vel
        self.nee_v[slot] = prim.box_min
        self.nee_n[slot] = tm.vec3(float(n[0]), float(n[1]), float(n[2]))
        self.nee_area[slot] = area
        self.nee_Le[slot] = mat.emitted

    def update_camera_rays(self):
        ca = np.asarray(self.camera_center[None], dtype=np.float64)
        la = np.asarray(self.lookat_f[None], dtype=np.float64)
        vu = np.asarray(self.vup_f[None], dtype=np.float64)
        w = _host_normalize(ca - la)
        u = _host_normalize(np.cross(vu, w))
        v = np.cross(w, u)
        theta = math.radians(float(self.vfov[None]))
        h = math.tan(theta * 0.5)
        viewport_height = 2.0 * h * float(self.focal_distance[None])
        viewport_width = viewport_height * (self.image_width / self.image_height)
        viewport_u = u * viewport_width
        viewport_v = -v * viewport_height
        pdu = viewport_u / self.image_width
        pdv = viewport_v / self.image_height
        self.pixel_delta_u[None] = tm.vec3(float(pdu[0]), float(pdu[1]), float(pdu[2]))
        self.pixel_delta_v[None] = tm.vec3(float(pdv[0]), float(pdv[1]), float(pdv[2]))
        vup_left = (
            ca
            - float(self.focal_distance[None]) * w
            - 0.5 * viewport_u
            - 0.5 * viewport_v
        )
        p00 = vup_left + 0.5 * (pdu + pdv)
        self.pixel00_loc[None] = tm.vec3(float(p00[0]), float(p00[1]), float(p00[2]))
        defocus_radius = float(self.focal_distance[None]) * math.tan(
            math.radians(float(self.defocus_angle[None]) * 0.5)
        )
        du = u * defocus_radius
        dv = v * defocus_radius
        self.defocus_disk_u[None] = tm.vec3(float(du[0]), float(du[1]), float(du[2]))
        self.defocus_disk_v[None] = tm.vec3(float(dv[0]), float(dv[1]), float(dv[2]))

    @ti.func
    def power_heuristic(self, pdf_a: ti.f32, pdf_b: ti.f32) -> ti.f32:
        a2 = pdf_a * pdf_a
        b2 = pdf_b * pdf_b
        return a2 / ti.max(a2 + b2, 1e-08)

    @ti.func
    def offset_ray_origin(
        self, p: vec3, n: vec3, geo_n: vec3, direction: vec3, eps: ti.f32, hit_t: ti.f32
    ) -> vec3:
        use_geo = tm.dot(geo_n, geo_n) > 1e-12
        base_n = ti.select(use_geo, geo_n, n)
        side_n = ti.select(tm.dot(direction, base_n) >= 0.0, base_n, -base_n)
        bias = eps * ti.max(ti.abs(hit_t), 1e-06)
        return p + side_n * bias

    @ti.func
    def sample_area_light(
        self, p: vec3, n: vec3, albedo: vec3, i: ti.i32, j: ti.i32
    ) -> vec3:
        acc = vec3(0.0, 0.0, 0.0)
        cnt = self.nee_count[None]
        ns = ti.min(ti.max(self.light_samples[None], 1), 8)
        selected = vec3(0.0, 0.0, 0.0)
        selected_w = 0.0
        wsum = 0.0
        m = 0
        for k in range(4):
            if k < cnt:
                Q = self.nee_Q[k]
                u = self.nee_u[k]
                v = self.nee_v[k]
                nL = self.nee_n[k]
                A = self.nee_area[k]
                Le = self.nee_Le[k]
                for s in range(8):
                    if s < ns:
                        ns_f = ti.cast(ns, ti.f32)
                        s_f = ti.cast(s, ti.f32)
                        su = (s_f + ti.random()) / ns_f
                        sv_seed = (s_f + ti.random()) * 0.61803398875 + ti.cast(
                            k, ti.f32
                        ) * 0.37
                        sv = sv_seed - ti.floor(sv_seed)
                        xl = Q + su * u + sv * v
                        disp_pre = xl - p
                        dist = tm.length(disp_pre)
                        wi_pre = disp_pre / ti.max(dist, 1e-08)
                        roff = self.offset_ray_origin(
                            p,
                            n,
                            vec3(0.0, 0.0, 0.0),
                            wi_pre,
                            self.shadow_epsilon[None],
                            dist,
                        )
                        disp = xl - roff
                        dist2 = tm.dot(disp, disp)
                        dist = ti.sqrt(ti.max(dist2, 1e-08))
                        wi = disp / dist
                        cos_x = ti.abs(tm.dot(n, wi))
                        cos_l = tm.dot(nL, -wi)
                        geom_ok = (cos_x > 1e-05) & (cos_l > 1e-05) & (dist > 0.0001)
                        if geom_ok:
                            sh = Ray(roff, wi, 0.0, RAY_SHADOW)
                            eps = self.shadow_epsilon[None]
                            blocked = self.bvh.any_hit(
                                sh,
                                Interval(eps, ti.max(eps, dist - eps)),
                                self.hit_rec[i, j].prim_id,
                            )
                            if blocked == 0:
                                brdf = albedo / ti.math.pi
                                light_pdf = dist2 / ti.max(cos_l * A, 1e-06)
                                bsdf_pdf = cos_x / ti.math.pi
                                mis_w = self.power_heuristic(light_pdf, bsdf_pdf)
                                geom = cos_x * cos_l * A / ti.max(dist2, 1e-06)
                                contrib = brdf * Le * geom * mis_w
                                if self.use_restir[None] != 0:
                                    w = ti.max(self.luminance(contrib), 1e-08)
                                    wsum += w
                                    m += 1
                                    if ti.random() < w / ti.max(wsum, 1e-08):
                                        selected = contrib
                                        selected_w = w
                                else:
                                    acc += contrib
        result = acc / ti.cast(ns, ti.f32)
        if self.use_restir[None] != 0:
            local = vec3(0.0, 0.0, 0.0)
            if m > 0:
                local = selected * wsum / ti.max(selected_w * ti.cast(ns, ti.f32), 1e-08)
            reused = self.restir_reuse_direct[i, j]
            result = 0.78 * local + 0.22 * reused
            self.restir_curr_direct[i, j] = local
        return result

    @ti.func
    def clamp_bright_pixels(self, c: vec3) -> vec3:
        return sanitize_rgb(tm.min(c, vec3(25.0)))

    @ti.func
    def luminance(self, c: vec3) -> ti.f32:
        return 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z

    @ti.func
    def bsdf_light_mis_weight(self, ray: Ray, prev_p: vec3, prev_n: vec3) -> ti.f32:
        weight = 1.0
        if (self.use_nee[None] != 0) & (self.nee_count[None] > 0):
            best_pdf = 0.0
            for k in range(4):
                if k < self.nee_count[None]:
                    Q = self.nee_Q[k]
                    u = self.nee_u[k]
                    v = self.nee_v[k]
                    nL = self.nee_n[k]
                    denom = tm.dot(nL, ray.direction)
                    if ti.abs(denom) > 1e-08:
                        t = tm.dot(Q - prev_p, nL) / denom
                        if t > 0.0001:
                            hp = prev_p + ray.direction * t
                            rel = hp - Q
                            uu = tm.dot(u, u)
                            uv = tm.dot(u, v)
                            vv = tm.dot(v, v)
                            wu = tm.dot(rel, u)
                            wv = tm.dot(rel, v)
                            det = uu * vv - uv * uv
                            if ti.abs(det) > 1e-20:
                                inv_det = 1.0 / det
                                a = (wu * vv - wv * uv) * inv_det
                                b = (uu * wv - uv * wu) * inv_det
                                if (a >= 0.0) & (a <= 1.0) & (b >= 0.0) & (b <= 1.0):
                                    area = self.nee_area[k]
                                    dist2 = tm.dot(hp - prev_p, hp - prev_p)
                                    cos_l = ti.abs(tm.dot(nL, -ray.direction))
                                    best_pdf = ti.max(
                                        best_pdf, dist2 / ti.max(cos_l * area, 1e-06)
                                    )
            cos_x = ti.abs(tm.dot(prev_n, ray.direction))
            bsdf_pdf = cos_x / ti.math.pi
            weight = self.power_heuristic(bsdf_pdf, best_pdf)
        return weight

    @ti.func
    def scatter_material(self, mat: Material, r_in: Ray, rec):
        scatter_ok = 0
        attenuation = vec3(0.0, 0.0, 0.0)
        scattered = Ray(rec.p, rec.normal, r_in.tm, RAY_SECONDARY)
        if mat.material_type == 0:
            scatter_direction = random_cosine_direction(rec.normal)
            if (
                (ti.abs(scatter_direction.x) < 1e-08)
                & (ti.abs(scatter_direction.y) < 1e-08)
                & (ti.abs(scatter_direction.z) < 1e-08)
            ):
                scatter_direction = rec.normal
            scattered = Ray(
                self.offset_ray_origin(
                    rec.p,
                    rec.normal,
                    rec.geo_normal,
                    scatter_direction,
                    self.ray_epsilon[None],
                    rec.t,
                ),
                scatter_direction,
                r_in.tm,
                RAY_SECONDARY,
            )
            attenuation = mat.texture.value(rec.u, rec.v, rec.p, rec.normal)
            scatter_ok = 1
        elif mat.material_type == 1:
            reflected = (
                reflect(r_in.direction, rec.normal) + mat.fuzz * random_unit_vec3()
            )
            scattered = Ray(
                self.offset_ray_origin(
                    rec.p,
                    rec.normal,
                    rec.geo_normal,
                    reflected,
                    self.ray_epsilon[None],
                    rec.t,
                ),
                reflected,
                r_in.tm,
                RAY_SECONDARY,
            )
            attenuation = mat.texture.value(rec.u, rec.v, rec.p, rec.normal)
            scatter_ok = ti.select(tm.dot(scattered.direction, rec.normal) > 0.0, 1, 0)
        elif mat.material_type == 2:
            attenuation = vec3(1.0, 1.0, 1.0)
            ri = ti.select(
                rec.front_face, 1.0 / mat.refractive_index, mat.refractive_index
            )
            unit_direction = tm.normalize(r_in.direction)
            costheta = ti.min(tm.dot(-unit_direction, rec.normal), 1.0)
            sintheta = ti.sqrt(ti.max(1.0 - costheta * costheta, 0.0))
            r0 = (1.0 - ri) / (1.0 + ri)
            r0 = r0 * r0
            reflectance = r0 + (1.0 - r0) * ti.pow(1.0 - costheta, 5)
            cannot_refract = sintheta * ri > 1.0
            direction = vec3(0.0, 0.0, 0.0)
            if cannot_refract | (reflectance > ti.random()):
                direction = reflect(unit_direction, rec.normal)
            else:
                direction = refract(unit_direction, rec.normal, ri)
            scattered = Ray(
                self.offset_ray_origin(
                    rec.p,
                    rec.normal,
                    rec.geo_normal,
                    direction,
                    self.ray_epsilon[None],
                    rec.t,
                ),
                direction,
                r_in.tm,
                RAY_SECONDARY,
            )
            scatter_ok = 1
        return (scatter_ok, attenuation, scattered)

    @ti.func
    def trace_path(self, r: Ray, i: ti.i32, j: ti.i32) -> vec3:
        ray = r
        rad = vec3(0.0, 0.0, 0.0)
        attenuation = tm.vec3(1.0, 1.0, 1.0)
        after_diffuse = 0
        prev_p = ray.origin
        prev_n = vec3(0.0, 1.0, 0.0)
        depth = 0
        max_b = ti.min(self.max_bounces[None], self.TRACE_BOUNCE_CAP)
        alive = 1
        while (depth < max_b) & (alive != 0):
            if (
                self.bvh.hit(
                    ray, Interval(self.ray_epsilon[None], 1e30), self.hit_rec, i, j
                )
                != 0
            ):
                hr = self.hit_rec[i, j]
                mat_idx = ti.math.clamp(hr.mat_id, 0, self.material_count[None] - 1)
                mat = materials[mat_idx]
                if mat.material_type == 3:
                    emit_w = ti.select(
                        after_diffuse != 0,
                        self.bsdf_light_mis_weight(ray, prev_p, prev_n),
                        1.0,
                    )
                    rad += attenuation * mat.emitted * emit_w
                can_sample_direct = (
                    (self.use_nee[None] != 0)
                    & (self.nee_count[None] > 0)
                    & (depth < self.nee_max_depth[None])
                    & (mat.material_type == 0)
                    & (tm.max(attenuation.x, attenuation.y, attenuation.z) > 0.0001)
                )
                if can_sample_direct:
                    alb = mat.texture.value(hr.u, hr.v, hr.p, hr.normal)
                    rad += attenuation * self.sample_area_light(
                        hr.p, hr.normal, alb, i, j
                    )
                scatter_ok, scatter_atten, scatter_ray = self.scatter_material(
                    mat, ray, hr
                )
                if scatter_ok != 0:
                    attenuation *= scatter_atten
                    prev_p = hr.p
                    prev_n = hr.normal
                    ray = scatter_ray
                    after_diffuse = ti.select(mat.material_type == 0, 1, 0)
                    depth += 1
                    if depth >= 3:
                        p_survive = ti.math.clamp(
                            self.luminance(attenuation), 0.08, 0.95
                        )
                        if ti.random() > p_survive:
                            alive = 0
                        attenuation /= p_survive
                else:
                    alive = 0
            else:
                unit_dir = tm.normalize(ray.direction)
                a = 0.5 * (unit_dir.y + 1.0)
                sky = (1.0 - a) * tm.vec3(1.0) + a * tm.vec3(0.5, 0.7, 1.0)
                bg = ti.select(self.dark_background[None] != 0, vec3(0.0), sky)
                rad += attenuation * bg
                alive = 0
        return self.clamp_bright_pixels(rad)

    @ti.func
    def defocus_disk_sample(self):
        p = random_in_unit_disk()
        return (
            self.camera_center[None]
            + p.x * self.defocus_disk_u[None]
            + p.y * self.defocus_disk_v[None]
        )

    @ti.func
    def get_ray(self, i, j, sample: ti.i32):
        offset = vec3(random_double(0, 1) - 0.5, random_double(0, 1) - 0.5, 0)
        pixel_sample = (
            self.pixel00_loc[None]
            + (i + offset.x) * self.pixel_delta_u[None]
            + (j + offset.y) * self.pixel_delta_v[None]
        )
        ray_origin = ti.select(
            self.defocus_angle[None] <= 0,
            self.camera_center[None],
            self.defocus_disk_sample(),
        )
        ray_direction = pixel_sample - ray_origin
        ray_time = random_double(0, 1)
        return Ray(ray_origin, ray_direction, ray_time, RAY_PRIMARY)

    def _debug_taichi_return(self, label: str, t0: float | None = None) -> None:
        if not self.debug_mode:
            return
        ti.sync()
        elapsed = ""
        if t0 is not None:
            elapsed = f" ({pf() - t0:.3f}s)"
        debug_tools.log(f"returned to Python after Taichi {label}{elapsed}")

    def _debug_taichi_launch(self, label: str) -> None:
        if self.debug_mode:
            debug_tools.log(f"launch Taichi {label}")

    @ti.kernel
    def clear_accum(self):
        for i, j in self.color_accum:
            self.color_accum[i, j] = vec3(0.0, 0.0, 0.0)
            self.restir_curr_direct[i, j] = vec3(0.0, 0.0, 0.0)
            self.restir_reuse_direct[i, j] = vec3(0.0, 0.0, 0.0)

    @ti.kernel
    def trace_batch(self, batch_size: ti.i32):
        ti.loop_config(block_dim=128)
        for i, j in ti.ndrange(self.image_width, self.image_height):
            px = vec3(0.0, 0.0, 0.0)
            self.restir_curr_direct[i, j] = vec3(0.0, 0.0, 0.0)
            for s in range(batch_size):
                r = self.get_ray(i, j, s)
                px += self.trace_path(r, i, j)
            self.color_accum[i, j] += px

    @ti.kernel
    def restir_spatiotemporal_pass(self):
        ti.loop_config(block_dim=128)
        for i, j in ti.ndrange(self.image_width, self.image_height):
            c = self.restir_curr_direct[i, j]
            w = 1.0
            for ox, oy in ti.static(((1, 0), (-1, 0), (0, 1), (0, -1))):
                x = i + ox
                y = j + oy
                if (x >= 0) & (x < self.image_width) & (y >= 0) & (y < self.image_height):
                    nbh = self.restir_curr_direct[x, y]
                    cw = 1.0 / (1.0 + ti.abs(self.luminance(c) - self.luminance(nbh)) * 4.0)
                    c += nbh * cw
                    w += cw
            spatial = c / ti.max(w, 1e-06)
            t = ti.math.clamp(self.restir_temporal[None], 0.0, 0.95)
            self.restir_reuse_direct[i, j] = (1.0 - t) * spatial + t * self.restir_reuse_direct[i, j]

    @ti.kernel
    def render_offline_kernel(self, spp: ti.i32):
        ti.loop_config(block_dim=128)
        for i, j in ti.ndrange(self.image_width, self.image_height):
            px = vec3(0.0, 0.0, 0.0)
            for s in range(spp):
                r = self.get_ray(i, j, s)
                px += self.trace_path(r, i, j)
            linear = px / ti.cast(spp, ti.f32)
            mapped = tonemap_filmic(linear, self.exposure[None])
            self.img[i, self.image_height - 1 - j] = float_to_rgb8(mapped)

    @ti.kernel
    def write_image(self):
        for i, j in self.img:
            linear = self.color_accum[i, j] * self.pixel_sample_scale
            mapped = tonemap_filmic(linear, self.exposure[None])
            self.img[i, self.image_height - 1 - j] = float_to_rgb8(mapped)

    @ti.kernel
    def prepare_denoise_input(self):
        for i, j in self.denoise_ping:
            self.denoise_ping[i, j] = self.color_accum[i, j] * self.pixel_sample_scale

    @ti.kernel
    def atrous_pass(self, src: ti.template(), dst: ti.template(), step: ti.i32):
        ti.loop_config(block_dim=128)
        for i, j in ti.ndrange(self.image_width, self.image_height):
            center = src[i, j]
            acc = vec3(0.0, 0.0, 0.0)
            wsum = 0.0
            phi = ti.max(self.denoise_phi_color[None], 1e-04)
            for ox, oy in ti.static((
                (-2, -2), (-1, -2), (0, -2), (1, -2), (2, -2),
                (-2, -1), (-1, -1), (0, -1), (1, -1), (2, -1),
                (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0),
                (-2, 1), (-1, 1), (0, 1), (1, 1), (2, 1),
                (-2, 2), (-1, 2), (0, 2), (1, 2), (2, 2),
            )):
                x = ti.math.clamp(i + ox * step, 0, self.image_width - 1)
                y = ti.math.clamp(j + oy * step, 0, self.image_height - 1)
                sx = ti.abs(ox)
                sy = ti.abs(oy)
                wx = ti.select(sx == 0, 6.0, ti.select(sx == 1, 4.0, 1.0))
                wy = ti.select(sy == 0, 6.0, ti.select(sy == 1, 4.0, 1.0))
                sample = src[x, y]
                diff = sample - center
                edge = ti.exp(-tm.dot(diff, diff) / (phi * phi))
                wt = wx * wy * edge
                acc += sample * wt
                wsum += wt
            dst[i, j] = acc / ti.max(wsum, 1e-06)

    @ti.kernel
    def write_denoised_image(self, src: ti.template()):
        for i, j in self.img:
            mapped = tonemap_filmic(src[i, j], self.exposure[None])
            self.img[i, self.image_height - 1 - j] = float_to_rgb8(mapped)

    def render(self):
        t0 = pf()
        self._debug_taichi_launch("clear_accum")
        self.clear_accum()
        self._debug_taichi_return("clear_accum", t0)
        batch = max(1, int(self.offline_batch_size))
        done = 0
        while done < int(self.samples_per_pixel):
            step = min(batch, int(self.samples_per_pixel) - done)
            t0 = pf()
            self._debug_taichi_launch(
                f"trace_batch {done + step}/{int(self.samples_per_pixel)} spp"
            )
            self.trace_batch(step)
            done += step
            self._debug_taichi_return(
                f"trace_batch {done}/{int(self.samples_per_pixel)} spp", t0
            )
            if int(self.use_restir[None]) != 0:
                self._debug_taichi_launch("restir_spatiotemporal_pass")
                self.restir_spatiotemporal_pass()
                self._debug_taichi_return("restir_spatiotemporal_pass")
            if not self.debug_mode:
                print(
                    f"  completed {done}/{int(self.samples_per_pixel)} spp", flush=True
                )
        t0 = pf()
        if int(self.denoise_enabled[None]) != 0:
            self._debug_taichi_launch("denoise")
            self.prepare_denoise_input()
            iterations = max(1, min(6, int(self.denoise_iterations[None])))
            step_size = 1
            for pass_id in range(iterations):
                if pass_id % 2 == 0:
                    self.atrous_pass(self.denoise_ping, self.denoise_pong, step_size)
                else:
                    self.atrous_pass(self.denoise_pong, self.denoise_ping, step_size)
                step_size *= 2
            if iterations % 2 == 0:
                self.write_denoised_image(self.denoise_ping)
            else:
                self.write_denoised_image(self.denoise_pong)
            self._debug_taichi_return("denoise+write_image", t0)
        else:
            self._debug_taichi_launch("write_image")
            self.write_image()
            self._debug_taichi_return("write_image", t0)

    @ti.kernel
    def progressive_step_accum(self):
        for i, j in self.hdr_accum:
            b = self.live_batch[None]
            sx = 0.0
            sy = 0.0
            sz = 0.0
            for _s in range(b):
                r = self.get_ray(i, j, 0)
                c = self.trace_path(r, i, j)
                sx += c.x
                sy += c.y
                sz += c.z
            buf = self.hdr_accum[i, j]
            self.hdr_accum[i, j] = buf + tm.vec4(sx, sy, sz, ti.cast(b, ti.f32))

    @ti.kernel
    def tonemap_accum_to_display(self):
        for i, j in self.img:
            buf = self.hdr_accum[i, j]
            rgb = tm.vec3(buf[0], buf[1], buf[2]) / ti.max(buf[3], 0.0001)
            mapped = tonemap_filmic(rgb, self.exposure[None])
            self.img[i, self.image_height - 1 - j] = float_to_rgb8(mapped)


def build_scene_and_bvh(bvh_split: str = "sah"):
    checker_floor = create_checker_texture(
        vec3(0.2, 0.3, 0.1), vec3(0.9, 0.9, 0.9), 2.0
    )
    checker_sphere_texture = create_uv_checker_texture(
        vec3(1.0, 0.0, 0.0), vec3(0.0, 0.0, 1.0), 5
    )
    scene_spheres = [
        (point3(0, -100.5, -1), 100.0, create_lambertian(checker_floor)),
        (point3(-1, 0, -1), 0.5, create_lambertian(checker_sphere_texture)),
        (point3(0, 0, -1), 0.5, create_dielectric(1.5)),
        (point3(1, 0, -1), 0.5, create_metal(vec3(0.8, 0.6, 0.2), 0.0)),
    ]
    MAT_QUAD = 4
    MAT_MESH = 5
    MAT_ELLIPSOID = 6
    quad_mat = create_lambertian(create_solid_texture(vec3(0.2, 0.45, 0.35)))
    mesh_mat = create_lambertian(
        create_uv_checker_texture(vec3(0.9, 0.4, 0.1), vec3(0.15, 0.2, 0.9), 6)
    )
    ellipsoid_mat = create_lambertian(create_solid_texture(vec3(0.85, 0.25, 0.55)))
    mats_list = []
    for _, _, mat in scene_spheres:
        mats_list.append(mat)
    mats_list.append(quad_mat)
    mats_list.append(mesh_mat)
    mats_list.append(ellipsoid_mat)
    primitives: list[ScenePrimitive] = []
    for idx, (center, radius, _) in enumerate(scene_spheres):
        primitives.append(scene_primitive_sphere(center, center, radius, idx))
    primitives.append(
        scene_primitive_quad(
            point3(-2.2, -1.2, -1.35),
            point3(4.4, 0.0, 0.0),
            point3(0.0, 3.2, 0.0),
            MAT_QUAD,
        )
    )
    primitives.append(
        scene_primitive_ellipsoid(
            point3(0.0, 0.65, -1.15), vec3(0.22, 0.12, 0.18), MAT_ELLIPSOID
        )
    )
    root = os.path.dirname(os.path.abspath(__file__))
    mesh_triangles = []
    cube_path = os.path.join(root, "models", "cube.obj")
    if os.path.isfile(cube_path):
        _, mesh_triangles = load_mesh_triangles(
            cube_path, scale=0.18, translate=(0.0, 0.35, -0.92), max_triangles=12000
        )
        print(
            f"Loaded mesh: {len(mesh_triangles)} triangles from {cube_path}", flush=True
        )
    else:
        print(f"No mesh at {cube_path}; skipping OBJ", flush=True)
    prim_field = ScenePrimitive.field(shape=len(primitives), layout=ti.Layout.SOA)
    for i, p in enumerate(primitives):
        prim_field[i] = p
    print(
        f"Building PrimBVH ({bvh_split}) over {len(primitives)} analytic primitives + {len(mesh_triangles)} triangles...",
        flush=True,
    )
    bvh = PrimBVH(
        prim_field,
        len(primitives),
        split=bvh_split,
        host_primitives=primitives,
        host_triangles=mesh_triangles,
        triangle_mat_id=MAT_MESH,
        host_materials=mats_list,
    )
    print(
        f"BVH max node index ~{bvh.node_count}, ordered prims {bvh.prim_count}",
        flush=True,
    )
    print(f"BVH memory: {bvh.memory_summary()}", flush=True)
    default_cam = dict(
        lookfrom=point3(0, 1, 2),
        lookat=point3(0, 0, -1),
        vup=vec3(0, 1, 0),
        vfov=50.0,
        aspect_ratio=16.0 / 9.0,
        dark_background=0,
    )
    return (mats_list, bvh, root, default_cam)


SCENE_BUILDERS = dict(TEST_SCENE_BUILDERS)
SCENE_BUILDERS.setdefault("default", build_scene_and_bvh)
SCENE_DEFAULT_PNG = {name: f"{name}.png" for name in ALL_TEST_SCENE_NAMES}
SCENE_DEFAULT_PNG["default"] = "checker_scene.png"


def main():
    parser = argparse.ArgumentParser(
        description="Taichi GPU path tracer (BVH + analytic shapes)."
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Interactive progressive render (Taichi UI), like the official Taichi example.",
    )
    parser.add_argument(
        "--scene",
        default="default",
        help=f"Scene name ({len(ALL_TEST_SCENE_NAMES)} tests; use --list-scenes).",
    )
    parser.add_argument(
        "--list-scenes",
        action="store_true",
        help="Print table of all test scene names and exit.",
    )
    parser.add_argument(
        "--render-all-testscenes",
        action="store_true",
        help="Render all test scenes to output/testscenes/ (see --width/--spp).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output PNG path (default: output/<scene>.png).",
    )
    parser.add_argument(
        "--width", type=int, default=DEFAULT_IMAGE_WIDTH, help="Image width"
    )
    parser.add_argument(
        "--spp",
        type=int,
        default=64,
        help="Samples per pixel (offline). Cornell scene defaults to 256 if unset.",
    )
    parser.add_argument(
        "--no-nee",
        action="store_true",
        help="Disable next-event estimation (area lights); much noisier, for comparison only.",
    )
    parser.add_argument(
        "--bvh-split",
        choices=("auto", "sah", "median", "gpu","gpu_sah"),
        default="auto",
        help="BVH build: auto, SAH, centroid median, or gpu (Morton LBVH for large meshes).",
    )
    parser.add_argument(
        "--live-spp",
        type=int,
        default=4,
        help="Path samples per pixel per frame in --live mode (default 4; cap 32).",
    )
    parser.add_argument(
        "--offline-batch-size",
        type=int,
        default=8,
        help="Offline path-trace samples per Taichi launch.",
    )
    parser.add_argument(
        "--light-samples",
        type=int,
        default=0,
        help="Area-light samples per hit for NEE. 0 keeps the scene default; renderer cap is 8.",
    )
    parser.add_argument(
        "--nee-max-depth",
        type=int,
        default=-1,
        help="Max path depth that runs NEE direct-light shadow rays. -1 keeps the scene default.",
    )
    parser.add_argument(
        "--restir",
        action="store_true",
        help="Enable reservoir resampling and spatiotemporal reuse for direct area-light samples.",
    )
    parser.add_argument(
        "--restir-temporal",
        type=float,
        default=0.75,
        help="Temporal reuse strength for --restir, 0 disables history and 0.95 is the cap.",
    )
    parser.add_argument(
        "--denoise",
        action="store_true",
        help="Enable an edge-aware A-Trous wavelet denoiser before PNG output.",
    )
    parser.add_argument(
        "--denoise-iterations",
        type=int,
        default=4,
        help="A-Trous denoise passes, capped to 1..6.",
    )
    parser.add_argument(
        "--denoise-phi-color",
        type=float,
        default=0.12,
        help="À-Trous color edge threshold. Lower preserves edges, higher smooths more.",
    )
    parser.add_argument(
        "--ray-epsilon",
        type=float,
        default=0.0001,
        help="Closest-hit ray epsilon/bias for secondary rays.",
    )
    parser.add_argument(
        "--shadow-epsilon",
        type=float,
        default=0.0002,
        help="Shadow/light ray epsilon/bias used for NEE occlusion rays.",
    )
    parser.add_argument(
        "--flip-winding",
        action="store_true",
        help="Flip all mesh winding and normals at load time (for CW-authored assets).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print completed milestones and synchronize timed Taichi work.",
    )
    args = parser.parse_args()
    debug_tools.set_debug(args.debug)
    debug_tools.completed("debug mode enabled")
    with debug_tools.span("Taichi GPU startup probe"):
        _probe_taichi_backend()
    from src.geometry.mesh_io import set_default_flip_winding

    set_default_flip_winding(bool(getattr(args, "flip_winding", False)))
    if args.list_scenes:
        print_scene_table()
        return
    global materials
    if args.render_all_testscenes:
        if args.live:
            print("--render-all-testscenes cannot be used with --live", flush=True)
            return
        out_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "output", "testscenes"
        )
        os.makedirs(out_dir, exist_ok=True)
        total_t0 = pf()
        batch_w = args.width if args.width != 1920 else 1024
        batch_spp = args.spp if args.spp != 64 else 256
        for scene_name in ALL_TEST_SCENE_NAMES:
            scene_args = argparse.Namespace(**vars(args))
            scene_args.scene = scene_name
            scene_args.render_all_testscenes = False
            scene_args.width = batch_w
            scene_args.spp = batch_spp
            scene_args.output = os.path.join(out_dir, f"{scene_name}.png")
            print(
                f"\n=== [{scene_name}] {batch_w}px, {batch_spp} spp -> {scene_args.output} ===",
                flush=True,
            )
            _render_one_scene(scene_args)
        print(f"\nAll test scenes written to {out_dir}", flush=True)
        print(f"Total batch time: {pf() - total_t0:.1f}s", flush=True)
        return
    _render_one_scene(args)


def _render_one_scene(args):
    global materials
    debug_tools.log(
        f"render request: scene={args.scene}, width={args.width}, spp={args.spp}, bvh_split={args.bvh_split}, live={args.live}"
    )
    if args.scene not in SCENE_BUILDERS:
        print(f"Unknown scene '{args.scene}'. Use --list-scenes.", flush=True)
        return
    builder = SCENE_BUILDERS[args.scene]
    bvh_split = args.bvh_split
    if bvh_split == "auto":
        bvh_split = "gpu_sah" if args.scene == "complex" else "sah"
    debug_tools.log(f"resolved BVH split: {bvh_split}")
    with debug_tools.span(f"build scene '{args.scene}'"):
        mats_list, bvh, root, cam = builder(bvh_split)
    debug_tools.completed(
        f"scene loaded ({len(mats_list)} materials, {getattr(bvh, 'triangle_count', 0):,} mesh triangles)"
    )
    debug_tools.log(
        f"scene built: materials={len(mats_list)}, bvh_prims={getattr(bvh, 'bvh_prim_count', getattr(bvh, 'prim_count', '?'))}, nodes={getattr(bvh, 'node_count', '?')}"
    )
    with debug_tools.span("upload materials"):
        materials = Material.field(shape=len(mats_list), layout=ti.Layout.SOA)
        for i, m in enumerate(mats_list):
            materials[i] = m
    debug_tools.completed("materials uploaded")
    output_dir = os.path.join(root, "output")
    with debug_tools.span("prepare output directory"):
        os.makedirs(output_dir, exist_ok=True)
    lookfrom = cam["lookfrom"]
    lookat = cam["lookat"]
    vup = cam["vup"]
    vfov = cam["vfov"]
    aspect_ratio = float(cam["aspect_ratio"])
    dark_bg = int(cam.get("dark_background", 0))
    defocus_angle = 0.0
    lf = np.asarray([lookfrom[0], lookfrom[1], lookfrom[2]], dtype=np.float64)
    la = np.asarray([lookat[0], lookat[1], lookat[2]], dtype=np.float64)
    focus_distance = float(np.linalg.norm(lf - la))
    use_nee_host = not args.no_nee
    if args.live:
        w, h = (args.width, max(1, int(args.width / aspect_ratio)))
        live_spp = max(1, min(args.live_spp, 32))
        with debug_tools.span(f"create Camera {w}x{h} live"):
            camera = Camera(
                bvh,
                image_width=w,
                aspect_ratio=aspect_ratio,
                vfov=vfov,
                lookfrom=lookfrom,
                lookat=lookat,
                vup=vup,
                defocus_angle=defocus_angle,
                focus_distance=focus_distance,
                use_hdr_accum=True,
                dark_background=dark_bg,
                max_bounces=int(cam.get("max_bounces", 12)),
            )
        camera.debug_mode = bool(args.debug)
        with debug_tools.span("configure camera lights"):
            camera.set_lights(cam.get("nee_lights", []), use_nee_host)
        scene_light_samples = int(cam.get("light_samples", 4))
        camera.light_samples[None] = max(
            1, min(8, int(args.light_samples or scene_light_samples))
        )
        scene_nee_depth = int(cam.get("nee_max_depth", 4))
        camera.nee_max_depth[None] = max(
            0,
            min(
                12,
                int(args.nee_max_depth if args.nee_max_depth >= 0 else scene_nee_depth),
            ),
        )
        camera.ray_epsilon[None] = max(1e-06, float(args.ray_epsilon))
        camera.shadow_epsilon[None] = max(1e-06, float(args.shadow_epsilon))
        camera.use_restir[None] = 1 if args.restir else 0
        camera.restir_temporal[None] = max(0.0, min(0.95, float(args.restir_temporal)))
        camera.denoise_enabled[None] = 1 if args.denoise else 0
        camera.denoise_iterations[None] = max(1, min(6, int(args.denoise_iterations)))
        camera.denoise_phi_color[None] = max(1e-04, float(args.denoise_phi_color))
        camera.samples_per_pixel = 1
        camera.live_batch[None] = live_spp
        camera.hdr_accum.fill(0)
        window = ti.ui.Window(
            f"GPU path trace | NEE | {live_spp} spp/frame — close to stop",
            (w, h),
            vsync=True,
        )
        canvas = window.get_canvas()
        frame = 0
        t_last = pf()
        print(
            f"Live mode: NEE={('off' if args.no_nee else 'on')}, {live_spp} spp/frame, scene={args.scene}, BVH={args.bvh_split}. Close window to stop.",
            flush=True,
        )
        while window.running:
            t0 = pf()
            camera._debug_taichi_launch("progressive_step_accum")
            camera.progressive_step_accum()
            camera._debug_taichi_return("progressive_step_accum", t0)
            if int(camera.use_restir[None]) != 0:
                camera.restir_spatiotemporal_pass()
            t0 = pf()
            camera._debug_taichi_launch("tonemap_accum_to_display")
            camera.tonemap_accum_to_display()
            camera._debug_taichi_return("tonemap_accum_to_display", t0)
            canvas.set_image(camera.img)
            window.show()
            frame += 1
            if frame % 30 == 0:
                now = pf()
                dt = now - t_last
                t_last = now
                est_spp = frame * live_spp
                print(
                    f"  ~{est_spp} accum spp / pixel · {30.0 / max(dt, 1e-06):.1f} frames/s",
                    flush=True,
                )
        out = os.path.join(output_dir, "live_final.png")
        ti.tools.imwrite(camera.img.to_numpy(), out)
        print(f"Saved {out}", flush=True)
        return
    with debug_tools.span(
        f"create Camera {args.width}x{max(1, int(args.width / aspect_ratio))}"
    ):
        camera = Camera(
            bvh,
            image_width=args.width,
            aspect_ratio=aspect_ratio,
            vfov=vfov,
            lookfrom=lookfrom,
            lookat=lookat,
            vup=vup,
            defocus_angle=defocus_angle,
            focus_distance=focus_distance,
            use_hdr_accum=False,
            dark_background=dark_bg,
            max_bounces=int(cam.get("max_bounces", 12)),
        )
    camera.debug_mode = bool(args.debug)
    with debug_tools.span("configure camera lights"):
        camera.set_lights(cam.get("nee_lights", []), use_nee_host)
    spp = args.spp
    if args.spp == 64 and cam.get("default_spp"):
        spp = int(cam["default_spp"])
    camera.samples_per_pixel = spp
    camera.pixel_sample_scale = 1.0 / spp
    camera.offline_batch_size = max(1, int(args.offline_batch_size))
    camera.exposure[None] = float(cam.get("exposure", 0.85))
    scene_light_samples = int(cam.get("light_samples", 4))
    camera.light_samples[None] = max(
        1, min(8, int(args.light_samples or scene_light_samples))
    )
    scene_nee_depth = int(cam.get("nee_max_depth", 4))
    camera.nee_max_depth[None] = max(
        0,
        min(
            12, int(args.nee_max_depth if args.nee_max_depth >= 0 else scene_nee_depth)
        ),
    )
    camera.ray_epsilon[None] = max(1e-06, float(args.ray_epsilon))
    camera.shadow_epsilon[None] = max(1e-06, float(args.shadow_epsilon))
    camera.max_bounces[None] = int(cam.get("max_bounces", 12))
    camera.use_restir[None] = 1 if args.restir else 0
    camera.restir_temporal[None] = max(0.0, min(0.95, float(args.restir_temporal)))
    camera.denoise_enabled[None] = 1 if args.denoise else 0
    camera.denoise_iterations[None] = max(1, min(6, int(args.denoise_iterations)))
    camera.denoise_phi_color[None] = max(1e-04, float(args.denoise_phi_color))
    print(
        f"Rendering: scene={args.scene}, {spp} spp, NEE={('on' if use_nee_host else 'off')}, ReSTIR={('on' if args.restir else 'off')}, denoise={('on' if args.denoise else 'off')}, bounces={camera.max_bounces[None]}, light_samples={camera.light_samples[None]}, nee_depth={camera.nee_max_depth[None]}, batch={camera.offline_batch_size}...",
        flush=True,
    )
    render_t0 = pf()
    debug_tools.log("renderer path: batched path tracer")
    camera.render()
    render_t1 = pf()
    debug_tools.completed(f"path trace render ({render_t1 - render_t0:.3f}s)")
    if args.output:
        output_path = args.output
        out_parent = os.path.dirname(os.path.abspath(output_path))
        if out_parent:
            os.makedirs(out_parent, exist_ok=True)
    else:
        output_path = os.path.join(output_dir, SCENE_DEFAULT_PNG[args.scene])
    t0 = pf()
    debug_tools.log("copy image field to NumPy")
    pixels = camera.img.to_numpy()
    camera._debug_taichi_return("img.to_numpy", t0)
    with debug_tools.span(f"write PNG {output_path}"):
        ti.tools.imwrite(pixels, output_path)
    debug_tools.completed(f"saved {output_path}")
    print(f"Image saved to {output_path}", flush=True)
    print(f"Render took {render_t1 - render_t0:.3f}s", flush=True)


if __name__ == "__main__":
    main()
