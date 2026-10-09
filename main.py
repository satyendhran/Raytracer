import argparse
import json
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
from src.core.Rand import reflect, refract
from src.core.Ray import Ray
from src.core.render_core import sanitize_rgb, tonemap_filmic
from src.core.sampling import PathSampler, cosine_direction
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
    ADAPTIVE_TILE_SIZE = 16

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
        enable_restir: bool = False,
        enable_denoise: bool = False,
        enable_adaptive: bool = False,
    ):
        self.samples_per_pixel = 64
        self.pixel_sample_scale = 1.0 / self.samples_per_pixel
        self.offline_batch_size = 8
        self.debug_mode = False
        self.enable_restir = bool(enable_restir)
        self.enable_denoise = bool(enable_denoise)
        self.enable_adaptive = bool(enable_adaptive)
        self.adaptive_scheduler = "auto"
        self.enable_edge_buffers = self.enable_restir or self.enable_denoise
        self.sampler = PathSampler()
        self.sample_offset = ti.field(ti.i32, shape=())
        self.timings = {}
        self.denoised_result = None
        self.last_render_adaptive = False
        self.image_width = image_width
        self.image_height = max(1, int(image_width / aspect_ratio))
        self.adaptive_tile_cols = (
            self.image_width + self.ADAPTIVE_TILE_SIZE - 1
        ) // self.ADAPTIVE_TILE_SIZE
        self.adaptive_tile_rows = (
            self.image_height + self.ADAPTIVE_TILE_SIZE - 1
        ) // self.ADAPTIVE_TILE_SIZE
        self.adaptive_tile_count = self.adaptive_tile_cols * self.adaptive_tile_rows
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
        if self.enable_restir:
            self.restir_curr_direct = ti.Vector.field(
                3,
                ti.f32,
                shape=(self.image_width, self.image_height),
                layout=ti.Layout.SOA,
            )
            self.restir_reuse_direct = ti.Vector.field(
                3,
                ti.f32,
                shape=(self.image_width, self.image_height),
                layout=ti.Layout.SOA,
            )
        if self.enable_edge_buffers:
            self.edge_normal = ti.Vector.field(
                3,
                ti.f32,
                shape=(self.image_width, self.image_height),
                layout=ti.Layout.SOA,
            )
            self.edge_depth = ti.field(
                dtype=ti.f32, shape=(self.image_width, self.image_height)
            )
        if self.enable_denoise:
            self.edge_albedo = ti.Vector.field(
                3,
                ti.f32,
                shape=(self.image_width, self.image_height),
                layout=ti.Layout.SOA,
            )
        self.denoise_enabled = ti.field(dtype=ti.i32, shape=())
        self.denoise_enabled[None] = 0
        self.denoise_iterations = ti.field(dtype=ti.i32, shape=())
        self.denoise_iterations[None] = 3
        self.denoise_phi_color = ti.field(dtype=ti.f32, shape=())
        self.denoise_phi_color[None] = 0.055
        if self.enable_denoise:
            self.sample_m2 = ti.Vector.field(
                3, ti.f32, shape=(self.image_width, self.image_height), layout=ti.Layout.SOA
            )
            self.denoise_variance = ti.field(ti.f32, shape=(self.image_width, self.image_height))
            self.denoise_ping = ti.Vector.field(
                3,
                ti.f32,
                shape=(self.image_width, self.image_height),
                layout=ti.Layout.SOA,
            )
            self.denoise_pong = ti.Vector.field(
                3,
                ti.f32,
                shape=(self.image_width, self.image_height),
                layout=ti.Layout.SOA,
            )
        if self.enable_adaptive:
            self.adaptive_pixel_indices = ti.field(ti.i32, shape=(2, self.image_width * self.image_height))
            self.adaptive_stable = ti.field(ti.i32, shape=(self.image_width, self.image_height))
            self.adaptive_mean = ti.Vector.field(
                3,
                ti.f32,
                shape=(self.image_width, self.image_height),
                layout=ti.Layout.SOA,
            )
            self.adaptive_m2 = ti.Vector.field(
                3,
                ti.f32,
                shape=(self.image_width, self.image_height),
                layout=ti.Layout.SOA,
            )
            self.adaptive_count = ti.field(
                dtype=ti.i32, shape=(self.image_width, self.image_height)
            )
            self.adaptive_active = ti.field(
                dtype=ti.i32, shape=(self.image_width, self.image_height)
            )
            self.adaptive_active_pixels = ti.field(dtype=ti.i32, shape=())
            self.adaptive_tile_active = ti.field(
                dtype=ti.i32, shape=(max(self.adaptive_tile_count, 1),)
            )
            self.adaptive_tile_next_active = ti.field(
                dtype=ti.i32, shape=(max(self.adaptive_tile_count, 1),)
            )
            self.adaptive_tile_indices = ti.field(
                dtype=ti.i32, shape=(max(self.adaptive_tile_count, 1),)
            )
            self.adaptive_tile_next_indices = ti.field(
                dtype=ti.i32, shape=(max(self.adaptive_tile_count, 1),)
            )
            self.adaptive_next_tile_count = ti.field(dtype=ti.i32, shape=())
            self.adaptive_active_tiles = ti.field(dtype=ti.i32, shape=())
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
        prim_index = int(L.get("prim_id", -1))
        for index, prim in enumerate(getattr(self.bvh, "host_analytic_primitives", ())):
            if int(prim.kind) == KIND_QUAD:
                expected = np.asarray([L["Q"], L["u"], L["v"]], dtype=np.float32)
                actual = np.asarray([prim.center0, prim.center_vel, prim.box_min], dtype=np.float32)
                if np.array_equal(expected, actual):
                    prim_index = index
                    break
        self.nee_light_indices[slot] = prim_index
        self.nee_Q[slot] = tm.vec3(float(L["Q"][0]), float(L["Q"][1]), float(L["Q"][2]))
        self.nee_u[slot] = tm.vec3(float(L["u"][0]), float(L["u"][1]), float(L["u"][2]))
        self.nee_v[slot] = tm.vec3(float(L["v"][0]), float(L["v"][1]), float(L["v"][2]))
        self.nee_n[slot] = tm.vec3(float(L["n"][0]), float(L["n"][1]), float(L["n"][2]))
        self.nee_area[slot] = float(L["area"])
        self.nee_Le[slot] = tm.vec3(
            float(L["Le"][0]), float(L["Le"][1]), float(L["Le"][2])
        )

    def _upload_indexed_nee_light(self, slot: int, prim_index: int) -> None:
        if prim_index < 0 or prim_index >= self.bvh.analytic_count:
            raise ValueError(f"NEE light primitive index {prim_index} is out of range.")
        prim = self.bvh.host_analytic_primitives[prim_index]
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
        scale = ti.max(ti.max(pdf_a, pdf_b), 1e-30)
        a2 = (pdf_a / scale) ** 2
        b2 = (pdf_b / scale) ** 2
        return a2 / ti.max(a2 + b2, 1e-30)

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
        self, p: vec3, n: vec3, albedo: vec3, i: ti.i32, j: ti.i32,
        ray_time: ti.f32, sample_index: ti.i32, depth: ti.i32
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
                        pixel = i * self.image_height + j
                        light_index = sample_index * ns + s
                        stream = depth * self.NEE_LIGHT_CAPACITY + k + 1
                        su = self.sampler.sample(pixel, light_index, 29, stream)
                        sv = self.sampler.sample(pixel, light_index, 30, stream)
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
                        cos_x = ti.max(tm.dot(n, wi), 0.0)
                        cos_l = ti.abs(tm.dot(nL, -wi))
                        geom_ok = (cos_x > 1e-05) & (cos_l > 1e-05) & (dist > 0.0001)
                        if geom_ok:
                            sh = Ray(roff, wi, ray_time, RAY_SHADOW)
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
                                if depth + 1 >= ti.min(self.max_bounces[None], self.TRACE_BOUNCE_CAP):
                                    bsdf_pdf = 0.0
                                mis_w = self.power_heuristic(ns_f * light_pdf, bsdf_pdf)
                                geom = cos_x * cos_l * A / ti.max(dist2, 1e-06)
                                contrib = brdf * Le * geom * mis_w
                                if ti.static(self.enable_restir):
                                    if self.use_restir[None] != 0:
                                        w = ti.max(self.luminance(contrib), 1e-08)
                                        wsum += w
                                        m += 1
                                        if ti.random() < w / ti.max(wsum, 1e-08):
                                            selected = contrib
                                            selected_w = w
                                    else:
                                        acc += contrib
                                else:
                                    acc += contrib
        result = acc / ti.cast(ns, ti.f32)
        if ti.static(self.enable_restir):
            if self.use_restir[None] != 0:
                local = vec3(0.0, 0.0, 0.0)
                if m > 0:
                    local = selected * wsum / ti.max(
                        selected_w * ti.cast(ns, ti.f32), 1e-08
                    )
                result = local
                self.restir_curr_direct[i, j] = local
        return result

    @ti.func
    def clamp_bright_pixels(self, c: vec3) -> vec3:
        return sanitize_rgb(c)

    @ti.func
    def luminance(self, c: vec3) -> ti.f32:
        return 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z

    @ti.func
    def bsdf_light_mis_weight(
        self, ray: Ray, prev_p: vec3, prev_n: vec3, hit_p: vec3, hit_prim_id: ti.i32 = -1
    ) -> ti.f32:
        weight = 1.0
        if (self.use_nee[None] != 0) & (self.nee_count[None] > 0):
            best_pdf = 0.0
            for k in range(4):
                if k < self.nee_count[None]:
                    Q = self.nee_Q[k]
                    u = self.nee_u[k]
                    v = self.nee_v[k]
                    nL = self.nee_n[k]
                    direction = tm.normalize(ray.direction)
                    denom = tm.dot(nL, direction)
                    if ti.abs(denom) > 1e-08:
                        t = tm.dot(Q - prev_p, nL) / denom
                        if t > 0.0001:
                            hp = prev_p + direction * t
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
                                registered_id = self.nee_light_indices[k]
                                on_hit_light = tm.length(hp - hit_p) <= 0.000001 * ti.max(t, 1.0)
                                if registered_id >= 0:
                                    on_hit_light = registered_id == hit_prim_id
                                if (a >= 0.0) & (a <= 1.0) & (b >= 0.0) & (b <= 1.0) & on_hit_light:
                                    area = self.nee_area[k]
                                    dist2 = tm.dot(hp - prev_p, hp - prev_p)
                                    cos_l = ti.abs(tm.dot(nL, -direction))
                                    best_pdf = ti.max(
                                        best_pdf, dist2 / ti.max(cos_l * area, 1e-06)
                                    )
            cos_x = ti.max(tm.dot(prev_n, tm.normalize(ray.direction)), 0.0)
            bsdf_pdf = cos_x / ti.math.pi
            ns = ti.cast(ti.min(ti.max(self.light_samples[None], 1), 8), ti.f32)
            weight = self.power_heuristic(bsdf_pdf, ns * best_pdf)
        return weight

    @ti.func
    def scatter_material(self, mat: Material, r_in: Ray, rec, u: ti.f32, v: ti.f32):
        scatter_ok = 0
        attenuation = vec3(0.0, 0.0, 0.0)
        scattered = Ray(rec.p, rec.normal, r_in.tm, RAY_SECONDARY)
        if mat.material_type == 0:
            scatter_direction = cosine_direction(rec.normal, u, v)
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
            z = 2.0 * u - 1.0
            radius = ti.sqrt(ti.max(1.0 - z * z, 0.0))
            phi = 2.0 * ti.math.pi * v
            fuzz_direction = vec3(radius * ti.cos(phi), radius * ti.sin(phi), z)
            reflected = reflect(tm.normalize(r_in.direction), rec.normal) + mat.fuzz * fuzz_direction
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
            if cannot_refract | (reflectance > u):
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
    def trace_path(self, r: Ray, i: ti.i32, j: ti.i32, sample_index: ti.i32) -> vec3:
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
                if ti.static(self.enable_restir and not self.enable_denoise):
                    if depth == 0:
                        self.edge_depth[i, j] = hr.t
                        self.edge_normal[i, j] = hr.geo_normal
                mat_idx = ti.math.clamp(hr.mat_id, 0, self.material_count[None] - 1)
                mat = materials[mat_idx]
                if mat.material_type == 3:
                    emit_w = ti.select(
                        after_diffuse != 0,
                        self.bsdf_light_mis_weight(ray, prev_p, prev_n, hr.p, hr.prim_id),
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
                        hr.p, hr.normal, alb, i, j, ray.tm, sample_index, depth
                    )
                pixel = i * self.image_height + j
                u = self.sampler.sample(pixel, sample_index, 5 + 2 * depth, 0)
                v = self.sampler.sample(pixel, sample_index, 6 + 2 * depth, 0)
                scatter_ok, scatter_atten, scatter_ray = self.scatter_material(mat, ray, hr, u, v)
                if scatter_ok != 0:
                    attenuation *= scatter_atten
                    prev_p = hr.p
                    prev_n = hr.normal
                    ray = scatter_ray
                    after_diffuse = ti.cast(can_sample_direct, ti.i32)
                    depth += 1
                    if depth >= 3:
                        p_survive = ti.math.clamp(
                            self.luminance(attenuation), 0.08, 0.95
                        )
                        roulette = self.sampler.sample(pixel, sample_index, 31, depth)
                        if roulette > p_survive:
                            alive = 0
                        attenuation /= p_survive
                else:
                    alive = 0
            else:
                if ti.static(self.enable_restir and not self.enable_denoise):
                    if depth == 0:
                        self.edge_depth[i, j] = 1e20
                        self.edge_normal[i, j] = vec3(0.0, 0.0, 0.0)
                unit_dir = tm.normalize(ray.direction)
                a = 0.5 * (unit_dir.y + 1.0)
                sky = (1.0 - a) * tm.vec3(1.0) + a * tm.vec3(0.5, 0.7, 1.0)
                bg = ti.select(self.dark_background[None] != 0, vec3(0.0), sky)
                rad += attenuation * bg
                alive = 0
        return self.clamp_bright_pixels(rad)

    @ti.func
    def get_ray(self, i, j, sample: ti.i32):
        pixel = i * self.image_height + j
        offset = vec3(
            self.sampler.sample(pixel, sample, 0, 0) - 0.5,
            self.sampler.sample(pixel, sample, 1, 0) - 0.5, 0.0,
        )
        pixel_sample = (
            self.pixel00_loc[None]
            + (i + offset.x) * self.pixel_delta_u[None]
            + (j + offset.y) * self.pixel_delta_v[None]
        )
        ray_origin = self.camera_center[None]
        if self.defocus_angle[None] > 0.0:
            radius = ti.sqrt(self.sampler.sample(pixel, sample, 2, 0))
            angle = 2.0 * ti.math.pi * self.sampler.sample(pixel, sample, 3, 0)
            ray_origin += radius * (ti.cos(angle) * self.defocus_disk_u[None] + ti.sin(angle) * self.defocus_disk_v[None])
        ray_direction = pixel_sample - ray_origin
        ray_time = self.sampler.sample(pixel, sample, 4, 0)
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
        self.sample_offset[None] = 0
        for i, j in self.color_accum:
            self.color_accum[i, j] = vec3(0.0, 0.0, 0.0)
            if ti.static(self.enable_restir):
                self.restir_curr_direct[i, j] = vec3(0.0, 0.0, 0.0)
                self.restir_reuse_direct[i, j] = vec3(0.0, 0.0, 0.0)
            if ti.static(self.enable_edge_buffers):
                self.edge_depth[i, j] = 1e20
                self.edge_normal[i, j] = vec3(0.0, 0.0, 0.0)
            if ti.static(self.enable_denoise):
                self.edge_albedo[i, j] = vec3(0.0, 0.0, 0.0)
                self.sample_m2[i, j] = vec3(0.0)

    @ti.kernel
    def trace_batch(self, batch_size: ti.i32):
        ti.loop_config(block_dim=128)
        for i, j in ti.ndrange(self.image_width, self.image_height):
            px = vec3(0.0, 0.0, 0.0)
            mean = vec3(0.0)
            m2 = vec3(0.0)
            if ti.static(self.enable_denoise):
                mean = self.color_accum[i, j] / ti.max(ti.cast(self.sample_offset[None], ti.f32), 1.0)
                m2 = self.sample_m2[i, j]
            if ti.static(self.enable_restir):
                self.restir_curr_direct[i, j] = vec3(0.0, 0.0, 0.0)
            if ti.static(self.enable_restir and not self.enable_denoise):
                self.edge_depth[i, j] = 1e20
                self.edge_normal[i, j] = vec3(0.0, 0.0, 0.0)
            for s in range(batch_size):
                index = self.sample_offset[None] + s
                r = self.get_ray(i, j, index)
                value = self.trace_path(r, i, j, index)
                px += value
                if ti.static(self.enable_denoise):
                    delta = value - mean
                    mean += delta / ti.cast(index + 1, ti.f32)
                    m2 += delta * (value - mean)
            self.color_accum[i, j] += px
            if ti.static(self.enable_denoise):
                self.sample_m2[i, j] = m2
        self.sample_offset[None] += batch_size

    @ti.kernel
    def restir_spatiotemporal_pass(self):
        ti.loop_config(block_dim=128)
        for i, j in ti.ndrange(self.image_width, self.image_height):
            c = self.restir_curr_direct[i, j]
            w = 1.0
            d0 = self.edge_depth[i, j]
            n0 = self.edge_normal[i, j]
            for ox, oy in ti.static(((1, 0), (-1, 0), (0, 1), (0, -1))):
                x = i + ox
                y = j + oy
                if (x >= 0) & (x < self.image_width) & (y >= 0) & (y < self.image_height):
                    nbh = self.restir_curr_direct[x, y]
                    d1 = self.edge_depth[x, y]
                    n1 = self.edge_normal[x, y]
                    same_kind = ti.abs(d0 - d1) < 1e10
                    depth_w = ti.exp(-ti.abs(d0 - d1) / ti.max(0.02 * ti.max(d0, 1.0), 1e-04))
                    normal_w = ti.math.clamp(tm.dot(n0, n1), 0.0, 1.0)
                    normal_w = normal_w * normal_w * normal_w * normal_w
                    color_w = 1.0 / (1.0 + ti.abs(self.luminance(c) - self.luminance(nbh)) * 8.0)
                    cw = ti.select(same_kind, depth_w * normal_w * color_w, 0.0)
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
                px += self.trace_path(r, i, j, s)
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
            count = ti.cast(self.sample_offset[None], ti.f32)
            variance = tm.max(self.sample_m2[i, j], vec3(0.0)) / ti.max(count * (count - 1.0), 1.0)
            self.denoise_variance[i, j] = tm.max(variance.x, ti.max(variance.y, variance.z))

    @ti.kernel
    def prepare_adaptive_denoise_input(self):
        for i, j in self.denoise_ping:
            self.denoise_ping[i, j] = self.adaptive_mean[i, j]
            count = ti.cast(self.adaptive_count[i, j], ti.f32)
            variance = tm.max(self.adaptive_m2[i, j], vec3(0.0)) / ti.max(count * (count - 1.0), 1.0)
            self.denoise_variance[i, j] = tm.max(variance.x, ti.max(variance.y, variance.z))

    @ti.kernel
    def prepare_surface_guides(self):
        ti.loop_config(block_dim=128)
        for i, j in ti.ndrange(self.image_width, self.image_height):
            center = self.camera_center[None]
            target = self.pixel00_loc[None] + i * self.pixel_delta_u[None] + j * self.pixel_delta_v[None]
            ray = Ray(center, tm.normalize(target - center), 0.5, RAY_PRIMARY)
            self.edge_depth[i, j] = 1e20
            self.edge_normal[i, j] = vec3(0.0)
            self.edge_albedo[i, j] = vec3(0.0)
            if self.bvh.hit(ray, Interval(self.ray_epsilon[None], 1e30), self.hit_rec, i, j) != 0:
                hr = self.hit_rec[i, j]
                self.edge_depth[i, j] = hr.t
                self.edge_normal[i, j] = hr.normal
                mat = materials[ti.math.clamp(hr.mat_id, 0, self.material_count[None] - 1)]
                guide = vec3(1.0)
                if mat.material_type == 0 or mat.material_type == 1:
                    guide = mat.texture.value(hr.u, hr.v, hr.p, hr.normal)
                elif mat.material_type == 3:
                    guide = vec3(0.0)
                self.edge_albedo[i, j] = guide

    @ti.kernel
    def atrous_pass(self, src: ti.template(), dst: ti.template(), step: ti.i32):
        ti.loop_config(block_dim=128)
        for i, j in ti.ndrange(self.image_width, self.image_height):
            center = src[i, j]
            acc = vec3(0.0, 0.0, 0.0)
            wsum = 0.0
            phi = ti.max(self.denoise_phi_color[None], 1e-04)
            d0 = self.edge_depth[i, j]
            n0 = self.edge_normal[i, j]
            a0 = self.edge_albedo[i, j]
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
                d1 = self.edge_depth[x, y]
                n1 = self.edge_normal[x, y]
                a1 = self.edge_albedo[x, y]
                env_pair = (d0 > 1e10) & (d1 > 1e10)
                same_kind = env_pair | (ti.abs(d0 - d1) < 1e10)
                variance = self.denoise_variance[i, j] + self.denoise_variance[x, y]
                color_w = ti.exp(-tm.dot(diff, diff) / (phi * phi + 4.0 * variance + 1e-10))
                depth_w = ti.exp(-ti.abs(d0 - d1) / ti.max(0.015 * ti.max(d0, 1.0), 1e-04))
                normal_w = ti.math.clamp(tm.dot(n0, n1), 0.0, 1.0)
                normal_w = ti.select(env_pair, 1.0, normal_w * normal_w * normal_w * normal_w)
                albedo_diff = a1 - a0
                albedo_w = ti.exp(-tm.dot(albedo_diff, albedo_diff) / 0.04)
                albedo_w = ti.select(env_pair, 1.0, albedo_w)
                edge = ti.select(same_kind, color_w * depth_w * normal_w * albedo_w, 0.0)
                wt = wx * wy * edge
                acc += sample * wt
                wsum += wt
            dst[i, j] = acc / ti.max(wsum, 1e-06)

    @ti.kernel
    def write_denoised_image(self, src: ti.template()):
        for i, j in self.img:
            mapped = tonemap_filmic(src[i, j], self.exposure[None])
            self.img[i, self.image_height - 1 - j] = float_to_rgb8(mapped)

    @ti.kernel
    def clear_adaptive(self):
        self.adaptive_active_pixels[None] = self.image_width * self.image_height
        self.adaptive_active_tiles[None] = self.adaptive_tile_count
        for i, j in ti.ndrange(self.image_width, self.image_height):
            self.adaptive_mean[i, j] = vec3(0.0, 0.0, 0.0)
            self.adaptive_m2[i, j] = vec3(0.0, 0.0, 0.0)
            self.adaptive_count[i, j] = 0
            self.adaptive_active[i, j] = 1
            self.adaptive_stable[i, j] = 0
            pixel = i * self.image_height + j
            self.adaptive_pixel_indices[0, pixel] = pixel
            if ti.static(self.enable_restir):
                self.restir_curr_direct[i, j] = vec3(0.0, 0.0, 0.0)
                self.restir_reuse_direct[i, j] = vec3(0.0, 0.0, 0.0)
            if ti.static(self.enable_edge_buffers):
                self.edge_depth[i, j] = 1e20
                self.edge_normal[i, j] = vec3(0.0, 0.0, 0.0)
            if ti.static(self.enable_denoise):
                self.edge_albedo[i, j] = vec3(0.0, 0.0, 0.0)
        for tile in range(self.adaptive_tile_count):
            self.adaptive_tile_active[tile] = 1
            self.adaptive_tile_next_active[tile] = 1
            self.adaptive_tile_indices[tile] = tile
            self.adaptive_tile_next_indices[tile] = tile
        self.adaptive_next_tile_count[None] = self.adaptive_tile_count

    @ti.kernel
    def compact_active_pixels(self):
        self.adaptive_active_pixels[None] = 0
        for i, j in self.adaptive_active:
            if self.adaptive_active[i, j] != 0:
                slot = ti.atomic_add(self.adaptive_active_pixels[None], 1)
                self.adaptive_pixel_indices[0, slot] = i * self.image_height + j

    @ti.func
    def trace_adaptive_pixel(self, i, j, batch_size, min_spp, max_spp, threshold):
        if ti.static(self.enable_restir):
            self.restir_curr_direct[i, j] = vec3(0.0)
        count = self.adaptive_count[i, j]
        mean = self.adaptive_mean[i, j]
        m2 = self.adaptive_m2[i, j]
        for s in range(ti.min(batch_size, max_spp - count)):
            index = count
            r = self.get_ray(i, j, index)
            value = self.trace_path(r, i, j, index)
            count += 1
            delta = value - mean
            mean += delta / ti.cast(count, ti.f32)
            m2 += delta * (value - mean)
        keep_sampling = count < max_spp
        if count >= min_spp:
            variance = tm.max(m2 / ti.max(ti.cast(count - 1, ti.f32), 1.0), vec3(0.0))
            stderr_rgb = ti.sqrt(tm.max(variance.x, ti.max(variance.y, variance.z)) / ti.cast(count, ti.f32))
            stderr_luma = ti.sqrt(self.luminance(variance) / ti.cast(count, ti.f32))
            mean_rgb = ti.max(ti.max(mean.x, ti.max(mean.y, mean.z)), 0.05)
            mean_luma = ti.max(self.luminance(mean), 0.05)
            err = ti.max(stderr_rgb / mean_rgb, stderr_luma / mean_luma)
            stable = ti.select(err <= threshold, self.adaptive_stable[i, j] + 1, 0)
            self.adaptive_stable[i, j] = stable
            keep_sampling = keep_sampling & (stable < 2)
        self.adaptive_count[i, j] = count
        self.adaptive_mean[i, j] = mean
        self.adaptive_m2[i, j] = m2
        self.adaptive_active[i, j] = ti.cast(keep_sampling, ti.i32)
        return keep_sampling

    @ti.kernel
    def trace_adaptive_pixel_batch(
        self, read_queue: ti.i32, active_count: ti.i32,
        batch_size: ti.i32, min_spp: ti.i32, max_spp: ti.i32, threshold: ti.f32,
    ):
        self.adaptive_active_pixels[None] = 0
        ti.loop_config(block_dim=128)
        for slot in range(active_count):
            pixel = self.adaptive_pixel_indices[read_queue, slot]
            i = pixel // self.image_height
            j = pixel % self.image_height
            if self.trace_adaptive_pixel(i, j, batch_size, min_spp, max_spp, threshold):
                next_slot = ti.atomic_add(self.adaptive_active_pixels[None], 1)
                self.adaptive_pixel_indices[1 - read_queue, next_slot] = pixel

    @ti.kernel
    def trace_adaptive_tile_batch(
        self,
        active_tile_count: ti.i32,
        batch_size: ti.i32,
        min_spp: ti.i32,
        max_spp: ti.i32,
        threshold: ti.f32,
    ):
        ti.loop_config(block_dim=128)
        self.adaptive_active_pixels[None] = 0
        self.adaptive_next_tile_count[None] = 0
        for slot in range(active_tile_count):
            tile = self.adaptive_tile_indices[slot]
            self.adaptive_tile_next_active[tile] = 0
        for slot, lane in ti.ndrange(
            active_tile_count,
            ti.static(self.ADAPTIVE_TILE_SIZE * self.ADAPTIVE_TILE_SIZE),
        ):
            tile = self.adaptive_tile_indices[slot]
            tx = tile % self.adaptive_tile_cols
            ty = tile // self.adaptive_tile_cols
            ox = lane % ti.static(self.ADAPTIVE_TILE_SIZE)
            oy = lane // ti.static(self.ADAPTIVE_TILE_SIZE)
            i = tx * ti.static(self.ADAPTIVE_TILE_SIZE) + ox
            j = ty * ti.static(self.ADAPTIVE_TILE_SIZE) + oy
            if (i < self.image_width) & (j < self.image_height):
                active = self.adaptive_active[i, j]
                if active != 0:
                    self.trace_adaptive_pixel(i, j, batch_size, min_spp, max_spp, threshold)
                if self.adaptive_active[i, j] != 0:
                    ti.atomic_add(self.adaptive_active_pixels[None], 1)
                    if ti.atomic_max(self.adaptive_tile_next_active[tile], 1) == 0:
                        next_slot = ti.atomic_add(self.adaptive_next_tile_count[None], 1)
                        self.adaptive_tile_next_indices[next_slot] = tile
        next_count = self.adaptive_next_tile_count[None]
        for slot in range(next_count):
            tile = self.adaptive_tile_next_indices[slot]
            self.adaptive_tile_indices[slot] = tile
            self.adaptive_tile_next_active[tile] = 0
        self.adaptive_active_tiles[None] = next_count

    @ti.kernel
    def write_adaptive_image(self):
        for i, j in self.img:
            mapped = tonemap_filmic(self.adaptive_mean[i, j], self.exposure[None])
            self.img[i, self.image_height - 1 - j] = float_to_rgb8(mapped)

    def finish_image(self, adaptive=False):
        ti.sync()
        self.denoised_result = None
        self.last_render_adaptive = adaptive
        t0 = pf()
        if self.enable_denoise and int(self.denoise_enabled[None]) != 0:
            print("  cleaning final image...", flush=True)
            self.prepare_surface_guides()
            if adaptive:
                self.prepare_adaptive_denoise_input()
            else:
                self.prepare_denoise_input()
            src, dst = self.denoise_ping, self.denoise_pong
            iterations = max(1, min(6, int(self.denoise_iterations[None])))
            for pass_id in range(iterations):
                self.atrous_pass(src, dst, 1 << pass_id)
                src, dst = dst, src
            self.denoised_result = src
            self.write_denoised_image(src)
        elif adaptive:
            self.write_adaptive_image()
        else:
            self.write_image()
        ti.sync()
        self.timings["filter_and_display_seconds"] = pf() - t0

    def linear_image(self):
        if self.last_render_adaptive:
            return self.adaptive_mean.to_numpy()
        return self.color_accum.to_numpy() * self.pixel_sample_scale

    def render(self):
        trace_t0 = pf()
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
            if not self.debug_mode:
                ti.sync()
                print(
                    f"  completed {done}/{int(self.samples_per_pixel)} spp", flush=True
                )
        ti.sync()
        self.timings["trace_seconds"] = pf() - trace_t0
        self.finish_image()

    def render_adaptive(self, min_spp: int, max_spp: int, threshold: float):
        trace_t0 = pf()
        t0 = pf()
        self._debug_taichi_launch("clear_adaptive")
        self.clear_adaptive()
        self._debug_taichi_return("clear_adaptive", t0)
        base_batch = max(1, int(self.offline_batch_size))
        pass_id = 0
        use_pixels = self.adaptive_scheduler == "pixels"
        read_queue = 0
        active_tiles = int(self.adaptive_active_tiles[None])
        active_pixels = int(self.adaptive_active_pixels[None])
        total_pixels = max(1, self.image_width * self.image_height)
        while active_pixels > 0:
            active_pixel_ratio = active_pixels / total_pixels
            if self.adaptive_scheduler == "auto" and not use_pixels:
                tile_slots = max(1, active_tiles * self.ADAPTIVE_TILE_SIZE ** 2)
                if active_pixels / tile_slots < 0.4:
                    self.compact_active_pixels()
                    read_queue = 0
                    use_pixels = True
            batch = base_batch
            if active_pixel_ratio < 0.15:
                batch = min(max_spp, base_batch * 2)
            t0 = pf()
            if use_pixels:
                self.trace_adaptive_pixel_batch(
                    read_queue, active_pixels, batch, min_spp, max_spp, threshold
                )
                read_queue = 1 - read_queue
            else:
                self.trace_adaptive_tile_batch(
                    active_tiles, batch, min_spp, max_spp, threshold
                )
            self._debug_taichi_return("adaptive sampling", t0)
            pass_id += 1
            active_pixels = int(self.adaptive_active_pixels[None])
            if not use_pixels:
                active_tiles = int(self.adaptive_active_tiles[None])
            if not self.debug_mode and (pass_id == 1 or pass_id % 4 == 0):
                done_pct = 100.0 * (1.0 - active_pixels / total_pixels)
                if active_pixels > 0:
                    print(
                        f"  sampling finished for {done_pct:.1f}% ({active_pixels:,} pixels remaining)",
                        flush=True,
                    )
        if not self.debug_mode:
            print("  sampling finished (noise target or sample limit reached)", flush=True)
        ti.sync()
        self.timings["trace_seconds"] = pf() - trace_t0
        self.finish_image(adaptive=True)

    @ti.kernel
    def progressive_step_accum(self):
        for i, j in self.hdr_accum:
            b = self.live_batch[None]
            sx = 0.0
            sy = 0.0
            sz = 0.0
            for _s in range(b):
                index = ti.cast(self.hdr_accum[i, j].w, ti.i32) + _s
                r = self.get_ray(i, j, index)
                c = self.trace_path(r, i, j, index)
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


def _scene_export_path(path, scene):
    if not path:
        return ""
    stem, extension = os.path.splitext(path)
    return f"{stem}_{scene}{extension}"


def _quality_settings(args, scene_spp: int):
    quality = getattr(args, "quality", "normal")
    user_set_spp = args.spp is not None
    requested_spp = max(1, int(args.spp if user_set_spp else scene_spp))
    if quality == "manual":
        return dict(
            adaptive=False,
            min_spp=max(1, min(requested_spp, 16)),
            max_spp=max(1, requested_spp),
            threshold=0.018,
            denoise=bool(args.denoise),
            batch=max(1, int(args.offline_batch_size)),
            light_samples=0,
            nee_max_depth=-1,
            denoise_iterations=int(args.denoise_iterations),
            denoise_phi_color=float(args.denoise_phi_color),
        )
    if quality == "quick":
        max_spp = requested_spp if user_set_spp else min(scene_spp, 96)
        return dict(
            adaptive=not bool(args.no_adaptive),
            min_spp=min(max_spp, 12),
            max_spp=max_spp,
            threshold=0.04,
            denoise=True,
            batch=max(32, int(args.offline_batch_size)),
            light_samples=1,
            nee_max_depth=2,
            denoise_iterations=4,
            denoise_phi_color=0.085,
        )
    if quality == "final":
        max_spp = requested_spp if user_set_spp else scene_spp
        return dict(
            adaptive=False,
            min_spp=max_spp,
            max_spp=max_spp,
            threshold=0.0,
            denoise=bool(args.denoise),
            batch=max(1, int(args.offline_batch_size)),
            light_samples=0,
            nee_max_depth=-1,
            denoise_iterations=int(args.denoise_iterations),
            denoise_phi_color=float(args.denoise_phi_color),
        )
    max_spp = requested_spp if user_set_spp else min(scene_spp, 320)
    return dict(
        adaptive=not bool(args.no_adaptive),
        min_spp=min(max_spp, 64),
        max_spp=max_spp,
        threshold=0.016,
        denoise=True,
        batch=max(32, int(args.offline_batch_size)),
        light_samples=1,
        nee_max_depth=12,
        denoise_iterations=4,
        denoise_phi_color=0.07,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Raytracer."
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
        default=None,
        help="Samples per pixel. When omitted, scene and quality defaults apply.",
    )
    parser.add_argument(
        "--sampler", choices=("sobol", "random"), default="sobol",
        help="Sample pattern for camera rays, lights and path directions.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Repeatable sampling seed.",
    )
    parser.add_argument(
        "--stats", default="", help="Optional render timing and sample report in JSON.",
    )
    parser.add_argument(
        "--benchmark-runs", type=int, default=1,
        help="Repeat an offline render in one session to measure prepared-kernel performance.",
    )
    parser.add_argument(
        "--linear-output", default="", help="Optional raw and filtered linear image archive (.npz).",
    )
    parser.add_argument(
        "--quality",
        choices=("quick", "normal", "final", "manual"),
        default="normal",
        help="Render style. quick is fast, normal is clean, final uses every requested sample, manual follows your flags.",
    )
    parser.add_argument(
        "--no-adaptive",
        action="store_true",
        help="Use the same sample count for every pixel.",
    )
    parser.add_argument(
        "--adaptive-scheduler", choices=("auto", "pixels", "tiles"), default="auto",
        help="Choose the work queue automatically, or use pixels or tiles throughout.",
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
        help="Area-light samples per hit. 0 uses the quality or scene default; cap is 8.",
    )
    parser.add_argument(
        "--nee-max-depth",
        type=int,
        default=-1,
        help="Max path depth for direct-light samples. -1 uses the quality or scene default.",
    )
    parser.add_argument(
        "--restir",
        action="store_true",
        help="Enable experimental local reservoir sampling for direct area lights.",
    )
    parser.add_argument(
        "--restir-temporal",
        type=float,
        default=0.75,
        help="Legacy option. Temporal reuse is disabled to preserve lighting energy.",
    )
    parser.add_argument(
        "--denoise",
        action="store_true",
        help="Enable an edge-aware A-Trous wavelet denoiser before PNG output.",
    )
    parser.add_argument(
        "--denoise-iterations",
        type=int,
        default=3,
        help="A-Trous denoise passes, capped to 1..6.",
    )
    parser.add_argument(
        "--denoise-phi-color",
        type=float,
        default=0.055,
        help="A-Trous color edge threshold. Lower preserves edges, higher smooths more.",
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
    if args.width < 1 or (args.spp is not None and args.spp < 1):
        parser.error("Width and sample count must be positive.")
    if args.benchmark_runs < 1:
        parser.error("Benchmark runs must be positive.")
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
        batch_spp = args.spp if args.spp is not None else 256
        for scene_name in ALL_TEST_SCENE_NAMES:
            scene_args = argparse.Namespace(**vars(args))
            scene_args.scene = scene_name
            scene_args.render_all_testscenes = False
            scene_args.width = batch_w
            scene_args.spp = batch_spp
            scene_args.output = os.path.join(out_dir, f"{scene_name}.png")
            scene_args.stats = _scene_export_path(args.stats, scene_name)
            scene_args.linear_output = _scene_export_path(args.linear_output, scene_name)
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
    total_t0 = pf()
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
                enable_restir=bool(args.restir),
                enable_denoise=bool(args.denoise),
                enable_adaptive=False,
            )
        camera.debug_mode = bool(args.debug)
        camera.sampler.mode[None] = int(args.sampler == "sobol")
        camera.sampler.seed[None] = args.seed & 0xFFFFFFFF
        with debug_tools.span("configure camera lights"):
            camera.set_lights(cam.get("nee_lights", []), use_nee_host)
        scene_light_samples = int(cam.get("light_samples", 4))
        default_light_samples = 1 if args.restir and args.light_samples == 0 else scene_light_samples
        camera.light_samples[None] = max(
            1, min(8, int(args.light_samples or default_light_samples))
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
        camera.exposure[None] = float(cam.get("exposure", 0.85))
        camera.live_batch[None] = live_spp
        camera.hdr_accum.fill(0)
        window = ti.ui.Window(
            f"Raytracer GUI | {args.scene}",
            (w, h),
            vsync=True,
        )
        canvas = window.get_canvas()
        frame = 0
        t_last = pf()
        print(
            f"Raytracer GUI: {args.scene}, {live_spp} samples each frame. Close the window to save.",
            flush=True,
        )
        while window.running:
            t0 = pf()
            camera._debug_taichi_launch("progressive_step_accum")
            camera.progressive_step_accum()
            camera._debug_taichi_return("progressive_step_accum", t0)
            if camera.enable_restir and int(camera.use_restir[None]) != 0:
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
                    f"  about {est_spp} samples per pixel, {30.0 / max(dt, 1e-06):.1f} frames/s",
                    flush=True,
                )
        out = os.path.join(output_dir, "live_final.png")
        ti.tools.imwrite(camera.img.to_numpy(), out)
        print(f"Saved {out}", flush=True)
        return
    with debug_tools.span(
        f"create Camera {args.width}x{max(1, int(args.width / aspect_ratio))}"
    ):
        scene_spp = args.spp if args.spp is not None else int(cam.get("default_spp", 64))
        quality = _quality_settings(args, int(scene_spp))
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
            enable_restir=bool(args.restir),
            enable_denoise=bool(quality["denoise"]),
            enable_adaptive=bool(quality["adaptive"]),
        )
    camera.debug_mode = bool(args.debug)
    camera.sampler.mode[None] = int(args.sampler == "sobol")
    camera.sampler.seed[None] = args.seed & 0xFFFFFFFF
    camera.adaptive_scheduler = args.adaptive_scheduler
    with debug_tools.span("configure camera lights"):
        camera.set_lights(cam.get("nee_lights", []), use_nee_host)
    spp = int(quality["max_spp"])
    camera.samples_per_pixel = spp
    camera.pixel_sample_scale = 1.0 / spp
    camera.offline_batch_size = max(1, int(quality["batch"]))
    camera.exposure[None] = float(cam.get("exposure", 0.85))
    scene_light_samples = int(cam.get("light_samples", 4))
    quality_light_samples = int(quality.get("light_samples", 0))
    default_light_samples = quality_light_samples or scene_light_samples
    if args.restir and args.light_samples == 0:
        default_light_samples = 1
    camera.light_samples[None] = max(
        1, min(8, int(args.light_samples or default_light_samples))
    )
    scene_nee_depth = int(cam.get("nee_max_depth", 4))
    quality_nee_depth = int(quality.get("nee_max_depth", -1))
    default_nee_depth = quality_nee_depth if quality_nee_depth >= 0 else scene_nee_depth
    camera.nee_max_depth[None] = max(
        0,
        min(
            12, int(args.nee_max_depth if args.nee_max_depth >= 0 else default_nee_depth)
        ),
    )
    camera.ray_epsilon[None] = max(1e-06, float(args.ray_epsilon))
    camera.shadow_epsilon[None] = max(1e-06, float(args.shadow_epsilon))
    camera.max_bounces[None] = int(cam.get("max_bounces", 12))
    camera.use_restir[None] = 1 if args.restir else 0
    camera.restir_temporal[None] = max(0.0, min(0.95, float(args.restir_temporal)))
    camera.denoise_enabled[None] = 1 if quality["denoise"] else 0
    camera.denoise_iterations[None] = max(
        1, min(6, int(quality.get("denoise_iterations", args.denoise_iterations)))
    )
    camera.denoise_phi_color[None] = max(
        1e-04, float(quality.get("denoise_phi_color", args.denoise_phi_color))
    )
    mode_text = "adaptive" if quality["adaptive"] else "fixed"
    print(
        f"Rendering {args.scene}: {args.quality} quality, {mode_text}, up to {spp} samples, light samples={camera.light_samples[None]}, NEE depth={camera.nee_max_depth[None]}, denoise={('on' if quality['denoise'] else 'off')}.",
        flush=True,
    )
    render_t0 = pf()
    camera.timings["setup_seconds"] = render_t0 - total_t0
    debug_tools.log("renderer path: batched path tracer")
    render_runs = []
    for run_index in range(args.benchmark_runs):
        run_t0 = pf()
        if quality["adaptive"]:
            camera.render_adaptive(
                int(quality["min_spp"]), int(quality["max_spp"]), float(quality["threshold"])
            )
        else:
            camera.render()
        ti.sync()
        run_seconds = pf() - run_t0
        render_runs.append(dict(
            run=run_index + 1, render_seconds=run_seconds,
            trace_seconds=camera.timings["trace_seconds"],
            filter_and_display_seconds=camera.timings["filter_and_display_seconds"],
        ))
        if args.benchmark_runs > 1:
            print(f"  render {run_index + 1}/{args.benchmark_runs}: {run_seconds:.3f}s", flush=True)
    ti.sync()
    render_t1 = pf()
    debug_tools.completed(f"path trace render ({render_t1 - render_t0:.3f}s)")
    if args.output:
        output_path = os.path.abspath(args.output)
        out_parent = os.path.dirname(output_path)
        if out_parent:
            os.makedirs(out_parent, exist_ok=True)
    else:
        output_path = os.path.abspath(os.path.join(output_dir, SCENE_DEFAULT_PNG[args.scene]))
    t0 = pf()
    print("Copying image from GPU...", flush=True)
    debug_tools.log("copy image field to NumPy")
    pixels = camera.img.to_numpy()
    copy_seconds = pf() - t0
    camera._debug_taichi_return("img.to_numpy", t0)
    print(f"Saving PNG to {output_path}...", flush=True)
    save_t0 = pf()
    with debug_tools.span(f"write PNG {output_path}"):
        ti.tools.imwrite(pixels, output_path)
    save_seconds = pf() - save_t0
    if args.linear_output or args.stats:
        if camera.last_render_adaptive:
            counts = camera.adaptive_count.to_numpy()
        else:
            counts = np.full((camera.image_width, camera.image_height), spp, dtype=np.int32)
        if args.linear_output:
            archive_path = os.path.abspath(args.linear_output)
            os.makedirs(os.path.dirname(archive_path), exist_ok=True)
            linear = camera.linear_image()
            filtered = camera.denoised_result.to_numpy() if camera.denoised_result is not None else linear
            np.savez_compressed(archive_path, linear=linear, filtered=filtered, sample_counts=counts)
        if args.stats:
            report = dict(camera.timings)
            report.update(
                scene=args.scene, width=camera.image_width, height=camera.image_height,
                sampler=args.sampler, seed=args.seed, scheduler=args.adaptive_scheduler,
                mean_spp=float(counts.mean()), max_spp=int(counts.max()),
                capped_pixels=int(np.count_nonzero(counts >= spp)),
                render_seconds=render_t1 - render_t0, copy_seconds=copy_seconds,
                save_seconds=save_seconds, total_seconds=pf() - total_t0,
                render_runs=render_runs,
            )
            report_path = os.path.abspath(args.stats)
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            with open(report_path, "w", encoding="utf-8") as report_file:
                json.dump(report, report_file, indent=2)
    debug_tools.completed(f"saved {output_path}")
    print(f"Image saved to {output_path}", flush=True)
    print(f"Render took {render_t1 - render_t0:.3f}s", flush=True)


if __name__ == "__main__":
    main()
