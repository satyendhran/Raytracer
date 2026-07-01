import argparse
import os
import sys
import time

import numpy as np
import taichi as ti
import taichi.math as tm

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from src.accel.scene_accel import (PrimBVH, ScenePrimitive, _hit_box,
                                   _hit_sphere, _hit_triangle,
                                   scene_primitive_box, scene_primitive_sphere,
                                   scene_primitive_triangle)
from src.core.interval import Interval
from src.core.Ray import Ray
from src.geometry.Objects import hit_record
from src.scenes.scene_cornell import build_cornell_scene


def _init_taichi():
    arch = ti.cuda if ti._lib.core.with_cuda() else ti.vulkan
    ti.init(arch=arch, default_ip=ti.i32, default_fp=ti.f32)


_init_taichi()


@ti.kernel
def _bench_sphere_kernel(
    prims: ti.template(), rec: ti.template(), out_hits: ti.template(), n: ti.i32
):
    for idx in range(n):
        ox = ti.cast(idx % 256, ti.f32) * 0.003 - 0.35
        oy = ti.cast(idx // 256 % 256, ti.f32) * 0.003 - 0.35
        r = Ray(tm.vec3(ox, oy, 2.0), tm.vec3(0.0, 0.0, -1.0), 0.0, 0)
        p = prims[0]
        out_hits[idx] = ti.select(
            _hit_sphere(p, r, Interval(0.001, 1e30), rec, 0, 0) != 0, 1, 0
        )


@ti.kernel
def _bench_box_kernel(
    prims: ti.template(), rec: ti.template(), out_hits: ti.template(), n: ti.i32
):
    for idx in range(n):
        ox = ti.cast(idx % 256, ti.f32) * 0.003 - 0.35
        oy = ti.cast(idx // 256 % 256, ti.f32) * 0.003 - 0.35
        r = Ray(tm.vec3(ox, oy, 2.0), tm.vec3(0.0, 0.0, -1.0), 0.0, 0)
        p = prims[0]
        out_hits[idx] = ti.select(
            _hit_box(p, r, Interval(0.001, 1e30), rec, 0, 0) != 0, 1, 0
        )


@ti.kernel
def _bench_triangle_inline_kernel(
    prims: ti.template(), rec: ti.template(), out_hits: ti.template(), n: ti.i32
):
    for idx in range(n):
        ox = ti.cast(idx % 256, ti.f32) * 0.004
        oy = ti.cast(idx // 256 % 256, ti.f32) * 0.004
        r = Ray(tm.vec3(ox, oy, 1.0), tm.vec3(0.0, 0.0, -1.0), 0.0, 0)
        p = prims[0]
        out_hits[idx] = ti.select(
            _hit_triangle(p, r, Interval(0.001, 1e30), rec, 0, 0) != 0, 1, 0
        )


@ti.kernel
def _bench_triangle_soa_kernel(
    bvh: ti.template(), rec: ti.template(), out_hits: ti.template(), n: ti.i32
):
    for idx in range(n):
        ox = ti.cast(idx % 256, ti.f32) * 0.004
        oy = ti.cast(idx // 256 % 256, ti.f32) * 0.004
        r = Ray(tm.vec3(ox, oy, 1.0), tm.vec3(0.0, 0.0, -1.0), 0.0, 0)
        out_hits[idx] = ti.select(
            bvh._hit_triangle_soa(0, r, Interval(0.001, 1e30), rec, 0, 0) != 0, 1, 0
        )


@ti.kernel
def _bench_bvh_kernel(
    bvh: ti.template(), rec: ti.template(), out_hits: ti.template(), n: ti.i32
):
    for idx in range(n):
        ox = ti.cast(idx % 512, ti.f32) * 0.002 - 0.5
        ti.cast(idx // 512 % 512, ti.f32) * 0.002 - 0.5
        r = Ray(tm.vec3(ox, 1.0, 3.5), tm.vec3(0.0, -0.15, -1.0), 0.0, 0)
        out_hits[idx] = bvh.hit(r, Interval(0.001, 1e30), rec, 0, 0)


def _time_kernel(fn, warmup: int = 2, repeats: int = 5) -> float:
    for _ in range(warmup):
        fn()
    ti.sync()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ti.sync()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def main():
    parser = argparse.ArgumentParser(description="Benchmark intersection throughput.")
    parser.add_argument(
        "--rays", type=int, default=65536, help="Rays per benchmark kernel"
    )
    args = parser.parse_args()
    n = args.rays
    rec = hit_record.field(shape=(1, 1))
    out = ti.field(dtype=ti.i32, shape=(n,))
    prims = ScenePrimitive.field(shape=(1,))
    prims[0] = scene_primitive_sphere(tm.vec3(0, 0, -1), tm.vec3(0, 0, -1), 0.5, 0)
    t_sphere = _time_kernel(lambda: _bench_sphere_kernel(prims, rec, out, n))
    prims[0] = scene_primitive_box(
        tm.vec3(-0.5, -0.5, -1.5), tm.vec3(0.5, 0.5, -0.5), 0
    )
    t_box = _time_kernel(lambda: _bench_box_kernel(prims, rec, out, n))
    prims[0] = scene_primitive_triangle(
        tm.vec3(0, 0, 0), tm.vec3(1, 0, 0), tm.vec3(0, 1, 0), 0
    )
    t_tri = _time_kernel(lambda: _bench_triangle_inline_kernel(prims, rec, out, n))
    from src.geometry.mesh_io import MeshTriangle

    tri = MeshTriangle(
        tm.vec3(0.0, 0.0, 0.0),
        tm.vec3(1.0, 0.0, 0.0),
        tm.vec3(0.0, 1.0, 0.0),
        tm.vec3(0.0, 0.0, 1.0),
        tm.vec3(0.0, 0.0, 1.0),
        tm.vec3(0.0, 0.0, 1.0),
        tm.vec3(0.0, 0.0, 0.0),
        tm.vec3(1.0, 0.0, 0.0),
        tm.vec3(0.0, 1.0, 0.0),
        geo_normal=tm.vec3(0.0, 0.0, 1.0),
    )
    tri_bvh = PrimBVH(
        ScenePrimitive.field(shape=(1,)), 0, split="sah", host_triangles=[tri]
    )
    t_tri_soa = _time_kernel(lambda: _bench_triangle_soa_kernel(tri_bvh, rec, out, n))
    mats, bvh, _, _cam = build_cornell_scene("sah")
    t_bvh = _time_kernel(lambda: _bench_bvh_kernel(bvh, rec, out, n))
    mrays = n / 1000000.0
    print("Benchmark report (median kernel time, post-warmup)")
    print(f"  rays/kernel: {n:,}")
    print(
        f"  sphere inline:     {mrays / t_sphere:.2f} Mrays/s  ({t_sphere * 1000:.2f} ms)"
    )
    print(
        f"  box inline:          {mrays / t_box:.2f} Mrays/s  ({t_box * 1000:.2f} ms)"
    )
    print(f"  triangle inline:   {mrays / t_tri:.2f} Mrays/s  ({t_tri * 1000:.2f} ms)")
    print(
        f"  triangle SOA:      {mrays / t_tri_soa:.2f} Mrays/s  ({t_tri_soa * 1000:.2f} ms)"
    )
    print(f"  cornell BVH:       {mrays / t_bvh:.2f} Mrays/s  ({t_bvh * 1000:.2f} ms)")
    print(f"  sphere/box ratio:  {t_sphere / max(t_box, 1e-09):.3f}x")
    print(f"  triangle/sphere:   {t_tri / max(t_sphere, 1e-09):.3f}x")
    print(f"  SOA vs inline tri: {t_tri / max(t_tri_soa, 1e-09):.3f}x speedup")


if __name__ == "__main__":
    main()
