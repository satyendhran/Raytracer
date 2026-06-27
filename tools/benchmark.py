import argparse
import os
import time
import random
import taichi as ti
from main import build_scene_and_bvh
from Camera import Camera
from src.accel.scene_accel import PrimBVH
from src.geometry.mesh_io import load_mesh_triangles
from src.geometry.Objects import scene_primitive_sphere

def generate_grid_mesh(nx: int, ny: int, scale: float=1.0):
    verts = []
    tris = []
    for i in range(nx):
        for j in range(ny):
            x0, y0 = (i * scale, j * scale)
            x1, y1 = ((i + 1) * scale, (j + 1) * scale)
            v0 = (x0, y0, 0.0)
            v1 = (x1, y0, 0.0)
            v2 = (x1, y1, 0.0)
            v3 = (x0, y1, 0.0)
            base = len(verts)
            verts.extend([v0, v1, v2, v3])
            tris.append((base, base + 1, base + 2))
            tris.append((base, base + 2, base + 3))
    return (verts, tris)

def main():
    parser = argparse.ArgumentParser(description='Taichi ray tracer performance benchmark')
    parser.add_argument('--triangles', type=int, default=50000, help='Number of triangles to generate')
    parser.add_argument('--spheres', type=int, default=1000, help='Number of random spheres to add')
    parser.add_argument('--spp', type=int, default=8, help='Samples per pixel for the benchmark run')
    parser.add_argument('--width', type=int, default=800, help='Image width (height derived from aspect)')
    args = parser.parse_args()
    target_quads = max(1, args.triangles // 2)
    side = int(target_quads ** 0.5) + 1
    verts, tris = generate_grid_mesh(side, side, scale=0.02)
    tris = tris[:args.triangles]
    tmp_obj = os.path.join(os.path.dirname(__file__), 'tmp_mesh.obj')
    with open(tmp_obj, 'w') as f:
        for v in verts:
            f.write(f'v {v[0]} {v[1]} {v[2]}\n')
        for t in tris:
            f.write(f'f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n')
    _, mesh_triangles = load_mesh_triangles(tmp_obj, scale=1.0, translate=(0.0, 0.0, -1.0), max_triangles=args.triangles)
    sphere_prims = []
    for _ in range(args.spheres):
        cx = random.uniform(-1.0, 1.0)
        cy = random.uniform(-0.5, 0.5)
        cz = random.uniform(-2.0, -0.5)
        radius = random.uniform(0.02, 0.08)
        sphere_prims.append((ti.math.vec3(cx, cy, cz), radius, 0))
    analytic_prims = []
    for idx, (center, radius, mat_id) in enumerate(sphere_prims):
        analytic_prims.append(scene_primitive_sphere(center, center, radius, idx))
    prim_field = ti.field(dtype=ti.math.vec3, shape=len(analytic_prims), layout=ti.Layout.SOA)
    bvh = PrimBVH(prim_field, len(analytic_prims), split='sah', host_primitives=analytic_prims, host_triangles=mesh_triangles, triangle_mat_id=5, host_materials=[])
    cam_cfg = dict(lookfrom=ti.math.vec3(0, 0, 2), lookat=ti.math.vec3(0, 0, -1), vup=ti.math.vec3(0, 1, 0), vfov=45.0, aspect_ratio=16.0 / 9.0, dark_background=0)
    camera = Camera(bvh, image_width=args.width, aspect_ratio=cam_cfg['aspect_ratio'], vfov=cam_cfg['vfov'], lookfrom=cam_cfg['lookfrom'], lookat=cam_cfg['lookat'], vup=cam_cfg['vup'], defocus_angle=0.0, focus_distance=1.0, use_hdr_accum=False, dark_background=cam_cfg['dark_background'], max_bounces=8)
    camera.samples_per_pixel = args.spp
    camera.render_batch(1)
    batches = 5
    start = time.perf_counter()
    for _ in range(batches):
        camera.render_batch(args.spp)
    elapsed = time.perf_counter() - start
    total_pixels = args.width * int(args.width / cam_cfg['aspect_ratio']) * args.spp * batches
    print(f'Benchmark: {batches} batches, {elapsed:.2f}s total')
    print(f'Throughput: {total_pixels / elapsed:,.0f} rays/s')
    os.remove(tmp_obj)
if __name__ == '__main__':
    main()