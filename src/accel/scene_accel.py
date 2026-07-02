import math

import numpy as np
import taichi as ti
import taichi.math as tm

from src.accel.BVH import (
    BVHBuilder,
    BVHNode,
    aabb_hit,
    aabb_hit_inv,
    aabb_ray_t_enter,
    aabb_ray_t_enter_inv,
)
from src.accel.BVHGPUSAH import GPUSAHBVHBuilder
from src.core.interval import Interval
from src.core.Ray import Ray
from src.geometry.Objects import get_sphere_uv
from tools import debug_tools

vec3 = tm.vec3
KIND_SPHERE = 0
KIND_BOX = 1
KIND_TRIANGLE = 2
KIND_QUAD = 3
KIND_ELLIPSOID = 4
KIND_SDF = 5
SDF_TORUS = 0
SDF_CAPSULE = 1
SDF_ROUNDED_BOX = 2
BVH_FLAT_EPS = 0.0001
TRI_DET_EPS = 1e-08
BARY_EPS = 1e-06
RAY_T_MAX_DEFAULT = 1e30
PRIM_FLAG_INVISIBLE_PRIMARY = 1 << 16
PRIM_FLAG_ONE_WAY_INSIDE = 1 << 17
PRIM_FLAG_TWO_SIDED = 1 << 18
SDF_KIND_MASK = 255
ScenePrimitive = ti.types.struct(
    kind=ti.i32,
    mat_id=ti.i32,
    center0=vec3,
    center_vel=vec3,
    radius=ti.f32,
    box_min=vec3,
    box_max=vec3,
    bvh_min=vec3,
    bvh_max=vec3,
    sdf_kind=ti.i32,
    n0=vec3,
    n1=vec3,
    n2=vec3,
    uv0=vec3,
    uv1=vec3,
    uv2=vec3,
)


@ti.func
def _sphere_center_at(p: ScenePrimitive, t: ti.f32) -> vec3:
    return p.center0 + p.center_vel * t


@ti.func
def _mt_triangle_intersect(
    v0: vec3,
    e1: vec3,
    e2: vec3,
    geom_n: vec3,
    plane_d: ti.f32,
    origin: vec3,
    direction: vec3,
    t_min: ti.f32,
    t_max: ti.f32,
    two_sided: ti.i32,
):
    nd = tm.dot(geom_n, direction)
    plane_ok = ti.cast(1, ti.i32)
    if ti.abs(nd) >= TRI_DET_EPS:
        t_plane = (plane_d - tm.dot(geom_n, origin)) / nd
        plane_ok = ti.cast((t_plane > t_min) & (t_plane < t_max), ti.i32)
    pvec = tm.cross(direction, e2)
    det = tm.dot(e1, pvec)
    abs_det = ti.abs(det)
    face_ok = ti.cast(0, ti.i32)
    if two_sided != 0:
        face_ok = ti.cast(abs_det >= TRI_DET_EPS, ti.i32)
    else:
        face_ok = ti.cast(det >= TRI_DET_EPS, ti.i32)
    inv_det = 1.0 / ti.select(abs_det >= TRI_DET_EPS, det, 1.0)
    tvec = origin - v0
    uu = tm.dot(tvec, pvec) * inv_det
    qvec = tm.cross(tvec, e1)
    vv = tm.dot(direction, qvec) * inv_det
    root = tm.dot(e2, qvec) * inv_det
    bary_ok = (
        (uu >= -BARY_EPS)
        & (uu <= 1.0 + BARY_EPS)
        & (vv >= -BARY_EPS)
        & (uu + vv <= 1.0 + BARY_EPS)
    )
    t_ok = (root > t_min) & (root < t_max)
    valid = (plane_ok != 0) & (face_ok != 0) & bary_ok & t_ok
    hit_ok = ti.cast(valid, ti.i32)
    out_t = ti.select(valid, root, 0.0)
    out_u = ti.select(valid, uu, 0.0)
    out_v = ti.select(valid, vv, 0.0)
    return (hit_ok, out_t, out_u, out_v)


@ti.func
def _hit_sphere(
    p: ScenePrimitive, r: Ray, ray_t: Interval, rec, xi: ti.i32, yi: ti.i32
) -> ti.u8:
    current_centre = _sphere_center_at(p, r.tm)
    oc = current_centre - r.origin
    d = r.direction
    a = tm.dot(d, d)
    h = tm.dot(d, oc)
    radius_sq = p.box_min.x
    c = tm.dot(oc, oc) - radius_sq
    disc = h * h - a * c
    ret = ti.cast(0, ti.u8)
    if disc >= 0.0:
        sqrt_disc = ti.math.sqrt(ti.max(disc, 0.0))
        inv_a = 1.0 / a
        root0 = (h - sqrt_disc) * inv_a
        root1 = (h + sqrt_disc) * inv_a
        use0 = ray_t.surrounds(root0)
        root_pick = ti.select(use0, root0, root1)
        valid = use0 | ray_t.surrounds(root1)
        valid = valid & (root_pick > ray_t.min) & (root_pick < ray_t.max)
        if valid:
            pt_h = r.at(root_pick)
            rec[xi, yi].t = root_pick
            rec[xi, yi].p = pt_h
            inv_r = 1.0 / p.radius
            outward_normal = (pt_h - current_centre) * inv_r
            rec[xi, yi].u, rec[xi, yi].v = get_sphere_uv(outward_normal)
            rec[xi, yi].bary_u = 0.0
            rec[xi, yi].bary_v = 0.0
            rec[xi, yi].geo_normal = outward_normal
            rec[xi, yi].set_face_normal(r, outward_normal)
            rec[xi, yi].mat_id = p.mat_id
            rec[xi, yi].prim_id = -1
            ret = ti.cast(1, ti.u8)
    return ret


@ti.func
def _hit_box(
    p: ScenePrimitive, r: Ray, ray_t: Interval, rec, xi: ti.i32, yi: ti.i32
) -> ti.u8:
    t0 = ray_t.min
    t1 = ray_t.max
    bmin = p.box_min
    bmax = p.box_max
    miss = ti.cast(0, ti.i32)
    for axis in ti.static(range(3)):
        inv_d = 1.0 / r.direction[axis]
        t_near = (bmin[axis] - r.origin[axis]) * inv_d
        t_far = (bmax[axis] - r.origin[axis]) * inv_d
        tn = ti.select(inv_d < 0.0, t_far, t_near)
        tf = ti.select(inv_d < 0.0, t_near, t_far)
        t0 = ti.max(tn, t0)
        t1 = ti.min(tf, t1)
        if t0 >= t1:
            miss = 1
    root = t0
    root = ti.select(t0 < ray_t.min, t1, root)
    valid = (miss == 0) & (root > ray_t.min) & (root < ray_t.max)
    if valid:
        rec[xi, yi].t = root
        pt = r.at(root)
        rec[xi, yi].p = pt
        eps = 0.0001
        outward = vec3(0.0, 0.0, 0.0)
        face_u = 0.0
        face_v = 0.0
        sz = bmax - bmin + vec3(1e-08, 1e-08, 1e-08)
        if ti.abs(pt.x - bmin.x) < eps:
            outward = vec3(-1.0, 0.0, 0.0)
            face_u = (pt.z - bmin.z) / sz.z
            face_v = (pt.y - bmin.y) / sz.y
        elif ti.abs(pt.x - bmax.x) < eps:
            outward = vec3(1.0, 0.0, 0.0)
            face_u = (pt.z - bmin.z) / sz.z
            face_v = (pt.y - bmin.y) / sz.y
        elif ti.abs(pt.y - bmin.y) < eps:
            outward = vec3(0.0, -1.0, 0.0)
            face_u = (pt.x - bmin.x) / sz.x
            face_v = (pt.z - bmin.z) / sz.z
        elif ti.abs(pt.y - bmax.y) < eps:
            outward = vec3(0.0, 1.0, 0.0)
            face_u = (pt.x - bmin.x) / sz.x
            face_v = (pt.z - bmin.z) / sz.z
        elif ti.abs(pt.z - bmin.z) < eps:
            outward = vec3(0.0, 0.0, -1.0)
            face_u = (pt.x - bmin.x) / sz.x
            face_v = (pt.y - bmin.y) / sz.y
        else:
            outward = vec3(0.0, 0.0, 1.0)
            face_u = (pt.x - bmin.x) / sz.x
            face_v = (pt.y - bmin.y) / sz.y
        rec[xi, yi].u = face_u
        rec[xi, yi].v = face_v
        rec[xi, yi].geo_normal = outward
        rec[xi, yi].set_face_normal(r, outward)
        rec[xi, yi].mat_id = p.mat_id
        rec[xi, yi].prim_id = -1
    return ti.select(valid, ti.cast(1, ti.u8), ti.cast(0, ti.u8))


@ti.func
def _hit_triangle(
    p: ScenePrimitive, r: Ray, ray_t: Interval, rec, xi: ti.i32, yi: ti.i32
) -> ti.u8:
    v0 = p.center0
    e1 = p.center_vel
    e2 = p.box_min
    geom_n = p.box_max
    plane_d = tm.dot(geom_n, v0)
    two_sided = ti.cast(p.sdf_kind & PRIM_FLAG_TWO_SIDED != 0, ti.i32)
    hit_ok, root, uu, vv = _mt_triangle_intersect(
        v0,
        e1,
        e2,
        geom_n,
        plane_d,
        r.origin,
        r.direction,
        ray_t.min,
        ray_t.max,
        two_sided,
    )
    valid = hit_ok != 0
    if valid:
        rec[xi, yi].t = root
        rec[xi, yi].p = r.at(root)
        rec[xi, yi].bary_u = uu
        rec[xi, yi].bary_v = vv
        shade_n = geom_n
        if tm.dot(p.n0, p.n0) + tm.dot(p.n1, p.n1) + tm.dot(p.n2, p.n2) > 1e-12:
            w0 = 1.0 - uu - vv
            raw_shade_n = w0 * p.n0 + uu * p.n1 + vv * p.n2
            shade_len2 = tm.dot(raw_shade_n, raw_shade_n)
            if shade_len2 > 1e-12:
                shade_n = raw_shade_n / ti.sqrt(shade_len2)
                if tm.dot(shade_n, geom_n) < 0.0:
                    shade_n = -shade_n
        rec[xi, yi].set_face_normal(r, geom_n)
        rec[xi, yi].geo_normal = geom_n
        if tm.dot(r.direction, geom_n) > 0.0:
            shade_n = -shade_n
        rec[xi, yi].normal = shade_n
        rec[xi, yi].mat_id = p.mat_id
        rec[xi, yi].prim_id = -1
        tex_u = uu
        tex_v = vv
        if tm.dot(p.uv0, p.uv0) + tm.dot(p.uv1, p.uv1) + tm.dot(p.uv2, p.uv2) > 0.0:
            w0 = 1.0 - uu - vv
            uv = w0 * p.uv0 + uu * p.uv1 + vv * p.uv2
            tex_u = uv.x
            tex_v = uv.y
        rec[xi, yi].u = ti.math.clamp(tex_u, 0.0, 1.0)
        rec[xi, yi].v = ti.math.clamp(tex_v, 0.0, 1.0)
    return ti.select(valid, ti.cast(1, ti.u8), ti.cast(0, ti.u8))


@ti.func
def _hit_quad(
    p: ScenePrimitive, r: Ray, ray_t: Interval, rec, xi: ti.i32, yi: ti.i32
) -> ti.u8:
    Q = p.center0
    u = p.center_vel
    v = p.box_min
    n = p.box_max
    denom = tm.dot(n, r.direction)
    denom_ok = ti.abs(denom) >= 1e-08
    t = tm.dot(Q - r.origin, n) / ti.select(denom_ok, denom, 1.0)
    pt = r.at(t)
    w = pt - Q
    uu = tm.dot(u, u)
    uv = tm.dot(u, v)
    vv = tm.dot(v, v)
    wu = tm.dot(w, u)
    wv = tm.dot(w, v)
    det = uu * vv - uv * uv
    det_ok = ti.abs(det) >= 1e-20
    inv_det = 1.0 / ti.select(det_ok, det, 1.0)
    alpha = (wu * vv - wv * uv) * inv_det
    beta = (uu * wv - uv * wu) * inv_det
    valid = denom_ok & ray_t.surrounds(t)
    valid = valid & det_ok
    valid = valid & (alpha >= 0.0) & (alpha <= 1.0) & (beta >= 0.0) & (beta <= 1.0)
    if valid:
        rec[xi, yi].t = t
        rec[xi, yi].p = pt
        rec[xi, yi].geo_normal = n
        rec[xi, yi].set_face_normal(r, n)
        rec[xi, yi].mat_id = p.mat_id
        rec[xi, yi].prim_id = -1
        rec[xi, yi].u = alpha
        rec[xi, yi].v = beta
    return ti.select(valid, ti.cast(1, ti.u8), ti.cast(0, ti.u8))


@ti.func
def _hit_ellipsoid(
    p: ScenePrimitive, r: Ray, ray_t: Interval, rec, xi: ti.i32, yi: ti.i32
) -> ti.u8:
    c = p.center0
    rad = p.box_min
    ro = (r.origin - c) / rad
    rd = r.direction / rad
    a = tm.dot(rd, rd)
    h = tm.dot(ro, rd)
    cc = tm.dot(ro, ro) - 1.0
    disc = h * h - a * cc
    ret = ti.cast(0, ti.u8)
    if disc >= 0.0:
        sd = ti.math.sqrt(ti.max(disc, 0.0))
        root0 = (-h - sd) / a
        root1 = (-h + sd) / a
        use0 = ray_t.surrounds(root0)
        root_pick = ti.select(use0, root0, root1)
        valid = use0 | ray_t.surrounds(root1)
        valid = valid & (root_pick > ray_t.min) & (root_pick < ray_t.max)
        if valid:
            rec[xi, yi].t = root_pick
            pw = r.at(root_pick)
            rec[xi, yi].p = pw
            gn = (pw - c) / (rad * rad)
            gn_norm = tm.normalize(gn)
            rec[xi, yi].geo_normal = gn_norm
            rec[xi, yi].set_face_normal(r, gn_norm)
            rec[xi, yi].mat_id = p.mat_id
            rec[xi, yi].prim_id = -1
            rec[xi, yi].u, rec[xi, yi].v = get_sphere_uv(gn_norm)
            ret = ti.cast(1, ti.u8)
    return ret


@ti.func
def _sdf_eval(p: ScenePrimitive, pw: vec3) -> ti.f32:
    lp = pw - p.center0
    d = 0.0
    sdf_kind = p.sdf_kind & SDF_KIND_MASK
    if sdf_kind == SDF_TORUS:
        major_r = p.box_min.x
        minor_r = p.box_min.y
        q = vec3(tm.length(vec3(lp.x, 0.0, lp.z)) - major_r, lp.y, 0.0)
        d = tm.length(vec3(q.x, q.y, 0.0)) - minor_r
    elif sdf_kind == SDF_CAPSULE:
        cap_r = p.box_min.x
        half_h = p.box_min.y
        clamped_y = ti.math.clamp(lp.y, -half_h, half_h)
        d = tm.length(lp - vec3(0.0, clamped_y, 0.0)) - cap_r
    else:
        hx = p.box_min.x
        hy = p.box_min.y
        hz = p.box_min.z
        cr = p.radius
        qx = ti.abs(lp.x) - hx
        qy = ti.abs(lp.y) - hy
        qz = ti.abs(lp.z) - hz
        outer = tm.length(vec3(ti.max(qx, 0.0), ti.max(qy, 0.0), ti.max(qz, 0.0)))
        inner = ti.min(ti.max(qx, ti.max(qy, qz)), 0.0)
        d = outer + inner - cr
    return d


@ti.func
def _hit_sdf(
    p: ScenePrimitive, r: Ray, ray_t: Interval, rec, xi: ti.i32, yi: ti.i32
) -> ti.u8:
    valid = ti.cast(0, ti.u8)
    t = ray_t.min
    prev_d = 10000000000.0
    for _step in range(64):
        pw = r.at(t)
        d = _sdf_eval(p, pw)
        if d < 5e-05:
            valid = ti.cast(1, ti.u8)
            break
        step = d * ti.select(d < prev_d, 1.5, 1.0)
        t += step
        prev_d = d
        if t >= ray_t.max:
            break
    if valid != 0:
        pw = r.at(t)
        eps = 5e-05
        nx = _sdf_eval(p, pw + vec3(eps, 0.0, 0.0)) - _sdf_eval(
            p, pw - vec3(eps, 0.0, 0.0)
        )
        ny = _sdf_eval(p, pw + vec3(0.0, eps, 0.0)) - _sdf_eval(
            p, pw - vec3(0.0, eps, 0.0)
        )
        nz = _sdf_eval(p, pw + vec3(0.0, 0.0, eps)) - _sdf_eval(
            p, pw - vec3(0.0, 0.0, eps)
        )
        n = tm.normalize(vec3(nx, ny, nz))
        rec[xi, yi].t = t
        rec[xi, yi].p = pw
        rec[xi, yi].set_face_normal(r, n)
        rec[xi, yi].mat_id = p.mat_id
        rec[xi, yi].prim_id = -1
        rec[xi, yi].u, rec[xi, yi].v = get_sphere_uv(n)
    return valid


@ti.func
def _occludes_sphere(p: ScenePrimitive, r: Ray, ray_t: Interval) -> ti.i32:
    c = _sphere_center_at(p, r.tm)
    oc = c - r.origin
    d = r.direction
    a = tm.dot(d, d)
    h = tm.dot(d, oc)
    cc = tm.dot(oc, oc) - p.box_min.x
    disc = h * h - a * cc
    ret = 0
    if disc >= 0.0:
        sd = ti.sqrt(ti.max(disc, 0.0))
        inv_a = 1.0 / a
        t0 = (h - sd) * inv_a
        t1 = (h + sd) * inv_a
        ret = ti.select(ray_t.surrounds(t0) | ray_t.surrounds(t1), 1, 0)
    return ret


@ti.func
def _occludes_triangle(p: ScenePrimitive, r: Ray, ray_t: Interval) -> ti.i32:
    v0 = p.center0
    e1 = p.center_vel
    e2 = p.box_min
    geom_n = p.box_max
    plane_d = tm.dot(geom_n, v0)
    two_sided = ti.cast(p.sdf_kind & PRIM_FLAG_TWO_SIDED != 0, ti.i32)
    hit_ok, _, _, _ = _mt_triangle_intersect(
        v0,
        e1,
        e2,
        geom_n,
        plane_d,
        r.origin,
        r.direction,
        ray_t.min,
        ray_t.max,
        two_sided,
    )
    return hit_ok


@ti.func
def _occludes_quad(p: ScenePrimitive, r: Ray, ray_t: Interval) -> ti.i32:
    Q = p.center0
    u = p.center_vel
    v = p.box_min
    n = p.box_max
    denom = tm.dot(n, r.direction)
    ret = 0
    if ti.abs(denom) >= 1e-08:
        t = tm.dot(Q - r.origin, n) / denom
        pt = r.at(t)
        w = pt - Q
        uu = tm.dot(u, u)
        uv = tm.dot(u, v)
        vv = tm.dot(v, v)
        wu = tm.dot(w, u)
        wv = tm.dot(w, v)
        det = uu * vv - uv * uv
        if ti.abs(det) >= 1e-20:
            inv_det = 1.0 / det
            alpha = (wu * vv - wv * uv) * inv_det
            beta = (uu * wv - uv * wu) * inv_det
            ret = ti.select(
                ray_t.surrounds(t)
                & (alpha >= 0.0)
                & (alpha <= 1.0)
                & (beta >= 0.0)
                & (beta <= 1.0),
                1,
                0,
            )
    return ret


@ti.func
def _occludes_ellipsoid(p: ScenePrimitive, r: Ray, ray_t: Interval) -> ti.i32:
    c = p.center0
    rad = p.box_min
    ro = (r.origin - c) / rad
    rd = r.direction / rad
    a = tm.dot(rd, rd)
    h = tm.dot(ro, rd)
    cc = tm.dot(ro, ro) - 1.0
    disc = h * h - a * cc
    ret = 0
    if disc >= 0.0:
        sd = ti.sqrt(ti.max(disc, 0.0))
        t0 = (-h - sd) / a
        t1 = (-h + sd) / a
        ret = ti.select(ray_t.surrounds(t0) | ray_t.surrounds(t1), 1, 0)
    return ret


@ti.func
def _occludes_sdf(p: ScenePrimitive, r: Ray, ray_t: Interval) -> ti.i32:
    ret = 0
    t = ray_t.min
    prev_d = 10000000000.0
    for _step in range(64):
        pw = r.at(t)
        d = _sdf_eval(p, pw)
        if d < 5e-05:
            ret = 1
            break
        t += d * ti.select(d < prev_d, 1.5, 1.0)
        prev_d = d
        if t >= ray_t.max:
            break
    return ret


@ti.func
def _skip_one_way_inside(p: ScenePrimitive, r: Ray) -> ti.i32:
    skip = 0
    if p.sdf_kind & PRIM_FLAG_ONE_WAY_INSIDE != 0:
        side = tm.dot(r.origin - p.center0, p.box_max)
        moving_toward_inside = tm.dot(r.direction, p.box_max) > 0.0
        skip = ti.select((side < 0.0) & moving_toward_inside, 1, 0)
    return skip


@ti.func
def primitive_any_hit(p: ScenePrimitive, r: Ray, ray_t: Interval) -> ti.i32:
    ret = 0
    if _skip_one_way_inside(p, r) == 0:
        if p.kind == KIND_SPHERE:
            ret = _occludes_sphere(p, r, ray_t)
        elif p.kind == KIND_BOX:
            ret = aabb_hit(p.box_min, p.box_max, r, ray_t.min, ray_t.max)
        elif p.kind == KIND_TRIANGLE:
            ret = _occludes_triangle(p, r, ray_t)
        elif p.kind == KIND_QUAD:
            ret = _occludes_quad(p, r, ray_t)
        elif p.kind == KIND_SDF:
            ret = _occludes_sdf(p, r, ray_t)
        else:
            ret = _occludes_ellipsoid(p, r, ray_t)
    return ret


@ti.func
def primitive_hit(
    p: ScenePrimitive, r: Ray, ray_t: Interval, rec, xi: ti.i32, yi: ti.i32
) -> ti.u8:
    ret = ti.cast(0, ti.u8)
    skip_primary = (p.sdf_kind & PRIM_FLAG_INVISIBLE_PRIMARY != 0) & (r.kind == 0)
    skip_one_way = _skip_one_way_inside(p, r) != 0
    if not skip_primary | skip_one_way:
        if p.kind == KIND_SPHERE:
            ret = _hit_sphere(p, r, ray_t, rec, xi, yi)
        elif p.kind == KIND_BOX:
            ret = _hit_box(p, r, ray_t, rec, xi, yi)
        elif p.kind == KIND_TRIANGLE:
            ret = _hit_triangle(p, r, ray_t, rec, xi, yi)
        elif p.kind == KIND_QUAD:
            ret = _hit_quad(p, r, ray_t, rec, xi, yi)
        elif p.kind == KIND_SDF:
            ret = _hit_sdf(p, r, ray_t, rec, xi, yi)
        else:
            ret = _hit_ellipsoid(p, r, ray_t, rec, xi, yi)
    return ret


def _sphere_world_aabb(c1, c2, r):
    c2 - c1
    rvec = vec3(r, r, r)
    min1, max1 = (c1 - rvec, c1 + rvec)
    aabb_min = vec3(min(min1[0], max1[0]), min(min1[1], max1[1]), min(min1[2], max1[2]))
    aabb_max = vec3(max(min1[0], max1[0]), max(min1[1], max1[1]), max(min1[2], max1[2]))
    if c1[0] != c2[0] or c1[1] != c2[1] or c1[2] != c2[2]:
        min2, max2 = (c2 - rvec, c2 + rvec)
        aabb_min = vec3(
            min(aabb_min[0], min2[0], max2[0]),
            min(aabb_min[1], min2[1], max2[1]),
            min(aabb_min[2], min2[2], max2[2]),
        )
        aabb_max = vec3(
            max(aabb_max[0], min2[0], max2[0]),
            max(aabb_max[1], min2[1], max2[1]),
            max(aabb_max[2], min2[2], max2[2]),
        )
    return (aabb_min, aabb_max)


def _pad_degenerate_aabb(aabb_min: vec3, aabb_max: vec3, eps: float = BVH_FLAT_EPS):
    mn = [float(aabb_min[0]), float(aabb_min[1]), float(aabb_min[2])]
    mx = [float(aabb_max[0]), float(aabb_max[1]), float(aabb_max[2])]
    for axis in range(3):
        if mx[axis] - mn[axis] < eps:
            center = 0.5 * (mn[axis] + mx[axis])
            mn[axis] = center - 0.5 * eps
            mx[axis] = center + 0.5 * eps
    return (vec3(mn[0], mn[1], mn[2]), vec3(mx[0], mx[1], mx[2]))


def scene_primitive_sphere(c1, c2, r, mat_id: int) -> ScenePrimitive:
    aabb_min, aabb_max = _sphere_world_aabb(c1, c2, r)
    v = c2 - c1
    r_sq = float(r) * float(r)
    return ScenePrimitive(
        kind=KIND_SPHERE,
        mat_id=mat_id,
        center0=c1,
        center_vel=v,
        radius=r,
        box_min=vec3(r_sq, r_sq, r_sq),
        box_max=vec3(0.0),
        bvh_min=aabb_min,
        bvh_max=aabb_max,
    )


def scene_primitive_box(bmin, bmax, mat_id: int) -> ScenePrimitive:
    return ScenePrimitive(
        kind=KIND_BOX,
        mat_id=mat_id,
        center0=vec3(0.0),
        center_vel=vec3(0.0),
        radius=0.0,
        box_min=bmin,
        box_max=bmax,
        bvh_min=bmin,
        bvh_max=bmax,
    )


def _componentwise_min_max(p0: vec3, p1: vec3, p2: vec3):
    mn = vec3(
        min(p0[0], p1[0], p2[0]), min(p0[1], p1[1], p2[1]), min(p0[2], p1[2], p2[2])
    )
    mx = vec3(
        max(p0[0], p1[0], p2[0]), max(p0[1], p1[1], p2[1]), max(p0[2], p1[2], p2[2])
    )
    return (mn, mx)


def scene_primitive_triangle(
    v0: vec3,
    v1: vec3,
    v2: vec3,
    mat_id: int,
    n0=None,
    n1=None,
    n2=None,
    uv0=None,
    uv1=None,
    uv2=None,
) -> ScenePrimitive:
    e1 = v1 - v0
    e2 = v2 - v0
    p0 = v0
    p1 = v0 + e1
    p2 = v0 + e2
    cx = e1[1] * e2[2] - e1[2] * e2[1]
    cy = e1[2] * e2[0] - e1[0] * e2[2]
    cz = e1[0] * e2[1] - e1[1] * e2[0]
    gn_len = math.sqrt(cx * cx + cy * cy + cz * cz)
    if gn_len > 1e-20:
        geo_n = vec3(cx / gn_len, cy / gn_len, cz / gn_len)
    else:
        geo_n = vec3(0.0, 0.0, 1.0)
    aabb_min, aabb_max = _pad_degenerate_aabb(*_componentwise_min_max(p0, p1, p2))
    return ScenePrimitive(
        kind=KIND_TRIANGLE,
        mat_id=mat_id,
        center0=v0,
        center_vel=e1,
        radius=0.0,
        box_min=e2,
        box_max=geo_n,
        bvh_min=aabb_min,
        bvh_max=aabb_max,
        sdf_kind=0,
        n0=n0 if n0 is not None else vec3(0.0),
        n1=n1 if n1 is not None else vec3(0.0),
        n2=n2 if n2 is not None else vec3(0.0),
        uv0=uv0 if uv0 is not None else vec3(0.0),
        uv1=uv1 if uv1 is not None else vec3(0.0),
        uv2=uv2 if uv2 is not None else vec3(0.0),
    )


def scene_primitive_quad(
    Q: vec3,
    u: vec3,
    v: vec3,
    mat_id: int,
    invisible_primary: bool = False,
    one_way_inside: bool = False,
) -> ScenePrimitive:
    p00 = Q
    p10 = Q + u
    p01 = Q + v
    p11 = Q + u + v
    xs = (p00[0], p10[0], p01[0], p11[0])
    ys = (p00[1], p10[1], p01[1], p11[1])
    zs = (p00[2], p10[2], p01[2], p11[2])
    aabb_min, aabb_max = _pad_degenerate_aabb(
        vec3(min(xs), min(ys), min(zs)), vec3(max(xs), max(ys), max(zs))
    )
    cx = u[1] * v[2] - u[2] * v[1]
    cy = u[2] * v[0] - u[0] * v[2]
    cz = u[0] * v[1] - u[1] * v[0]
    ln = math.sqrt(cx * cx + cy * cy + cz * cz) + 1e-20
    n = vec3(cx / ln, cy / ln, cz / ln)
    return ScenePrimitive(
        kind=KIND_QUAD,
        mat_id=mat_id,
        center0=Q,
        center_vel=u,
        radius=0.0,
        box_min=v,
        box_max=n,
        bvh_min=aabb_min,
        bvh_max=aabb_max,
        sdf_kind=(PRIM_FLAG_INVISIBLE_PRIMARY if invisible_primary else 0)
        | (PRIM_FLAG_ONE_WAY_INSIDE if one_way_inside else 0),
    )


def scene_primitive_ellipsoid(center: vec3, radii: vec3, mat_id: int) -> ScenePrimitive:
    aabb_min = center - radii
    aabb_max = center + radii
    return ScenePrimitive(
        kind=KIND_ELLIPSOID,
        mat_id=mat_id,
        center0=center,
        center_vel=vec3(0.0),
        radius=0.0,
        box_min=radii,
        box_max=vec3(0.0),
        bvh_min=aabb_min,
        bvh_max=aabb_max,
        sdf_kind=0,
    )


def scene_primitive_sdf_torus(
    center: vec3, major_r: float, minor_r: float, mat_id: int
) -> ScenePrimitive:
    ext = major_r + minor_r
    aabb_min = center - vec3(ext, minor_r, ext)
    aabb_max = center + vec3(ext, minor_r, ext)
    return ScenePrimitive(
        kind=KIND_SDF,
        mat_id=mat_id,
        center0=center,
        center_vel=vec3(0.0),
        radius=0.0,
        box_min=vec3(major_r, minor_r, 0.0),
        box_max=vec3(0.0),
        bvh_min=aabb_min,
        bvh_max=aabb_max,
        sdf_kind=SDF_TORUS,
    )


def scene_primitive_sdf_capsule(
    center: vec3, cap_radius: float, half_height: float, mat_id: int
) -> ScenePrimitive:
    aabb_min = center - vec3(cap_radius, half_height + cap_radius, cap_radius)
    aabb_max = center + vec3(cap_radius, half_height + cap_radius, cap_radius)
    return ScenePrimitive(
        kind=KIND_SDF,
        mat_id=mat_id,
        center0=center,
        center_vel=vec3(0.0),
        radius=0.0,
        box_min=vec3(cap_radius, half_height, 0.0),
        box_max=vec3(0.0),
        bvh_min=aabb_min,
        bvh_max=aabb_max,
        sdf_kind=SDF_CAPSULE,
    )


def scene_primitive_sdf_rounded_box(
    center: vec3, half_extents: vec3, corner_r: float, mat_id: int
) -> ScenePrimitive:
    ext = half_extents + vec3(corner_r, corner_r, corner_r)
    aabb_min = center - ext
    aabb_max = center + ext
    return ScenePrimitive(
        kind=KIND_SDF,
        mat_id=mat_id,
        center0=center,
        center_vel=vec3(0.0),
        radius=corner_r,
        box_min=half_extents,
        box_max=vec3(0.0),
        bvh_min=aabb_min,
        bvh_max=aabb_max,
        sdf_kind=SDF_ROUNDED_BOX,
    )


def load_mesh_primitives(
    path: str,
    mat_id: int,
    scale: float = 1.0,
    translate=(0.0, 0.0, 0.0),
    material_slots: int = 1,
):
    from src.geometry.mesh_io import load_mesh_auto

    mesh = load_mesh_auto(path, scale=scale, translate=translate)
    slots = max(1, int(material_slots))
    return [
        scene_primitive_triangle(
            tri.p0,
            tri.p1,
            tri.p2,
            mat_id + min(max(int(tri.material_id), 0), slots - 1),
            tri.n0,
            tri.n1,
            tri.n2,
            tri.uv0,
            tri.uv1,
            tri.uv2,
        )
        for tri in mesh.triangles
    ]


def load_mesh_triangles(
    path: str,
    scale: float = 1.0,
    translate=(0.0, 0.0, 0.0),
    material_slots: int = 1,
    max_triangles: int | None = None,
    sort_by_material: bool = False,
    flip_winding: bool = False,
    expected_face_count: int | None = None,
):
    from dataclasses import replace

    from src.geometry.mesh_io import load_mesh_auto

    mesh = load_mesh_auto(
        path, scale=scale, translate=translate, flip_winding=flip_winding
    )
    tris = list(mesh.triangles)
    if len(tris) != mesh.stats.valid_face_count:
        raise RuntimeError(
            f"Mesh {path}: tri_count={len(tris)} != valid_face_count={mesh.stats.valid_face_count}"
        )
    if expected_face_count is not None and len(tris) != expected_face_count:
        raise RuntimeError(
            f"Mesh {path}: loaded {len(tris)} triangles, expected {expected_face_count}"
        )
    if max_triangles is not None and max_triangles > 0 and (len(tris) > max_triangles):
        step = max(1, len(tris) // int(max_triangles))
        tris = tris[::step][: int(max_triangles)]
        print(
            f"[mesh load] WARNING: subsampled {path} to {len(tris):,} triangles (max_triangles={max_triangles})",
            file=__import__("sys").stderr,
        )
    slots = max(1, int(material_slots))
    if slots == 1:
        tris = [replace(tri, material_id=0) for tri in tris]
    else:
        tris = [
            replace(tri, material_id=min(max(int(tri.material_id), 0), slots - 1))
            for tri in tris
        ]
    if sort_by_material:
        tris.sort(key=lambda tri: tri.material_id)
    return (mesh, tris)


@ti.data_oriented
class PrimBVH:
    MAX_HIT_STEPS = 65536
    MAX_SHADOW_STEPS = 32768
    HIT_STACK_SIZE = 64
    SHADOW_STACK_SIZE = 64
    GPU_LEAF_SIZE = 4

    @staticmethod
    def triangle_ref(tri_idx: ti.i32) -> ti.i32:
        return -tri_idx - 1

    @staticmethod
    @ti.func
    def _is_triangle_ref(ref: ti.i32) -> ti.i32:
        return ti.select(ref < 0, 1, 0)

    @staticmethod
    @ti.func
    def _triangle_index(ref: ti.i32) -> ti.i32:
        return -ref - 1

    @ti.func
    def _storage_slot(self, ref: ti.i32) -> ti.i32:
        tri_idx = PrimBVH._triangle_index(ref)
        return ti.select(ref < 0, self.analytic_count + tri_idx, ref)

    def __init__(
        self,
        primitives_field,
        prim_count: int,
        split: str = "sah",
        host_primitives=None,
        host_triangles=None,
        triangle_mat_id: int = 0,
        host_materials=None,
    ):
        debug_tools.log(
            f"PrimBVH init start: analytic_field_count={prim_count}, split={split}"
        )
        init_t0 = debug_tools.now() if debug_tools.enabled() else 0.0
        self.primitives = primitives_field
        self.analytic_count = prim_count
        self.has_sphere = False
        self.has_box = False
        self.has_triangle = False
        self.has_quad = False
        self.has_ellipsoid = False
        self.has_sdf = False
        self.use_compact_nodes = False
        host_t0 = debug_tools.now() if debug_tools.enabled() else 0.0
        host_prims = (
            host_primitives
            if host_primitives is not None
            else [primitives_field[i] for i in range(prim_count)]
        )
        mesh_triangles = list(host_triangles) if host_triangles is not None else []
        if debug_tools.enabled():
            debug_tools.log(
                f"host primitive inputs ready in {debug_tools.now() - host_t0:.3f}s: {len(host_prims):,} analytic candidates, {len(mesh_triangles):,} mesh triangles"
            )
        analytic_prims = []
        triangle_specs = []
        classify_t0 = debug_tools.now() if debug_tools.enabled() else 0.0
        for p in host_prims:
            if int(p.kind) == KIND_TRIANGLE:
                triangle_specs.append(
                    (
                        p.center0,
                        p.center0 + p.center_vel,
                        p.center0 + p.box_min,
                        p.mat_id,
                        p.n0,
                        p.n1,
                        p.n2,
                        p.uv0,
                        p.uv1,
                        p.uv2,
                        p.bvh_min,
                        p.bvh_max,
                        p.box_max,
                        int(p.sdf_kind & PRIM_FLAG_TWO_SIDED != 0),
                    )
                )
            else:
                analytic_prims.append(p)
        self.analytic_count = len(analytic_prims)
        for tri in mesh_triangles:
            triangle_specs.append(
                (
                    tri.p0,
                    tri.p1,
                    tri.p2,
                    triangle_mat_id + int(tri.material_id),
                    tri.n0,
                    tri.n1,
                    tri.n2,
                    tri.uv0,
                    tri.uv1,
                    tri.uv2,
                    None,
                    None,
                    tri.geo_normal,
                    1,
                )
            )
        self.triangle_count = len(triangle_specs)
        self.has_triangle = self.triangle_count > 0
        if self.triangle_count > 0 and mesh_triangles:
            expected = len(mesh_triangles)
            if self.triangle_count != expected:
                raise RuntimeError(
                    f"Triangle upload count mismatch: tri_count={self.triangle_count}, expected={expected}"
                )
            print(
                f"[PrimBVH] Triangle SOA upload validated: {self.triangle_count:,} triangles",
                file=__import__("sys").stderr,
            )
        if debug_tools.enabled():
            debug_tools.log(
                f"classified primitives in {debug_tools.now() - classify_t0:.3f}s: {len(analytic_prims):,} analytic, {self.triangle_count:,} triangles"
            )
        tri_shape = max(self.triangle_count, 1)
        self.tri_v0 = ti.Vector.field(
            3, dtype=ti.f32, shape=tri_shape, layout=ti.Layout.SOA
        )
        self.tri_e1 = ti.Vector.field(
            3, dtype=ti.f32, shape=tri_shape, layout=ti.Layout.SOA
        )
        self.tri_e2 = ti.Vector.field(
            3, dtype=ti.f32, shape=tri_shape, layout=ti.Layout.SOA
        )
        self.tri_normal = ti.Vector.field(
            3, dtype=ti.f32, shape=tri_shape, layout=ti.Layout.SOA
        )
        self.tri_plane_d = ti.field(dtype=ti.f32, shape=tri_shape)
        self.tri_n0 = ti.Vector.field(
            3, dtype=ti.f32, shape=tri_shape, layout=ti.Layout.SOA
        )
        self.tri_n1 = ti.Vector.field(
            3, dtype=ti.f32, shape=tri_shape, layout=ti.Layout.SOA
        )
        self.tri_n2 = ti.Vector.field(
            3, dtype=ti.f32, shape=tri_shape, layout=ti.Layout.SOA
        )
        self.tri_uv0 = ti.Vector.field(
            3, dtype=ti.f32, shape=tri_shape, layout=ti.Layout.SOA
        )
        self.tri_uv1 = ti.Vector.field(
            3, dtype=ti.f32, shape=tri_shape, layout=ti.Layout.SOA
        )
        self.tri_uv2 = ti.Vector.field(
            3, dtype=ti.f32, shape=tri_shape, layout=ti.Layout.SOA
        )
        self.tri_mat_id = ti.field(dtype=ti.i32, shape=tri_shape)
        self.tri_two_sided = ti.field(dtype=ti.i32, shape=tri_shape)
        mat_two_sided: dict[int, int] = {}
        if host_materials is not None:
            for mid, mat in enumerate(host_materials):
                mat_two_sided[mid] = int(getattr(mat, "two_sided", 0))
        prim_data = []
        scene_min = [float("inf"), float("inf"), float("inf")]
        scene_max = [float("-inf"), float("-inf"), float("-inf")]
        for i, p in enumerate(analytic_prims):
            kind = int(p.kind)
            self.has_sphere = self.has_sphere or kind == KIND_SPHERE
            self.has_box = self.has_box or kind == KIND_BOX
            self.has_quad = self.has_quad or kind == KIND_QUAD
            self.has_ellipsoid = self.has_ellipsoid or kind == KIND_ELLIPSOID
            self.has_sdf = self.has_sdf or kind == KIND_SDF
            for axis in range(3):
                scene_min[axis] = min(scene_min[axis], float(p.bvh_min[axis]))
                scene_max[axis] = max(scene_max[axis], float(p.bvh_max[axis]))
        if self.triangle_count > 0:
            tri_upload_t0 = debug_tools.now() if debug_tools.enabled() else 0.0
            n_tri = self.triangle_count
            tri_v0 = np.zeros((n_tri, 3), dtype=np.float32)
            tri_e1 = np.zeros((n_tri, 3), dtype=np.float32)
            tri_e2 = np.zeros((n_tri, 3), dtype=np.float32)
            tri_normal = np.zeros((n_tri, 3), dtype=np.float32)
            tri_plane_d = np.zeros((n_tri,), dtype=np.float32)
            tri_n0 = np.zeros((n_tri, 3), dtype=np.float32)
            tri_n1 = np.zeros((n_tri, 3), dtype=np.float32)
            tri_n2 = np.zeros((n_tri, 3), dtype=np.float32)
            tri_uv0 = np.zeros((n_tri, 3), dtype=np.float32)
            tri_uv1 = np.zeros((n_tri, 3), dtype=np.float32)
            tri_uv2 = np.zeros((n_tri, 3), dtype=np.float32)
            tri_mat = np.zeros((n_tri,), dtype=np.int32)
            tri_two_sided = np.zeros((n_tri,), dtype=np.int32)
            all_p0 = np.empty((n_tri, 3), dtype=np.float32)
            all_p1 = np.empty((n_tri, 3), dtype=np.float32)
            all_p2 = np.empty((n_tri, 3), dtype=np.float32)
            for i, spec in enumerate(triangle_specs):
                p0, p1, p2, mat_id = (spec[0], spec[1], spec[2], spec[3])
                all_p0[i, 0] = float(p0[0])
                all_p0[i, 1] = float(p0[1])
                all_p0[i, 2] = float(p0[2])
                all_p1[i, 0] = float(p1[0])
                all_p1[i, 1] = float(p1[1])
                all_p1[i, 2] = float(p1[2])
                all_p2[i, 0] = float(p2[0])
                all_p2[i, 1] = float(p2[1])
                all_p2[i, 2] = float(p2[2])
                n0_s, n1_s, n2_s = (spec[4], spec[5], spec[6])
                tri_n0[i, 0] = float(n0_s[0])
                tri_n0[i, 1] = float(n0_s[1])
                tri_n0[i, 2] = float(n0_s[2])
                tri_n1[i, 0] = float(n1_s[0])
                tri_n1[i, 1] = float(n1_s[1])
                tri_n1[i, 2] = float(n1_s[2])
                tri_n2[i, 0] = float(n2_s[0])
                tri_n2[i, 1] = float(n2_s[1])
                tri_n2[i, 2] = float(n2_s[2])
                uv0_s, uv1_s, uv2_s = (spec[7], spec[8], spec[9])
                tri_uv0[i] = (float(uv0_s[0]), float(uv0_s[1]), float(uv0_s[2]))
                tri_uv1[i] = (float(uv1_s[0]), float(uv1_s[1]), float(uv1_s[2]))
                tri_uv2[i] = (float(uv2_s[0]), float(uv2_s[1]), float(uv2_s[2]))
                tri_mat[i] = int(mat_id)
                tri_two_sided[i] = (
                    int(spec[13])
                    if len(spec) > 13
                    else mat_two_sided.get(int(mat_id), 0)
                )
                if len(spec) > 12 and spec[12] is not None:
                    gn = spec[12]
                    tri_normal[i, 0] = float(gn[0])
                    tri_normal[i, 1] = float(gn[1])
                    tri_normal[i, 2] = float(gn[2])
            tri_v0[:] = all_p0
            tri_e1[:] = all_p1 - all_p0
            tri_e2[:] = all_p2 - all_p0
            cross = np.cross(tri_e1, tri_e2)
            cross_len = np.linalg.norm(cross, axis=1, keepdims=True)
            computed_normal = np.where(cross_len > 1e-20, cross / cross_len, 0.0)
            unset = np.sum(np.abs(tri_normal), axis=1) < 1e-12
            tri_normal[unset] = computed_normal[unset]
            tri_plane_d[:] = np.sum(tri_normal * tri_v0, axis=1)
            normal_err = np.sum(tri_normal * computed_normal, axis=1)
            bad = (cross_len[:, 0] > 1e-20) & (normal_err < np.cos(np.radians(5.0)))
            if np.any(bad):
                raise RuntimeError(
                    f"Triangle normal validation failed for {int(np.sum(bad))} triangles (stored normal deviates >1° from cross(e1,e2) with {normal_err})"
                )
            all_mins = np.minimum(np.minimum(all_p0, all_p1), all_p2)
            all_maxs = np.maximum(np.maximum(all_p0, all_p1), all_p2)
            for axis in range(3):
                thin = all_maxs[:, axis] - all_mins[:, axis] < BVH_FLAT_EPS
                centers = 0.5 * (all_mins[:, axis] + all_maxs[:, axis])
                all_mins[thin, axis] = centers[thin] - 0.5 * BVH_FLAT_EPS
                all_maxs[thin, axis] = centers[thin] + 0.5 * BVH_FLAT_EPS
            for i, spec in enumerate(triangle_specs):
                bmn, bmx = (spec[10], spec[11])
                if bmn is not None and bmx is not None:
                    all_mins[i] = (float(bmn[0]), float(bmn[1]), float(bmn[2]))
                    all_maxs[i] = (float(bmx[0]), float(bmx[1]), float(bmx[2]))
            global_tri_min = all_mins.min(axis=0)
            global_tri_max = all_maxs.max(axis=0)
            for axis in range(3):
                scene_min[axis] = min(scene_min[axis], float(global_tri_min[axis]))
                scene_max[axis] = max(scene_max[axis], float(global_tri_max[axis]))
            self.tri_v0.from_numpy(tri_v0)
            self.tri_e1.from_numpy(tri_e1)
            self.tri_e2.from_numpy(tri_e2)
            self.tri_normal.from_numpy(tri_normal)
            self.tri_plane_d.from_numpy(tri_plane_d)
            self.tri_n0.from_numpy(tri_n0)
            self.tri_n1.from_numpy(tri_n1)
            self.tri_n2.from_numpy(tri_n2)
            self.tri_uv0.from_numpy(tri_uv0)
            self.tri_uv1.from_numpy(tri_uv1)
            self.tri_uv2.from_numpy(tri_uv2)
            self.tri_mat_id.from_numpy(tri_mat)
            self.tri_two_sided.from_numpy(tri_two_sided)
            if debug_tools.enabled():
                debug_tools.log(
                    f"triangle SOA arrays built/uploaded in {debug_tools.now() - tri_upload_t0:.3f}s ({self.triangle_count:,} triangles)"
                )
        self.bvh_prim_count = len(analytic_prims) + self.triangle_count
        if self.bvh_prim_count == 0:
            scene_min = [0.0, 0.0, 0.0]
            scene_max = [0.0, 0.0, 0.0]
        prim_mins = np.zeros((max(self.bvh_prim_count, 1), 3), dtype=np.float32)
        prim_maxs = np.zeros((max(self.bvh_prim_count, 1), 3), dtype=np.float32)
        prim_refs = np.zeros((max(self.bvh_prim_count, 1),), dtype=np.int32)
        
        for i, p in enumerate(analytic_prims):
            prim_mins[i] = (float(p.bvh_min[0]), float(p.bvh_min[1]), float(p.bvh_min[2]))
            prim_maxs[i] = (float(p.bvh_max[0]), float(p.bvh_max[1]), float(p.bvh_max[2]))
            prim_refs[i] = i
        
        if self.triangle_count > 0:
            ofst = len(analytic_prims)
            prim_mins[ofst:ofst + self.triangle_count] = all_mins
            prim_maxs[ofst:ofst + self.triangle_count] = all_maxs
            prim_refs[ofst:ofst + self.triangle_count] = -np.arange(self.triangle_count, dtype=np.int32) - 1

        storage_mins = np.zeros(
            (max(len(analytic_prims) + self.triangle_count, 1), 3), dtype=np.float32
        )
        storage_maxs = np.zeros(
            (max(len(analytic_prims) + self.triangle_count, 1), 3), dtype=np.float32
        )
        for i in range(self.bvh_prim_count):
            ref = prim_refs[i]
            storage_slot = int(ref) if int(ref) >= 0 else len(analytic_prims) + (-int(ref) - 1)
            storage_mins[storage_slot] = prim_mins[i]
            storage_maxs[storage_slot] = prim_maxs[i]
        self.prim_bvh_min = ti.Vector.field(
            3, dtype=ti.f32, shape=max(self.bvh_prim_count, 1), layout=ti.Layout.SOA
        )
        self.prim_bvh_max = ti.Vector.field(
            3, dtype=ti.f32, shape=max(self.bvh_prim_count, 1), layout=ti.Layout.SOA
        )
        self.prim_refs = ti.field(dtype=ti.i32, shape=max(self.bvh_prim_count, 1))
        self.storage_bvh_min = ti.Vector.field(
            3,
            dtype=ti.f32,
            shape=max(len(analytic_prims) + self.triangle_count, 1),
            layout=ti.Layout.SOA,
        )
        self.storage_bvh_max = ti.Vector.field(
            3,
            dtype=ti.f32,
            shape=max(len(analytic_prims) + self.triangle_count, 1),
            layout=ti.Layout.SOA,
        )
        self.prim_bvh_min.from_numpy(prim_mins)
        self.prim_bvh_max.from_numpy(prim_maxs)
        self.prim_refs.from_numpy(prim_refs)
        self.storage_bvh_min.from_numpy(storage_mins)
        self.storage_bvh_max.from_numpy(storage_maxs)
        if debug_tools.enabled():
            debug_tools.log(
                f"BVH bounds uploaded: bvh_prims={self.bvh_prim_count:,}, storage_slots={len(analytic_prims) + self.triangle_count:,}"
            )
        if split in ["sah", "binned_sah", "cpu", "spatial"]:
            split = "gpu_sah"

        self.indices_are_bvh_indices = split == "gpu"
        if split == "gpu":
            gpu_bvh_t0 = debug_tools.now() if debug_tools.enabled() else 0.0
            leaf_count = max(
                1, (self.bvh_prim_count + self.GPU_LEAF_SIZE - 1) // self.GPU_LEAF_SIZE
            )
            base = 1
            while base < leaf_count:
                base *= 2
            self.node_count = max(1, 2 * base - 1)
            self.nodes = BVHNode.field(
                shape=(self.node_count + 1,), layout=ti.Layout.SOA
            )
            self.prim_indices = ti.field(
                dtype=ti.i32, shape=(max(self.bvh_prim_count, 1),)
            )
            self.prim_count = self.bvh_prim_count
            self._build_gpu_morton_bvh(
                self.bvh_prim_count,
                base,
                vec3(scene_min[0], scene_min[1], scene_min[2]),
                vec3(
                    1.0 / max(scene_max[0] - scene_min[0], 1e-10),
                    1.0 / max(scene_max[1] - scene_min[1], 1e-10),
                    1.0 / max(scene_max[2] - scene_min[2], 1e-10),
                ),
            )
            if debug_tools.enabled():
                debug_tools.log(
                    f"PrimBVH GPU Morton build complete in {debug_tools.now() - gpu_bvh_t0:.3f}s: nodes={self.node_count:,}, prim_indices={self.prim_count:,}"
                )
                debug_tools.log(
                    f"PrimBVH init end ({debug_tools.now() - init_t0:.3f}s)"
                )
            return
        elif split == "gpu_sah":
            from time import perf_counter as _pf
            gpu_sah_t0 = _pf()
            if debug_tools.enabled():
                debug_tools.log("PrimBVH gpu_sah: starting Karras LBVH build")

            builder = GPUSAHBVHBuilder(prim_mins, prim_maxs, prim_refs, leaf_size=4)
            builder.build()
            gpu_sah_build_t = _pf() - gpu_sah_t0

            max_idx = max(builder.max_heap_idx, 1)
            self.node_count = max_idx
            n_prims = len(builder.ordered_prim_indices)
            self.prim_count = n_prims
            self.indices_are_bvh_indices = False

            upload_t0 = _pf()
            import numpy as _np
            node_shape = max_idx + 1
            self.use_compact_nodes = True
            self.nodes = None
            self.node_aabb_min = ti.Vector.field(
                3, dtype=ti.f32, shape=(node_shape,), layout=ti.Layout.SOA
            )
            self.node_aabb_max = ti.Vector.field(
                3, dtype=ti.f32, shape=(node_shape,), layout=ti.Layout.SOA
            )
            self.node_left_or_first = ti.field(dtype=ti.i32, shape=(node_shape,))
            self.node_count_or_zero = ti.field(dtype=ti.i32, shape=(node_shape,))
            h_mn = builder.heap_aabb_min
            h_mx = builder.heap_aabb_max
            h_lf = builder.heap_left_or_first
            h_c = builder.heap_count
            pad = node_shape - h_mn.shape[0]
            if pad > 0:
                h_mn = _np.pad(h_mn, ((0, pad), (0, 0)), constant_values=0.0)
                h_mx = _np.pad(h_mx, ((0, pad), (0, 0)), constant_values=0.0)
                h_lf = _np.pad(h_lf, (0, pad), constant_values=0)
                h_c = _np.pad(h_c, (0, pad), constant_values=0)
            self.node_aabb_min.from_numpy(h_mn[:node_shape])
            self.node_aabb_max.from_numpy(h_mx[:node_shape])
            self.node_left_or_first.from_numpy(h_lf[:node_shape])
            self.node_count_or_zero.from_numpy(h_c[:node_shape])

            self.prim_indices = ti.field(dtype=ti.i32, shape=(max(n_prims, 1),))
            prim_idx_np = _np.asarray(builder.ordered_prim_indices, dtype=_np.int32)
            self.prim_indices.from_numpy(prim_idx_np)

            upload_t = _pf() - upload_t0
            gpu_sah_total = _pf() - gpu_sah_t0
            print(
                f"[PrimBVH] Karras gpu_sah: {gpu_sah_build_t:.3f}s build + "
                f"{upload_t:.3f}s upload = {gpu_sah_total:.3f}s total "
                f"({self.bvh_prim_count:,} prims, {self.node_count:,} nodes, {self.prim_count:,} ordered)",
                flush=True,
            )
            if debug_tools.enabled():
                debug_tools.log(
                    f"PrimBVH GPU SAH build complete in {gpu_sah_total:.3f}s: "
                    f"nodes={self.node_count:,}, prim_indices={self.prim_count:,}"
                )
                debug_tools.log(f"PrimBVH init end ({_pf() - gpu_sah_t0:.3f}s)")
            return

        else:
            cpu_bvh_t0 = debug_tools.now() if debug_tools.enabled() else 0.0
            builder = BVHBuilder(prim_data, layout="prim", split=split)
            builder.build()
            if debug_tools.enabled():
                debug_tools.log(
                    f"PrimBVH CPU {split} builder complete in {debug_tools.now() - cpu_bvh_t0:.3f}s: ordered={len(builder.ordered_prim_indices):,}"
                )
            max_idx = max(builder.get_max_index(), 1)
            self.nodes = BVHNode.field(shape=(max_idx + 1,), layout=ti.Layout.SOA)
            self.node_count = max_idx
            n_prims = len(builder.ordered_prim_indices)
            self.prim_indices = ti.field(dtype=ti.i32, shape=(max(n_prims, 1),))
            self.prim_count = n_prims
            for i in range(1, max_idx + 1):
                node = builder.nodes[i]
                if node is not None:
                    self.nodes[i].aabb_min = vec3(
                        float(node.aabb_min[0]),
                        float(node.aabb_min[1]),
                        float(node.aabb_min[2]),
                    )
                    self.nodes[i].aabb_max = vec3(
                        float(node.aabb_max[0]),
                        float(node.aabb_max[1]),
                        float(node.aabb_max[2]),
                    )
                    if node.is_leaf:
                        self.nodes[i].left_or_first = int(node.first_prim)
                        self.nodes[i].count = int(node.prim_count)
                        self.nodes[i].prim_count = int(node.prim_count)
                    else:
                        self.nodes[i].left_or_first = 2 * i
                        self.nodes[i].count = 0
                        self.nodes[i].prim_count = 0
                else:
                    self.nodes[i].aabb_min = vec3(0, 0, 0)
                    self.nodes[i].aabb_max = vec3(0, 0, 0)
                    self.nodes[i].left_or_first = 0
                    self.nodes[i].count = 0
                    self.nodes[i].prim_count = 0
            for i, idx in enumerate(builder.ordered_prim_indices):
                self.prim_indices[i] = idx
            self.nodes[1].aabb_min = vec3(scene_min[0], scene_min[1], scene_min[2])
            self.nodes[1].aabb_max = vec3(scene_max[0], scene_max[1], scene_max[2])
            if debug_tools.enabled():
                debug_tools.log(
                    f"PrimBVH CPU fields uploaded: nodes={self.node_count:,}, prim_indices={self.prim_count:,}"
                )
                debug_tools.log(f"PrimBVH init end ({debug_tools.now() - init_t0:.3f}s)")

    def resident_memory_estimate_bytes(self) -> int:
        node_bytes = int(self.node_count + 1) * (
            (6 * 4 + 2 * 4) if self.use_compact_nodes else (6 * 4 + 3 * 4)
        )
        prim_index_bytes = max(int(self.prim_count), 1) * 4
        bvh_bounds_bytes = max(int(self.bvh_prim_count), 1) * (2 * 3 * 4 + 4)
        storage_bounds_bytes = max(
            int(self.analytic_count + self.triangle_count), 1
        ) * (2 * 3 * 4)
        triangle_soa_bytes = max(int(self.triangle_count), 1) * (11 * 3 * 4 + 4)
        analytic_soa_bytes = max(int(self.analytic_count), 1) * (12 * 3 * 4 + 4 * 4)
        return (
            node_bytes
            + prim_index_bytes
            + bvh_bounds_bytes
            + storage_bounds_bytes
            + triangle_soa_bytes
            + analytic_soa_bytes
        )

    def memory_summary(self) -> str:
        mb = self.resident_memory_estimate_bytes() / (1024.0 * 1024.0)
        return f"analytic={self.analytic_count:,}, triangles={self.triangle_count:,}, bvh_prims={self.bvh_prim_count:,}, nodes={self.node_count:,}, resident_est={mb:.2f} MiB"

    @ti.func
    def _primitive_hit_enabled(
        self, p: ScenePrimitive, r: Ray, ray_t: Interval, rec, xi: ti.i32, yi: ti.i32
    ) -> ti.u8:
        ret = ti.cast(0, ti.u8)
        skip_primary = (p.sdf_kind & PRIM_FLAG_INVISIBLE_PRIMARY != 0) & (r.kind == 0)
        skip_one_way = _skip_one_way_inside(p, r) != 0
        if not skip_primary | skip_one_way:
            if ti.static(self.has_sphere):
                if p.kind == KIND_SPHERE:
                    ret = _hit_sphere(p, r, ray_t, rec, xi, yi)
            if ti.static(self.has_box):
                if p.kind == KIND_BOX:
                    ret = _hit_box(p, r, ray_t, rec, xi, yi)
            if ti.static(self.has_triangle):
                if p.kind == KIND_TRIANGLE:
                    ret = _hit_triangle(p, r, ray_t, rec, xi, yi)
            if ti.static(self.has_quad):
                if p.kind == KIND_QUAD:
                    ret = _hit_quad(p, r, ray_t, rec, xi, yi)
            if ti.static(self.has_sdf):
                if p.kind == KIND_SDF:
                    ret = _hit_sdf(p, r, ray_t, rec, xi, yi)
            if ti.static(self.has_ellipsoid):
                if p.kind == KIND_ELLIPSOID:
                    ret = _hit_ellipsoid(p, r, ray_t, rec, xi, yi)
        return ret

    @ti.func
    def _primitive_any_hit_enabled(
        self, p: ScenePrimitive, r: Ray, ray_t: Interval
    ) -> ti.i32:
        ret = 0
        if _skip_one_way_inside(p, r) == 0:
            if ti.static(self.has_sphere):
                if p.kind == KIND_SPHERE:
                    ret = _occludes_sphere(p, r, ray_t)
            if ti.static(self.has_box):
                if p.kind == KIND_BOX:
                    ret = aabb_hit(p.box_min, p.box_max, r, ray_t.min, ray_t.max)
            if ti.static(self.has_triangle):
                if p.kind == KIND_TRIANGLE:
                    ret = _occludes_triangle(p, r, ray_t)
            if ti.static(self.has_quad):
                if p.kind == KIND_QUAD:
                    ret = _occludes_quad(p, r, ray_t)
            if ti.static(self.has_sdf):
                if p.kind == KIND_SDF:
                    ret = _occludes_sdf(p, r, ray_t)
            if ti.static(self.has_ellipsoid):
                if p.kind == KIND_ELLIPSOID:
                    ret = _occludes_ellipsoid(p, r, ray_t)
        return ret

    @staticmethod
    @ti.func
    def _expand_bits_gpu(v: ti.u32) -> ti.u32:
        v = v & ti.u32(1023)
        v = (v | v << ti.u32(16)) & ti.u32(50331903)
        v = (v | v << ti.u32(8)) & ti.u32(50393103)
        v = (v | v << ti.u32(4)) & ti.u32(51130563)
        v = (v | v << ti.u32(2)) & ti.u32(153391689)
        return v

    @staticmethod
    @ti.func
    def _morton3d_gpu(x: ti.f32, y: ti.f32, z: ti.f32) -> ti.u32:
        xx = PrimBVH._expand_bits_gpu(
            ti.cast(ti.math.clamp(x, 0.0, 1.0) * 1023.0, ti.u32)
        )
        yy = PrimBVH._expand_bits_gpu(
            ti.cast(ti.math.clamp(y, 0.0, 1.0) * 1023.0, ti.u32)
        )
        zz = PrimBVH._expand_bits_gpu(
            ti.cast(ti.math.clamp(z, 0.0, 1.0) * 1023.0, ti.u32)
        )
        return xx << ti.u32(2) | yy << ti.u32(1) | zz

    @ti.kernel
    def _compute_gpu_morton(
        self,
        codes: ti.template(),
        values: ti.template(),
        sort_count: ti.i32,
        scene_min: vec3,
        inv_extent: vec3,
    ):
        for i in range(sort_count):
            if i < self.prim_count:
                c = (self.prim_bvh_min[i] + self.prim_bvh_max[i]) * 0.5
                n = (c - scene_min) * inv_extent
                codes[i] = PrimBVH._morton3d_gpu(n.x, n.y, n.z)
                values[i] = i
            else:
                codes[i] = ti.u32(4294967295)
                values[i] = i

    @ti.kernel
    def _bitonic_sort_step(
        self,
        codes: ti.template(),
        values: ti.template(),
        sort_count: ti.i32,
        j: ti.i32,
        k: ti.i32,
    ):
        for i in range(sort_count):
            ixj = i ^ j
            if ixj > i:
                ci = codes[i]
                cj = codes[ixj]
                vi = values[i]
                vj = values[ixj]
                ascending = i & k == 0
                greater = (ci > cj) | (ci == cj) & (vi > vj)
                less = (ci < cj) | (ci == cj) & (vi < vj)
                swap = ti.select(ascending, greater, less)
                if swap:
                    codes[i] = cj
                    codes[ixj] = ci
                    values[i] = vj
                    values[ixj] = vi

    @ti.kernel
    def _copy_sorted_prim_indices(self, values: ti.template()):
        for i in range(self.prim_count):
            self.prim_indices[i] = values[i]

    @ti.kernel
    def _build_gpu_leaves(self, base: ti.i32):
        for leaf in range(base):
            node_idx = base + leaf
            start = leaf * ti.static(self.GPU_LEAF_SIZE)
            remain = self.prim_count - start
            count = ti.max(0, ti.min(ti.static(self.GPU_LEAF_SIZE), remain))
            mn = vec3(1e30, 1e30, 1e30)
            mx = vec3(-1e30, -1e30, -1e30)
            for k in ti.static(range(self.GPU_LEAF_SIZE)):
                if k < count:
                    bvh_idx = self.prim_indices[start + k]
                    mn = tm.min(mn, self.prim_bvh_min[bvh_idx])
                    mx = tm.max(mx, self.prim_bvh_max[bvh_idx])
            self.nodes[node_idx].aabb_min = mn
            self.nodes[node_idx].aabb_max = mx
            self.nodes[node_idx].left_or_first = start
            self.nodes[node_idx].count = count
            self.nodes[node_idx].prim_count = count

    @ti.kernel
    def _refit_gpu_level(self, start: ti.i32, end: ti.i32):
        for idx in range(start, end):
            left = self.nodes[idx * 2]
            right = self.nodes[idx * 2 + 1]
            self.nodes[idx].aabb_min = tm.min(left.aabb_min, right.aabb_min)
            self.nodes[idx].aabb_max = tm.max(left.aabb_max, right.aabb_max)
            self.nodes[idx].left_or_first = idx * 2
            self.nodes[idx].count = 0
            self.nodes[idx].prim_count = 0

    def _build_gpu_morton_bvh(
        self, prim_count: int, base: int, scene_min: vec3, inv_extent: vec3
    ) -> None:
        sort_count = 1
        while sort_count < max(prim_count, 1):
            sort_count *= 2
        debug_tools.log(
            f"GPU Morton BVH setup: prim_count={prim_count:,}, sort_count={sort_count:,}, leaf_base={base:,}"
        )
        codes = ti.field(dtype=ti.u32, shape=sort_count)
        values = ti.field(dtype=ti.i32, shape=sort_count)
        t0 = debug_tools.now() if debug_tools.enabled() else 0.0
        self._compute_gpu_morton(codes, values, sort_count, scene_min, inv_extent)
        if debug_tools.enabled():
            ti.sync()
            debug_tools.log(
                f"GPU Morton codes computed in {debug_tools.now() - t0:.3f}s"
            )
        k = 2
        sort_t0 = debug_tools.now() if debug_tools.enabled() else 0.0
        sort_passes = 0
        while k <= sort_count:
            j = k // 2
            while j > 0:
                self._bitonic_sort_step(codes, values, sort_count, j, k)
                sort_passes += 1
                j //= 2
            k *= 2
        if debug_tools.enabled():
            ti.sync()
            debug_tools.log(
                f"GPU bitonic sort finished in {debug_tools.now() - sort_t0:.3f}s ({sort_passes:,} kernel launches)"
            )
        copy_t0 = debug_tools.now() if debug_tools.enabled() else 0.0
        self._copy_sorted_prim_indices(values)
        if debug_tools.enabled():
            ti.sync()
            debug_tools.log(
                f"GPU sorted indices copied in {debug_tools.now() - copy_t0:.3f}s"
            )
        leaves_t0 = debug_tools.now() if debug_tools.enabled() else 0.0
        self._build_gpu_leaves(base)
        if debug_tools.enabled():
            ti.sync()
            debug_tools.log(f"GPU leaves built in {debug_tools.now() - leaves_t0:.3f}s")
        level_start = base // 2
        level_end = base
        refit_t0 = debug_tools.now() if debug_tools.enabled() else 0.0
        refit_levels = 0
        while level_start >= 1:
            self._refit_gpu_level(level_start, level_end)
            refit_levels += 1
            level_end = level_start
            level_start //= 2
        ti.sync()
        if debug_tools.enabled():
            debug_tools.log(
                f"GPU refit complete in {debug_tools.now() - refit_t0:.3f}s ({refit_levels:,} levels)"
            )

    @ti.func
    def _hit_triangle_soa(
        self, prim_idx: ti.i32, r: Ray, ray_t: Interval, rec, xi: ti.i32, yi: ti.i32
    ) -> ti.u8:
        v0 = self.tri_v0[prim_idx]
        e1 = self.tri_e1[prim_idx]
        e2 = self.tri_e2[prim_idx]
        geom_n = self.tri_normal[prim_idx]
        plane_d = self.tri_plane_d[prim_idx]
        two_sided = self.tri_two_sided[prim_idx]
        hit_ok, root, uu, vv = _mt_triangle_intersect(
            v0,
            e1,
            e2,
            geom_n,
            plane_d,
            r.origin,
            r.direction,
            ray_t.min,
            ray_t.max,
            two_sided,
        )
        valid = hit_ok != 0
        if valid:
            rec[xi, yi].t = root
            rec[xi, yi].p = r.at(root)
            rec[xi, yi].bary_u = uu
            rec[xi, yi].bary_v = vv
            shade_n = geom_n
            n0 = self.tri_n0[prim_idx]
            n1 = self.tri_n1[prim_idx]
            n2 = self.tri_n2[prim_idx]
            if tm.dot(n0, n0) + tm.dot(n1, n1) + tm.dot(n2, n2) > 1e-12:
                w0 = 1.0 - uu - vv
                raw_shade_n = w0 * n0 + uu * n1 + vv * n2
                shade_len2 = tm.dot(raw_shade_n, raw_shade_n)
                if shade_len2 > 1e-12:
                    shade_n = raw_shade_n / ti.sqrt(shade_len2)
                    if tm.dot(shade_n, geom_n) < 0.0:
                        shade_n = -shade_n
            rec[xi, yi].set_face_normal(r, geom_n)
            rec[xi, yi].geo_normal = geom_n
            if tm.dot(r.direction, geom_n) > 0.0:
                shade_n = -shade_n
            rec[xi, yi].normal = shade_n
            rec[xi, yi].mat_id = self.tri_mat_id[prim_idx]
            rec[xi, yi].prim_id = prim_idx
            tex_u = uu
            tex_v = vv
            uv0 = self.tri_uv0[prim_idx]
            uv1 = self.tri_uv1[prim_idx]
            uv2 = self.tri_uv2[prim_idx]
            if tm.dot(uv0, uv0) + tm.dot(uv1, uv1) + tm.dot(uv2, uv2) > 0.0:
                w0 = 1.0 - uu - vv
                uv = w0 * uv0 + uu * uv1 + vv * uv2
                tex_u = uv.x
                tex_v = uv.y
            rec[xi, yi].u = ti.math.clamp(tex_u, 0.0, 1.0)
            rec[xi, yi].v = ti.math.clamp(tex_v, 0.0, 1.0)
        return ti.select(valid, ti.cast(1, ti.u8), ti.cast(0, ti.u8))

    @ti.func
    def _occludes_triangle_soa(
        self, prim_idx: ti.i32, r: Ray, ray_t: Interval
    ) -> ti.i32:
        hit_ok, _, _, _ = _mt_triangle_intersect(
            self.tri_v0[prim_idx],
            self.tri_e1[prim_idx],
            self.tri_e2[prim_idx],
            self.tri_normal[prim_idx],
            self.tri_plane_d[prim_idx],
            r.origin,
            r.direction,
            ray_t.min,
            ray_t.max,
            self.tri_two_sided[prim_idx],
        )
        return hit_ok

    @ti.func
    def hit(self, r: Ray, ray_t: Interval, rec, x: ti.i32, y: ti.i32) -> ti.i32:
        did_hit = 0
        closest_so_far = ray_t.max
        stack = ti.Vector([0] * self.HIT_STACK_SIZE, dt=ti.i32)
        stack_ptr = 0
        current = 1
        step = 0
        safe_dir = vec3(
            ti.select(ti.abs(r.direction.x) < 1e-12, 1e-12, r.direction.x),
            ti.select(ti.abs(r.direction.y) < 1e-12, 1e-12, r.direction.y),
            ti.select(ti.abs(r.direction.z) < 1e-12, 1e-12, r.direction.z),
        )
        inv_dir = 1.0 / safe_dir
        while (step < self.MAX_HIT_STEPS) & ((current > 0) | (stack_ptr > 0)):
            step += 1
            if (current > 0) & (current <= self.node_count):
                node_idx = current
                current = 0
                node_min = vec3(0.0)
                node_max = vec3(0.0)
                left_or_first = 0
                count = 0
                if ti.static(self.use_compact_nodes):
                    node_min = self.node_aabb_min[node_idx]
                    node_max = self.node_aabb_max[node_idx]
                    left_or_first = self.node_left_or_first[node_idx]
                    count = self.node_count_or_zero[node_idx]
                else:
                    node = self.nodes[node_idx]
                    node_min = node.aabb_min
                    node_max = node.aabb_max
                    left_or_first = node.left_or_first
                    count = node.count
                if (
                    aabb_hit_inv(node_min, node_max, r.origin, inv_dir, ray_t.min, closest_so_far)
                    != 0
                ):
                    if count > 0:
                        for i in range(count):
                            raw_ref = self.prim_indices[left_or_first + i]
                            prim_ref = raw_ref
                            if ti.static(self.indices_are_bvh_indices):
                                prim_ref = self.prim_refs[raw_ref]
                            hit_ok = ti.cast(0, ti.u8)
                            if PrimBVH._is_triangle_ref(prim_ref) != 0:
                                hit_ok = self._hit_triangle_soa(
                                    PrimBVH._triangle_index(prim_ref),
                                    r,
                                    Interval(ray_t.min, closest_so_far),
                                    rec,
                                    x,
                                    y,
                                )
                            else:
                                prim = self.primitives[prim_ref]
                                hit_ok = self._primitive_hit_enabled(
                                    prim,
                                    r,
                                    Interval(ray_t.min, closest_so_far),
                                    rec,
                                    x,
                                    y,
                                )
                            if hit_ok != 0:
                                did_hit = 1
                                closest_so_far = rec[x, y].t
                                rec[x, y].prim_id = prim_ref
                    else:
                        left_child = left_or_first
                        right_child = left_or_first + 1
                        has_l = left_child <= self.node_count
                        has_r = right_child <= self.node_count
                        if has_l & has_r:
                            lmin = vec3(0.0)
                            lmax = vec3(0.0)
                            rmin = vec3(0.0)
                            rmax = vec3(0.0)
                            if ti.static(self.use_compact_nodes):
                                lmin = self.node_aabb_min[left_child]
                                lmax = self.node_aabb_max[left_child]
                                rmin = self.node_aabb_min[right_child]
                                rmax = self.node_aabb_max[right_child]
                            else:
                                nl = self.nodes[left_child]
                                nr = self.nodes[right_child]
                                lmin = nl.aabb_min
                                lmax = nl.aabb_max
                                rmin = nr.aabb_min
                                rmax = nr.aabb_max
                            t_l = aabb_ray_t_enter_inv(
                                lmin, lmax, r.origin, inv_dir, ray_t.min, closest_so_far
                            )
                            t_r = aabb_ray_t_enter_inv(
                                rmin, rmax, r.origin, inv_dir, ray_t.min, closest_so_far
                            )
                            hit_l = t_l < 1e29
                            hit_r = t_r < 1e29
                            if hit_l & hit_r:
                                if t_l < t_r:
                                    if stack_ptr < self.HIT_STACK_SIZE - 1:
                                        stack[stack_ptr] = right_child
                                        stack_ptr += 1
                                    current = left_child
                                else:
                                    if stack_ptr < self.HIT_STACK_SIZE - 1:
                                        stack[stack_ptr] = left_child
                                        stack_ptr += 1
                                    current = right_child
                                continue
                            if hit_l:
                                current = left_child
                                continue
                            if hit_r:
                                current = right_child
                                continue
                            continue
                        if has_l & (has_r == 0):
                            current = left_child
                            continue
                        if has_r & (has_l == 0):
                            current = right_child
                            continue
            if stack_ptr > 0:
                stack_ptr -= 1
                current = stack[stack_ptr]
            else:
                break
        return did_hit

    @ti.func
    def any_hit(self, r: Ray, ray_t: Interval, ignore_prim: ti.i32) -> ti.i32:
        blocked = 0
        stack = ti.Vector([0] * self.SHADOW_STACK_SIZE, dt=ti.i32)
        stack_ptr = 0
        current = 1
        t_max = ray_t.max
        step = 0
        safe_dir = vec3(
            ti.select(ti.abs(r.direction.x) < 1e-12, 1e-12, r.direction.x),
            ti.select(ti.abs(r.direction.y) < 1e-12, 1e-12, r.direction.y),
            ti.select(ti.abs(r.direction.z) < 1e-12, 1e-12, r.direction.z),
        )
        inv_dir = 1.0 / safe_dir
        while (
            (step < self.MAX_SHADOW_STEPS)
            & (blocked == 0)
            & ((current > 0) | (stack_ptr > 0))
        ):
            step += 1
            if blocked != 0:
                break
            if (current > 0) & (current <= self.node_count):
                node_idx = current
                current = 0
                node_min = vec3(0.0)
                node_max = vec3(0.0)
                left_or_first = 0
                count = 0
                if ti.static(self.use_compact_nodes):
                    node_min = self.node_aabb_min[node_idx]
                    node_max = self.node_aabb_max[node_idx]
                    left_or_first = self.node_left_or_first[node_idx]
                    count = self.node_count_or_zero[node_idx]
                else:
                    node = self.nodes[node_idx]
                    node_min = node.aabb_min
                    node_max = node.aabb_max
                    left_or_first = node.left_or_first
                    count = node.count
                if aabb_hit_inv(node_min, node_max, r.origin, inv_dir, ray_t.min, t_max) != 0:
                    if count > 0:
                        for i in range(count):
                            raw_ref = self.prim_indices[left_or_first + i]
                            prim_ref = raw_ref
                            if ti.static(self.indices_are_bvh_indices):
                                prim_ref = self.prim_refs[raw_ref]
                            if prim_ref != ignore_prim:
                                bbox_slot = self._storage_slot(prim_ref)
                                if (
                                    aabb_hit_inv(
                                        self.storage_bvh_min[bbox_slot],
                                        self.storage_bvh_max[bbox_slot],
                                        r.origin,
                                        inv_dir,
                                        ray_t.min,
                                        t_max,
                                    )
                                    != 0
                                ):
                                    shadow_hit = 0
                                    if PrimBVH._is_triangle_ref(prim_ref) != 0:
                                        shadow_hit = self._occludes_triangle_soa(
                                            PrimBVH._triangle_index(prim_ref), r, ray_t
                                        )
                                    else:
                                        prim = self.primitives[prim_ref]
                                        shadow_hit = self._primitive_any_hit_enabled(
                                            prim, r, ray_t
                                        )
                                    if shadow_hit != 0:
                                        blocked = 1
                                        break
                    else:
                        left_child = left_or_first
                        right_child = left_or_first + 1
                        has_l = left_child <= self.node_count
                        has_r = right_child <= self.node_count
                        if has_l & has_r:
                            lmin = vec3(0.0)
                            lmax = vec3(0.0)
                            rmin = vec3(0.0)
                            rmax = vec3(0.0)
                            if ti.static(self.use_compact_nodes):
                                lmin = self.node_aabb_min[left_child]
                                lmax = self.node_aabb_max[left_child]
                                rmin = self.node_aabb_min[right_child]
                                rmax = self.node_aabb_max[right_child]
                            else:
                                nl = self.nodes[left_child]
                                nr = self.nodes[right_child]
                                lmin = nl.aabb_min
                                lmax = nl.aabb_max
                                rmin = nr.aabb_min
                                rmax = nr.aabb_max
                            t_l = aabb_ray_t_enter_inv(
                                lmin, lmax, r.origin, inv_dir, ray_t.min, t_max
                            )
                            t_r = aabb_ray_t_enter_inv(
                                rmin, rmax, r.origin, inv_dir, ray_t.min, t_max
                            )
                            hit_l = t_l < 1e29
                            hit_r = t_r < 1e29
                            if hit_l & hit_r:
                                if t_l < t_r:
                                    if stack_ptr < self.SHADOW_STACK_SIZE - 1:
                                        stack[stack_ptr] = right_child
                                        stack_ptr += 1
                                    current = left_child
                                else:
                                    if stack_ptr < self.SHADOW_STACK_SIZE - 1:
                                        stack[stack_ptr] = left_child
                                        stack_ptr += 1
                                    current = right_child
                                continue
                            if hit_l:
                                current = left_child
                                continue
                            if hit_r:
                                current = right_child
                                continue
                        if has_l & (has_r == 0):
                            current = left_child
                            continue
                        if has_r & (has_l == 0):
                            current = right_child
                            continue
            if stack_ptr > 0:
                stack_ptr -= 1
                current = stack[stack_ptr]
            else:
                break
        return blocked
