"""Tonemapping and color sanitization (shared by Camera kernels)."""

import taichi as ti
import taichi.math as tm

vec3 = tm.vec3


@ti.func
def sanitize_rgb(c: vec3) -> vec3:
    """Remove NaN/Inf fireflies that show up as green or magenta speckles."""
    bad = (c.x != c.x) | (c.y != c.y) | (c.z != c.z)
    big = (ti.abs(c.x) > 1e8) | (ti.abs(c.y) > 1e8) | (ti.abs(c.z) > 1e8)
    return ti.select(bad | big, vec3(0.0, 0.0, 0.0), tm.clamp(c, 0.0, 1e4))


@ti.func
def tonemap_filmic(c: vec3, exposure: ti.f32) -> vec3:
    """ACES filmic tonemap + gamma (stable contrast, no blown ceiling)."""
    x = sanitize_rgb(c * exposure)
    a = 2.51
    b = 0.03
    c_ = 2.43
    d = 0.59
    e = 0.14
    mapped = tm.clamp((x * (a * x + b)) / (x * (c_ * x + d) + e), 0.0, 1.0)
    return tm.pow(mapped, vec3(1.0 / 2.2))
