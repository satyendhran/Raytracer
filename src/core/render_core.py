import taichi as ti
import taichi.math as tm

vec3 = tm.vec3


@ti.func
def sanitize_rgb(c: vec3) -> vec3:
    bad = (c.x != c.x) | (c.y != c.y) | (c.z != c.z)
    big = (
        (ti.abs(c.x) > 100000000.0)
        | (ti.abs(c.y) > 100000000.0)
        | (ti.abs(c.z) > 100000000.0)
    )
    return ti.select(bad | big, vec3(0.0, 0.0, 0.0), tm.clamp(c, 0.0, 10000.0))


@ti.func
def tonemap_filmic(c: vec3, exposure: ti.f32) -> vec3:
    x = sanitize_rgb(c * exposure)
    a = 2.51
    b = 0.03
    c_ = 2.43
    d = 0.59
    e = 0.14
    mapped = tm.clamp(x * (a * x + b) / (x * (c_ * x + d) + e), 0.0, 1.0)
    return tm.pow(mapped, vec3(1.0 / 2.2))
