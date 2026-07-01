import taichi as ti
import taichi.math as tm

from src.core.interval import Interval

color = ti.math.vec3


@ti.func
def transform_vec(v: tm.vec3) -> tm.vec3:
    return tm.vec3(
        ti.select(v.x > 0.0, tm.sqrt(v.x), 0.0),
        ti.select(v.y > 0.0, tm.sqrt(v.y), 0.0),
        ti.select(v.z > 0.0, tm.sqrt(v.z), 0.0),
    )


@ti.func
def float_to_rgb8(col: color) -> color:
    intensity = Interval(0, 0.9999)
    return ti.cast(intensity.clamp(col) * 256, ti.u8)
