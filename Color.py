import taichi as ti
import taichi.math as tm

from interval import Interval

color = ti.math.vec3


@ti.func
def transform_vec(v: tm.vec3) -> tm.vec3:
    """
    Apply gamma correction to a color vector using gamma = 2.0.

    Applies square root to each positive component (equivalent to gamma 1/2),
    clamping negative values to 0.

    Args:
        v: RGB color vector with components in any range.

    Returns:
        Gamma-corrected color vector with sqrt applied to each positive component.
    """
    return tm.vec3(
        ti.select(v.x > 0.0, tm.sqrt(v.x), 0.0),
        ti.select(v.y > 0.0, tm.sqrt(v.y), 0.0),
        ti.select(v.z > 0.0, tm.sqrt(v.z), 0.0),
    )


@ti.func
def float_to_rgb8(col: color) -> color:
    """
    Convert a normalized RGB color [0, 1] to 8-bit RGB [0, 255].

    Applies gamma correction, clamps values to valid range, and converts
    to 8-bit unsigned integer format.

    Args:
        col: RGB color vector with values in [0, 1].

    Returns:
        RGB color vector with 8-bit values [0, 255] as unsigned integers.
    """
    intensity = Interval(0, 0.9999)
    return ti.cast(intensity.clamp(transform_vec(col)) * 256, ti.u8)
