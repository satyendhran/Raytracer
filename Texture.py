import taichi as ti

@ti.dataclass
class Texture:
    """
    Represents a texture that can provide color values at surface points.

    Supports multiple texture types:
    - Solid color (type 0): Returns constant color
    - Spatial Checker (type 1): 3D world-space checker pattern
    - UV Checker (type 2): Surface-space checker pattern (for spheres)

    Attributes:
        texture_type: Type identifier (0=Solid, 1=Spatial Checker, 2=UV Checker)
        color1: Primary color (for solid) or first checker color (even)
        color2: Secondary color (used for checker pattern, odd)
        scale: Scale factor for checker pattern
    """
    texture_type: ti.u8
    color1: ti.math.vec3
    color2: ti.math.vec3
    scale: ti.f32

    @ti.func
    def value(self, u: ti.f32, v: ti.f32, p: ti.math.vec3) -> ti.math.vec3:
        color = ti.math.vec3(0.0, 0.0, 0.0)


        if self.texture_type == 0:
            color = self.color1


        if self.texture_type == 1:
            inv_scale = 1.0 / self.scale
            x_integer = ti.cast(ti.floor(inv_scale * p.x), ti.i32)
            y_integer = ti.cast(ti.floor(inv_scale * p.y), ti.i32)
            z_integer = ti.cast(ti.floor(inv_scale * p.z), ti.i32)

            is_even = (x_integer + y_integer + z_integer) % 2 == 0
            color = ti.select(is_even, self.color1, self.color2)


        if self.texture_type == 2:
            u_int = int(ti.floor(self.scale * u))
            v_int = int(ti.floor(self.scale * v))

            is_even = (u_int + v_int) % 2 == 0
            color = ti.select(is_even, self.color1, self.color2)

        return color