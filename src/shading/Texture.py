import taichi as ti


@ti.func
def _spherical_uv_from_normal(n: ti.math.vec3) -> ti.math.vec2:
    theta = ti.acos(ti.math.clamp(n.y, -1.0, 1.0))
    phi = ti.atan2(n.z, n.x)
    u = (phi + ti.math.pi) / (2.0 * ti.math.pi)
    v = theta / ti.math.pi
    return ti.math.vec2(u, v)


@ti.func
def _triplanar_blend(n: ti.math.vec3) -> ti.math.vec3:
    an = ti.abs(n)
    w = an / ti.max(an.x + an.y + an.z, 1e-08)
    return w


@ti.dataclass
class Texture:
    texture_type: ti.u8
    color1: ti.math.vec3
    color2: ti.math.vec3
    scale: ti.f32

    @ti.func
    def value(
        self, u: ti.f32, v: ti.f32, p: ti.math.vec3, n: ti.math.vec3
    ) -> ti.math.vec3:
        color = self.color1
        if self.texture_type == 0:
            color = self.color1
        elif self.texture_type == 1:
            inv_scale = 1.0 / self.scale
            xi = ti.cast(ti.floor(inv_scale * p.x), ti.i32)
            yi = ti.cast(ti.floor(inv_scale * p.y), ti.i32)
            zi = ti.cast(ti.floor(inv_scale * p.z), ti.i32)
            is_even = (xi + yi + zi) % 2 == 0
            color = ti.select(is_even, self.color1, self.color2)
        elif self.texture_type == 2:
            u_int = ti.cast(ti.floor(self.scale * u), ti.i32)
            v_int = ti.cast(ti.floor(self.scale * v), ti.i32)
            is_even = (u_int + v_int) % 2 == 0
            color = ti.select(is_even, self.color1, self.color2)
        else:
            nn = ti.math.normalize(n + ti.math.vec3(1e-20, 1e-20, 1e-20))
            suv = _spherical_uv_from_normal(nn)
            if self.texture_type == 3:
                tu = suv.x
                tv = suv.y
                su = ti.sin(self.scale * tu * 2.0 * ti.math.pi)
                sv = ti.sin(self.scale * tv * 3.0 * ti.math.pi)
                rings = 0.5 + 0.5 * ti.sin((tu * 6.28 + tv * 4.2) * self.scale * 2.0)
                blend = ti.math.clamp(
                    0.35 * su * su + 0.35 * sv * sv + 0.3 * rings, 0.0, 1.0
                )
                color = self.color1 * (1.0 - blend) + self.color2 * blend
            elif self.texture_type == 4:
                tw = _triplanar_blend(nn)
                tv = suv.y * tw.x + u * tw.y + suv.y * tw.z
                t = 0.5 + 0.5 * ti.sin(self.scale * tv * 2.0 * ti.math.pi)
                color = self.color1 * (1.0 - t) + self.color2 * t
            elif self.texture_type == 5:
                tw = _triplanar_blend(nn)
                nx = p.x * tw.x + nn.x * 0.35 * tw.y + p.x * tw.z
                ny = p.y * tw.x + p.y * tw.y + nn.y * 0.35 * tw.z
                nz = p.z * tw.x + p.z * tw.y + p.z * tw.z
                noise_val = ti.sin(
                    self.scale
                    * (
                        nx
                        + 2.0 * ti.sin(self.scale * ny * 0.7)
                        + 1.5 * ti.sin(self.scale * nz * 0.9)
                    )
                )
                t = 0.5 + 0.5 * noise_val
                color = self.color1 * (1.0 - t) + self.color2 * t
        return color
