import taichi as ti

@ti.dataclass
class Ray:
    origin: ti.math.vec3
    direction: ti.math.vec3
    tm: ti.f32
    kind: ti.i32

    @ti.func
    def at(self, t: ti.f32) -> ti.math.vec3:
        return self.origin + t * self.direction