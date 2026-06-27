import taichi as ti

@ti.dataclass
class Interval:
    min: ti.f32
    max: ti.f32

    @ti.func
    def size(self):
        return self.max - self.min

    @ti.func
    def contains(self, x: ti.f32):
        return self.min <= x <= self.max

    @ti.func
    def surrounds(self, x: ti.f32):
        return self.min < x < self.max

    @ti.func
    def clamp(self, x: ti.f32):
        return ti.math.clamp(x, self.min, self.max)

    @ti.func
    def expand(self, delta: ti.f32):
        padding = delta / 2
        return Interval(self.min - padding, self.max + padding)