import taichi as ti

from src.core.interval import Interval

vec3 = ti.math.vec3


@ti.dataclass
class AABB:
    x: Interval
    y: Interval
    z: Interval

    @ti.func
    def axis_interval(self, n):
        return ti.select(n == 0, self.x, ti.select(n == 1, self.y, self.z))

    @ti.func
    def hit(self, r, ray_t):
        ray_origin = r.origin
        ray_dir = r.direction
        no_hit = False
        for axis in ti.static(range(3)):
            ax = self.axis_interval(axis)
            d = ray_dir[axis]
            if ti.abs(d) < 1e-12:
                if (ray_origin[axis] < ax.min) | (ray_origin[axis] > ax.max):
                    no_hit = True
            else:
                adinv = 1.0 / d
                t0 = (ax.min - ray_origin[axis]) * adinv
                t1 = (ax.max - ray_origin[axis]) * adinv
                if t0 > t1:
                    t0, t1 = (t1, t0)
                if t0 > ray_t[None].min:
                    ray_t[None].min = t0
                if t1 < ray_t[None].max:
                    ray_t[None].max = t1
                if ray_t[None].min >= ray_t[None].max:
                    no_hit = True
        return ti.select(no_hit, 0, 1)


@ti.func
def init_aabb(a, b):
    x = Interval(a[0], b[0]) if a[0] <= b[0] else Interval(b[0], a[0])
    y = Interval(a[1], b[1]) if a[1] <= b[1] else Interval(b[1], a[1])
    z = Interval(a[2], b[2]) if a[2] <= b[2] else Interval(b[2], a[2])
    return AABB(x, y, z)


@ti.func
def init_aabb_2(box1, box2):
    return init_aabb(
        vec3(
            min(box1.x.min, box2.x.min),
            min(box1.y.min, box2.y.min),
            min(box1.z.min, box2.z.min),
        ),
        vec3(
            max(box1.x.max, box2.x.max),
            max(box1.y.max, box2.y.max),
            max(box1.z.max, box2.z.max),
        ),
    )
