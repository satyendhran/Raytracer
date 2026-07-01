import taichi as ti
import taichi.math as tm

from src.core.interval import Interval
from src.core.Ray import Ray
from src.geometry.AABB import AABB


@ti.dataclass
class hit_record:
    p: ti.math.vec3
    normal: ti.math.vec3
    geo_normal: ti.math.vec3
    t: ti.f32
    u: ti.f32
    v: ti.f32
    bary_u: ti.f32
    bary_v: ti.f32
    front_face: ti.u8
    mat_id: ti.i32
    prim_id: ti.i32

    @ti.func
    def set_face_normal(self, r: Ray, outward_normal: ti.math.vec3):
        self.front_face = ti.u8(ti.math.dot(outward_normal, r.direction) < 0)
        self.normal = ti.select(self.front_face, outward_normal, -outward_normal)


@ti.func
def set(set_rec: hit_record, other_rec: hit_record, i, j):
    set_rec[i, j].t = other_rec[None].t
    set_rec[i, j].p = other_rec[None].p
    set_rec[i, j].normal = other_rec[None].normal
    set_rec[i, j].geo_normal = other_rec[None].geo_normal
    set_rec[i, j].u = other_rec[None].u
    set_rec[i, j].v = other_rec[None].v
    set_rec[i, j].bary_u = other_rec[None].bary_u
    set_rec[i, j].bary_v = other_rec[None].bary_v
    set_rec[i, j].front_face = other_rec[None].front_face
    set_rec[i, j].mat_id = other_rec[None].mat_id
    set_rec[i, j].prim_id = other_rec[None].prim_id


@ti.func
def get_sphere_uv(p: ti.math.vec3):
    theta = ti.acos(ti.math.clamp(p.y, -1.0, 1.0))
    phi = ti.atan2(p.z, p.x)
    u = (phi + ti.math.pi) / (2.0 * ti.math.pi)
    v = theta / ti.math.pi
    return (u, v)


@ti.dataclass
class Sphere:
    center: Ray
    radius: ti.f32
    mat_id: ti.i32
    aabb: AABB

    @ti.func
    def hit(self, r: Ray, ray_t: Interval, rec) -> ti.u1:
        ret = 0
        current_centre = self.center.at(r.tm)
        oc = current_centre - r.origin
        a = tm.dot(r.direction, r.direction)
        h = tm.dot(r.direction, oc)
        c = tm.dot(oc, oc) - self.radius * self.radius
        D = h * h - a * c
        if D >= 0.0:
            sqrtD = ti.math.sqrt(ti.max(D, 0.0))
            root = (h - sqrtD) / a
            if not ray_t.surrounds(root):
                root = (h + sqrtD) / a
                if not ray_t.surrounds(root):
                    D = -1.0
            if D >= 0.0:
                ret = 1
                rec[None].t = root
                rec[None].p = r.at(root)
                outward_normal = (rec[None].p - current_centre) / self.radius
                rec[None].u, rec[None].v = get_sphere_uv(outward_normal)
                rec[None].set_face_normal(r, outward_normal)
                rec[None].mat_id = self.mat_id
                rec[None].prim_id = -1
        return ret


@ti.data_oriented
class Hittable_list:
    def __init__(self, no_of_spheres):
        self.spheres = Sphere.field(shape=no_of_spheres)
        self.sphere_counter = ti.field(dtype=ti.i32, shape=())
        self.sphere_counter[None] = 0
        self.temp_rec = hit_record.field(shape=())
        self.bbox = None

    @ti.kernel
    def add_sphere(self, sphere: Sphere):
        if self.sphere_counter[None] >= self.spheres.shape[0]:
            ...
        else:
            self.spheres[self.sphere_counter[None]] = sphere
            self.sphere_counter[None] += 1

    def get_sphere_count(self):
        return self.sphere_counter[None]

    def get_spheres_field(self):
        return self.spheres

    @ti.func
    def clear(self):
        self.sphere_counter = 0

    @ti.func
    def hit(self, r: Ray, ray_t: Interval, rec, x, y):
        did_hit = False
        closest_so_far = ray_t.max
        for i in range(self.sphere_counter[None]):
            if self.spheres[i].hit(
                r, Interval(ray_t.min, closest_so_far), self.temp_rec
            ):
                did_hit = True
                closest_so_far = self.temp_rec[None].t
                set(rec, self.temp_rec, x, y)
        return did_hit
