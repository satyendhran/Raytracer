import taichi as ti
import taichi.math as tm

from AABB import AABB
from interval import Interval
from Ray import Ray


@ti.dataclass
class hit_record:
    """
    Stores information about a ray–object intersection.

    Attributes
    ----------
    p : ti.math.vec3
        Point of intersection in world coordinates.
    normal : ti.math.vec3
        Surface normal at the intersection point.
    t : ti.f32
        Ray parameter at the intersection.
    u : ti.f32
        Horizontal texture coordinate [0, 1].
    v : ti.f32
        Vertical texture coordinate [0, 1].
    front_face : ti.u8
        Flag indicating whether the hit was on the front face (1) or back face (0).
    mat_id : ti.i32
        Material ID index for the intersected object.
    """

    p: ti.math.vec3
    normal: ti.math.vec3
    t: ti.f32
    u: ti.f32
    v: ti.f32
    front_face: ti.u8
    mat_id: ti.i32

    @ti.func
    def set_face_normal(self, r: Ray, outward_normal: ti.math.vec3):
        """
        Sets the correct surface normal orientation based on ray direction.

        The normal is flipped if the ray hits the surface from inside.

        Parameters
        ----------
        r : Ray
            Incoming ray.
        outward_normal : ti.math.vec3
            Geometric outward normal of the surface.

        Returns
        -------
        None
        """
        self.front_face = ti.u8(ti.math.dot(outward_normal, r.direction) < 0)
        self.normal = ti.select(self.front_face, outward_normal, -outward_normal)


@ti.func
def set(set_rec: hit_record, other_rec: hit_record, i, j):
    """
    Copies hit record data from a temporary record into a target record field.
    """
    set_rec[i, j].t = other_rec[None].t
    set_rec[i, j].p = other_rec[None].p
    set_rec[i, j].normal = other_rec[None].normal
    set_rec[i, j].u = other_rec[None].u
    set_rec[i, j].v = other_rec[None].v
    set_rec[i, j].front_face = other_rec[None].front_face
    set_rec[i, j].mat_id = other_rec[None].mat_id


@ti.func
def get_sphere_uv(p: ti.math.vec3):
    """
    Calculate UV coordinates for a point on a unit sphere.
    p is the outward normal (normalized vector from center to surface point)
    """
    
    
    theta = ti.acos(ti.math.clamp(p.y, -1.0, 1.0))

    
    phi = ti.atan2(p.z, p.x)

    
    u = (phi + ti.math.pi) / (2.0 * ti.math.pi)
    v = theta / ti.math.pi

    return u, v

@ti.dataclass
class Sphere:
    """
    Represents a sphere primitive for ray intersection tests.

    Supports animated spheres with time-dependent positions.

    Attributes
    ----------
    center : Ray
        Center position of the sphere (Ray.origin is center, Ray.direction is velocity).
    radius : ti.f32
        Radius of the sphere.
    mat_id : ti.i32
        Material ID index for rendering.
    aabb : AABB
        Axis-aligned bounding box for acceleration structure.
    """

    center: Ray
    radius: ti.f32
    mat_id: ti.i32
    aabb: AABB

    @ti.func
    def hit(self, r: Ray, ray_t: Interval, rec) -> ti.u1:
        ret = 1
        current_centre = self.center.at(r.tm)
        oc = current_centre - r.origin
        a = tm.dot(r.direction, r.direction)
        h = tm.dot(r.direction, oc)
        c = tm.dot(oc, oc) - self.radius * self.radius
        D = h * h - a * c

        
        if D < 0.0:
            ret = 0

        sqrtD = ti.math.sqrt(D)

        
        root = (h - sqrtD) / a
        if not ray_t.surrounds(root):
            root = (h + sqrtD) / a
            if not ray_t.surrounds(root):
                ret = 0  

        if ret:
            
            rec[None].t = root
            rec[None].p = r.at(root)
            outward_normal = (rec[None].p - current_centre) / self.radius

            
            rec[None].u, rec[None].v = get_sphere_uv(outward_normal)
            
            rec[None].set_face_normal(r, outward_normal)
            rec[None].mat_id = self.mat_id

        return ret


@ti.data_oriented
class Hittable_list:
    """
    Collection of hittable objects supporting closest-hit queries.

    Maintains a dynamic list of spheres and provides ray intersection
    testing against all contained objects.
    """

    def __init__(self, no_of_spheres):
        """
        Initializes a fixed-size container for spheres.

        Parameters
        ----------
        no_of_spheres : int
            Maximum number of spheres that can be stored.
        """
        self.spheres = Sphere.field(shape=no_of_spheres)
        self.sphere_counter = ti.field(dtype=ti.i32, shape=())
        self.sphere_counter[None] = 0
        self.temp_rec = hit_record.field(shape=())
        
        self.bbox = None

    @ti.kernel
    def add_sphere(self, sphere: Sphere):
        """
        Adds a sphere to the list if capacity permits.

        Parameters
        ----------
        sphere : Sphere
            Sphere to be added.

        Returns
        -------
        None
        """
        if self.sphere_counter[None] >= self.spheres.shape[0]:
            ...
        else:
            self.spheres[self.sphere_counter[None]] = sphere
            self.sphere_counter[None] += 1
        

    def get_sphere_count(self):
        """
        Return the current number of spheres.

        Returns
        -------
        int
            Number of spheres currently in the list.
        """
        return self.sphere_counter[None]

    def get_spheres_field(self):
        """
        Return the spheres field for BVH construction.

        Returns
        -------
        ti.Field
            Taichi field containing all sphere data.
        """
        return self.spheres

    @ti.func
    def clear(self):
        """
        Resets the list, removing all spheres.

        Returns
        -------
        None
        """
        self.sphere_counter = 0

    @ti.func
    def hit(self, r: Ray, ray_t: Interval, rec, x, y):
        """
        Finds the closest intersection between a ray and any sphere in the list.

        Iterates through all spheres and returns the closest hit.

        Parameters
        ----------
        r : Ray
            Ray to test for intersections.
        ray_t : Interval
            Valid ray parameter interval.
        rec : hit_record
            Output hit record field.
        x : int
            First index into the output record field.
        y : int
            Second index into the output record field.

        Returns
        -------
        bool
            True if any sphere is hit, otherwise False.
        """
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
