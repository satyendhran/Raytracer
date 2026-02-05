import taichi as ti

from interval import Interval

vec3 = ti.math.vec3


@ti.dataclass
class AABB:
    """
    Axis-Aligned Bounding Box for efficient ray-object intersection testing.

    Uses the slab method to test ray intersections by treating the box
    as the intersection of three axis-aligned slabs.

    Attributes
    ----------
    x : Interval
        Extent along the x-axis.
    y : Interval
        Extent along the y-axis.
    z : Interval
        Extent along the z-axis.
    """

    x: Interval
    y: Interval
    z: Interval

    @ti.func
    def axis_interval(self, n):
        """
        Get the interval for a specific axis.

        Parameters
        ----------
        n : int
            Axis index (0=x, 1=y, 2=z).

        Returns
        -------
        Interval
            The interval for the specified axis.
        """
        return ti.select(n == 0, self.x, ti.select(n == 1, self.y, self.z))

    @ti.func
    def hit(self, r, ray_t):
        """
        Test if a ray intersects this AABB using the slab method.

        Updates ray_t to the intersection interval if hit occurs.

        Parameters
        ----------
        r : Ray
            Ray to test for intersection.
        ray_t : Interval field
            Ray parameter range, updated in-place with intersection bounds.

        Returns
        -------
        int
            1 if ray intersects AABB, 0 otherwise.
        """
        ray_origin = r.origin
        ray_dir = r.direction
        no_hit = False

        for axis in range(3):
            ax = self.axis_interval(axis)
            adinv = 1 / ray_dir[axis]
            t0 = (ax.min - ray_origin[axis]) * adinv
            t1 = (ax.max - ray_origin[axis]) * adinv

            if t0 > t1:
                t0, t1 = t1, t0

            if t0 > ray_t[None].min:
                ray_t[None].min = t0
            if t1 < ray_t[None].max:
                ray_t[None].max = t1
            if ray_t[None].min >= ray_t[None].max:
                no_hit = True

        return ti.select(no_hit, 0, 1)


@ti.func
def init_aabb(a, b):
    """
    Initialize an AABB from two corner points.

    Automatically handles cases where corners are not properly ordered
    (min/max can be swapped).

    Parameters
    ----------
    a : vec3
        First corner point.
    b : vec3
        Second corner point.

    Returns
    -------
    AABB
        Axis-aligned bounding box with properly ordered min/max per axis.
    """
    x = Interval(a[0], b[0]) if a[0] <= b[0] else Interval(b[0], a[0])
    y = Interval(a[1], b[1]) if a[1] <= b[1] else Interval(b[1], a[1])
    z = Interval(a[2], b[2]) if a[2] <= b[2] else Interval(b[2], a[2])
    return AABB(x, y, z)


@ti.func
def init_aabb_2(box1, box2):
    """
    Create an AABB that encloses two existing AABBs.

    Computes the union of two bounding boxes by taking the minimum
    of all min corners and maximum of all max corners.

    Parameters
    ----------
    box1 : AABB
        First bounding box.
    box2 : AABB
        Second bounding box.

    Returns
    -------
    AABB
        Bounding box that contains both input boxes.
    """
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
