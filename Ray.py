import taichi as ti


@ti.dataclass
class Ray:
    """
    Represents a 3D ray defined by an origin point and a direction vector.

    A ray is parameterized as:
        P(t) = origin + t * direction

    where `t >= 0` moves forward along the ray.
    """

    origin: ti.math.vec3
    direction: ti.math.vec3
    tm: ti.f64

    @ti.func
    def at(self, t: ti.f64) -> ti.math.vec3:
        """
        Compute a point along the ray at parameter `t`.

        Args:
            t: Ray parameter. Larger values move farther
               along the ray direction.

        Returns:
            3D point at position P(t).
        """
        return self.origin + t * self.direction
