import taichi as ti


@ti.dataclass
class Interval:
    """
    Represents a numeric interval [min, max] used for range checks.

    Attributes
    ----------
    min : ti.f32
        Lower bound of the interval.
    max : ti.f32
        Upper bound of the interval.
    """

    min: ti.f32
    max: ti.f32

    @ti.func
    def size(self):
        """
        Computes the size of the interval.

        Returns
        -------
        ti.f32
            Length of the interval, computed as (max - min).
        """
        return self.max - self.min

    @ti.func
    def contains(self, x: ti.f32):
        """
        Checks whether a value lies within the interval, inclusive.

        Parameters
        ----------
        x : ti.f32
            Value to be tested.

        Returns
        -------
        bool
            True if min <= x <= max, otherwise False.
        """
        return self.min <= x <= self.max

    @ti.func
    def surrounds(self, x: ti.f32):
        """
        Checks whether a value lies strictly inside the interval.

        Parameters
        ----------
        x : ti.f32
            Value to be tested.

        Returns
        -------
        bool
            True if min < x < max, otherwise False.
        """
        return self.min < x < self.max

    @ti.func
    def clamp(self, x: ti.f32):
        """
        Clamps a value to lie within the interval.

        Parameters
        ----------
        x : ti.f32
            Value to clamp.

        Returns
        -------
        ti.f32
            Clamped value in [min, max].
        """
        return ti.math.clamp(x, self.min, self.max)

    @ti.func
    def expand(self, delta: ti.f32):
        """
        Expands the interval by a given amount on both sides.

        Parameters
        ----------
        delta : ti.f32
            Total expansion amount (split equally on both sides).

        Returns
        -------
        Interval
            New interval expanded by delta/2 on each side.
        """
        padding = delta / 2
        return Interval(self.min - padding, self.max + padding)
