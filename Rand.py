import taichi as ti
import taichi.math as tm





@ti.func
def random_double(a: ti.f32, b: ti.f32) -> ti.f32:
    """
    Generate a random float in [a, b).

    Parameters
    ----------
    a, b : float
        Range bounds.

    Returns
    -------
    float
        Random value in [a, b).
    """
    return a + (b - a) * ti.random()


@ti.func
def random_vec3(a: ti.f32, b: ti.f32) -> tm.vec3:
    """
    Generate a random 3D vector with each component in [a, b).

    Parameters
    ----------
    a, b : float
        Range bounds for each component.

    Returns
    -------
    tm.vec3
        Random vector.
    """
    return tm.vec3(
        random_double(a, b),
        random_double(a, b),
        random_double(a, b),
    )





@ti.func
def random_unit_vec3() -> tm.vec3:
    """
    Generate a random point inside the unit sphere.

    Uses rejection sampling to ensure uniform distribution within
    the sphere.

    Returns
    -------
    tm.vec3
        Random vector in unit sphere.
    """
    p = tm.vec3(0.0)
    found = False

    while not found:
        p = 2.0 * random_vec3(0.0, 1.0) - 1.0
        lensq = tm.dot(p, p)
        if (lensq > 1e-16) & (lensq <= 1.0):
            found = True

    return p


@ti.func
def random_on_hemisphere(normal: tm.vec3) -> tm.vec3:
    """
    Generate a random vector on the hemisphere defined by a normal.

    Parameters
    ----------
    normal : tm.vec3
        Surface normal defining the hemisphere orientation.

    Returns
    -------
    tm.vec3
        Hemisphere-aligned random direction.
    """
    on_unit_sphere = random_unit_vec3()
    return ti.select(
        tm.dot(on_unit_sphere, normal) > 0.0,
        on_unit_sphere,
        -on_unit_sphere,
    )





@ti.func
def reflect(v: tm.vec3, normal: tm.vec3) -> tm.vec3:
    """
    Reflect a vector about a normal.

    Parameters
    ----------
    v : tm.vec3
        Incoming vector to reflect.
    normal : tm.vec3
        Surface normal.

    Returns
    -------
    tm.vec3
        Reflected direction.
    """
    return v - 2.0 * tm.dot(v, normal) * normal


@ti.func
def refract(
    uv: tm.vec3,
    n: tm.vec3,
    etai_over_etat: ti.f32,
) -> tm.vec3:
    """
    Refract a vector using Snell's law.

    Parameters
    ----------
    uv : tm.vec3
        Incoming unit vector.
    n : tm.vec3
        Surface normal.
    etai_over_etat : float
        Ratio of refractive indices (eta_incident / eta_transmitted).

    Returns
    -------
    tm.vec3
        Refracted direction.
    """
    cos_theta = ti.min(tm.dot(-uv, n), 1.0)
    r_out_perp = etai_over_etat * (uv + cos_theta * n)

    
    r_out_parallel = -ti.sqrt(ti.abs(1.0 - tm.dot(r_out_perp, r_out_perp))) * n

    return r_out_perp + r_out_parallel





@ti.func
def random_in_unit_disk() -> tm.vec3:
    """
    Generate a random point inside the unit disk (z = 0).

    Uses rejection sampling for uniform distribution. Used for
    depth of field camera lens sampling.

    Returns
    -------
    tm.vec3
        Random disk sample with z = 0.
    """
    p = tm.vec3(0.0)
    found = False

    while not found:
        p = tm.vec3(
            random_double(-1.0, 1.0),
            random_double(-1.0, 1.0),
            0.0,
        )
        if tm.dot(p, p) <= 1.0:
            found = True

    return p
