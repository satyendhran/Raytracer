import taichi as ti
import taichi.math as tm


@ti.func
def random_double(a: ti.f32, b: ti.f32) -> ti.f32:
    return a + (b - a) * ti.random()


@ti.func
def random_vec3(a: ti.f32, b: ti.f32) -> tm.vec3:
    return tm.vec3(
        random_double(a, b),
        random_double(a, b),
        random_double(a, b),
    )


@ti.func
def random_unit_vec3() -> tm.vec3:
    """Uniformly distributed unit vector on the full sphere."""
    z = 2.0 * ti.random() - 1.0
    r = ti.sqrt(ti.max(1.0 - z * z, 0.0))
    phi = 2.0 * ti.math.pi * ti.random()
    return tm.vec3(r * ti.cos(phi), r * ti.sin(phi), z)


@ti.func
def random_cosine_direction(normal: tm.vec3) -> tm.vec3:
    """Cosine-weighted hemisphere direction around a unit surface normal."""
    r1 = ti.random()
    r2 = ti.random()
    phi = 2.0 * ti.math.pi * r1
    r = ti.sqrt(r2)
    local = tm.vec3(r * ti.cos(phi), r * ti.sin(phi), ti.sqrt(ti.max(1.0 - r2, 0.0)))

    sign = ti.select(normal.z >= 0.0, 1.0, -1.0)
    a = -1.0 / (sign + normal.z)
    b = normal.x * normal.y * a
    tangent = tm.vec3(1.0 + sign * normal.x * normal.x * a, sign * b, -sign * normal.x)
    bitangent = tm.vec3(b, sign + normal.y * normal.y * a, -normal.y)
    return tm.normalize(local.x * tangent + local.y * bitangent + local.z * normal)


@ti.func
def random_on_hemisphere(normal: tm.vec3) -> tm.vec3:
    on_unit_sphere = random_unit_vec3()
    return ti.select(
        tm.dot(on_unit_sphere, normal) > 0.0,
        on_unit_sphere,
        -on_unit_sphere,
    )


@ti.func
def reflect(v: tm.vec3, normal: tm.vec3) -> tm.vec3:
    return v - 2.0 * tm.dot(v, normal) * normal


@ti.func
def refract(
    uv: tm.vec3,
    n: tm.vec3,
    etai_over_etat: ti.f32,
) -> tm.vec3:
    cos_theta = ti.min(tm.dot(-uv, n), 1.0)
    r_out_perp = etai_over_etat * (uv + cos_theta * n)
    r_out_parallel = -ti.sqrt(ti.max(1.0 - tm.dot(r_out_perp, r_out_perp), 0.0)) * n
    return r_out_perp + r_out_parallel


@ti.func
def random_in_unit_disk() -> tm.vec3:
    """Concentric disk sampling (no rejection loop)."""
    theta = 2.0 * ti.math.pi * ti.random()
    r = ti.sqrt(ti.random())
    return tm.vec3(r * ti.cos(theta), r * ti.sin(theta), 0.0)
