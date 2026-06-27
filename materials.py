import taichi as ti
from Rand import random_cosine_direction, random_double, random_unit_vec3, reflect, refract
from Ray import Ray
from Texture import Texture

@ti.func
def near_zero_vec3(v: ti.math.vec3):
    """
    Check if a vector is near zero in all components.

    Args:
        v: 3D vector to test.

    Returns:
        True if all components have absolute value less than 1e-8, False otherwise.
    """
    s = 1e-8  
    return (ti.abs(v.x) < s) & (ti.abs(v.y) < s) & (ti.abs(v.z) < s)


@ti.dataclass
class Material:
    """
    Represents a material in a ray tracing scene.

    Supports four material types:
    - Lambertian (diffuse): Scatters rays in random directions
    - Metal: Reflects rays with optional fuzziness
    - Dielectric: Refracts/reflects based on Fresnel equations
    - Diffuse light: Emits radiance; does not scatter secondary rays

    Attributes:
        material_type: Material type identifier (0=Lambertian, 1=Metal, 2=Dielectric, 3=Diffuse light)
        texture: Texture defining the surface color/pattern
        fuzz: Fuzziness factor for metal reflections [0, 1]
        refractive_index: Index of refraction for dielectric materials
        emitted: Emitted radiance (used when material_type == 3, or additive on every hit)
    """

    material_type: ti.u8  
    texture: Texture
    fuzz: ti.f32
    refractive_index: ti.f32
    emitted: ti.math.vec3

    @ti.func
    def scatter(self, r_in, rec, attenuation, scattered, i, j):
        """
        Compute how a ray scatters when hitting this material.

        Args:
            r_in: Incoming ray
            rec: Hit record containing intersection information (must have p, normal)
            attenuation: Output array for color attenuation at (i, j)
            scattered: Output array for scattered ray at (i, j)
            i: First index for output arrays
            j: Second index for output arrays

        Returns:
            1 if ray was scattered successfully, 0 otherwise.
        """
        inside = 0
        ret = 1
        is_light = self.material_type == 3
        spawn_p = rec.p + rec.normal * 1e-4

        if self.material_type == 0:
            scatter_direction = random_cosine_direction(rec.normal)
            if near_zero_vec3(scatter_direction):
                scatter_direction = rec.normal
            scattered[i, j] = Ray(spawn_p, scatter_direction, r_in.tm, 1)

            attenuation[i, j] = self.texture.value(rec.u, rec.v, rec.p, rec.normal)
            ret = 1
            inside = 1

        
        if self.material_type == 1:
            reflected = (
                    reflect(r_in.direction, rec.normal) + self.fuzz * random_unit_vec3()
            )
            scattered[i, j] = Ray(spawn_p, reflected, r_in.tm, 1)
            attenuation[i, j] = self.texture.value(rec.u, rec.v, rec.p, rec.normal)
            ret = ti.math.dot(scattered[i, j].direction, rec.normal) > 0
            inside = 1



        
        if self.material_type == 2:
            attenuation[i, j] = ti.math.vec3(1, 1, 1)
            ri = ti.select(
                rec.front_face, 1 / self.refractive_index, self.refractive_index
            )
            unit_direction = ti.math.normalize(r_in.direction)
            costheta = ti.min(ti.math.dot(-unit_direction, rec.normal), 1.0)
            sintheta = ti.sqrt(1 - (costheta * costheta))

            
            r0 = (1 - ri) / (1 + ri)
            r0 = r0 * r0
            reflectance = r0 + (1 - r0) * ti.math.pow(1 - costheta, 5)

            cannot_refract = (sintheta * ri) > 1
            direction = ti.math.vec3(0, 0, 0)

            if cannot_refract or reflectance > random_double(0, 1):
                direction = reflect(unit_direction, rec.normal)
            else:
                direction = refract(unit_direction, rec.normal, ri)

            dielectric_p = rec.p + ti.select(
                ti.math.dot(direction, rec.normal) > 0.0,
                rec.normal,
                -rec.normal,
            ) * 1e-4
            scattered[i, j] = Ray(dielectric_p, direction, r_in.tm, 1)
            inside = 1
            ret = 1

        return ti.select(is_light, ti.cast(0, ti.i32), ti.select(inside, ret, 0))




def create_solid_texture(color):
    """Create a solid color texture (Python-scope)."""
    return Texture(
        texture_type=0,
        color1=color,
        color2=ti.math.vec3(0, 0, 0),
        scale=1.0
    )


def create_checker_texture(color1, color2, scale):
    """Create a checker pattern texture (Python-scope)."""
    return Texture(
        texture_type=1,
        color1=color1,
        color2=color2,
        scale=scale
    )


def create_lambertian(texture, emitted=None):
    """Create a Lambertian (diffuse) material with the given texture (Python-scope)."""
    if emitted is None:
        emitted = ti.math.vec3(0.0, 0.0, 0.0)
    return Material(
        material_type=0,
        texture=texture,
        fuzz=0.0,
        refractive_index=1.0,
        emitted=emitted,
    )


def create_metal(albedo, fuzz):
    """Create a metal material (Python-scope)."""
    return Material(
        material_type=1,
        texture=create_solid_texture(albedo),
        fuzz=min(fuzz, 1.0),
        refractive_index=1.0,
        emitted=ti.math.vec3(0.0, 0.0, 0.0),
    )


def create_dielectric(refractive_index):
    """Create a dielectric (glass-like) material (Python-scope)."""
    return Material(
        material_type=2,
        texture=create_solid_texture(ti.math.vec3(1, 1, 1)),
        fuzz=0.0,
        refractive_index=refractive_index,
        emitted=ti.math.vec3(0.0, 0.0, 0.0),
    )


def create_diffuse_light(emitted_color):
    """Area light / emissive surface (Python-scope)."""
    return Material(
        material_type=3,
        texture=create_solid_texture(ti.math.vec3(0.0, 0.0, 0.0)),
        fuzz=0.0,
        refractive_index=1.0,
        emitted=emitted_color,
    )


def create_apple_peel_texture(scale: float = 3.5):
    """Procedural red / yellow-green peel pattern for spheres (Python-scope)."""
    return Texture(
        texture_type=3,
        color1=ti.math.vec3(0.78, 0.06, 0.08),
        color2=ti.math.vec3(0.42, 0.62, 0.14),
        scale=scale,
    )

def create_uv_checker_texture(color1, color2, scale):
    """Create a UV-based checker pattern texture (Python-scope) - better for spheres."""
    return Texture(
        texture_type=2,
        color1=color1,
        color2=color2,
        scale=scale
    )


def create_gradient_stripe_texture(color1, color2, scale):
    """Smooth sine gradient stripes along V axis (Python-scope)."""
    return Texture(
        texture_type=4,
        color1=color1,
        color2=color2,
        scale=scale,
    )


def create_marble_texture(color1, color2, scale):
    """3D sine-turbulence marble veining (Python-scope)."""
    return Texture(
        texture_type=5,
        color1=color1,
        color2=color2,
        scale=scale,
    )
