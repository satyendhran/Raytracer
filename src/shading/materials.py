import taichi as ti
from src.core.Rand import random_cosine_direction, random_double, random_unit_vec3, reflect, refract
from src.core.Ray import Ray
from src.shading.Texture import Texture

@ti.func
def near_zero_vec3(v: ti.math.vec3):
    s = 1e-08
    return (ti.abs(v.x) < s) & (ti.abs(v.y) < s) & (ti.abs(v.z) < s)

@ti.dataclass
class Material:
    material_type: ti.u8
    texture: Texture
    fuzz: ti.f32
    refractive_index: ti.f32
    emitted: ti.math.vec3

    @ti.func
    def scatter(self, r_in, rec, attenuation, scattered, i, j):
        inside = 0
        ret = 1
        is_light = self.material_type == 3
        spawn_p = rec.p + rec.normal * 0.0001
        if self.material_type == 0:
            scatter_direction = random_cosine_direction(rec.normal)
            if near_zero_vec3(scatter_direction):
                scatter_direction = rec.normal
            scattered[i, j] = Ray(spawn_p, scatter_direction, r_in.tm, 1)
            attenuation[i, j] = self.texture.value(rec.u, rec.v, rec.p, rec.normal)
            ret = 1
            inside = 1
        if self.material_type == 1:
            reflected = reflect(r_in.direction, rec.normal) + self.fuzz * random_unit_vec3()
            scattered[i, j] = Ray(spawn_p, reflected, r_in.tm, 1)
            attenuation[i, j] = self.texture.value(rec.u, rec.v, rec.p, rec.normal)
            ret = ti.math.dot(scattered[i, j].direction, rec.normal) > 0
            inside = 1
        if self.material_type == 2:
            attenuation[i, j] = ti.math.vec3(1, 1, 1)
            ri = ti.select(rec.front_face, 1 / self.refractive_index, self.refractive_index)
            unit_direction = ti.math.normalize(r_in.direction)
            costheta = ti.min(ti.math.dot(-unit_direction, rec.normal), 1.0)
            sintheta = ti.sqrt(1 - costheta * costheta)
            r0 = (1 - ri) / (1 + ri)
            r0 = r0 * r0
            reflectance = r0 + (1 - r0) * ti.math.pow(1 - costheta, 5)
            cannot_refract = sintheta * ri > 1
            direction = ti.math.vec3(0, 0, 0)
            if cannot_refract or reflectance > random_double(0, 1):
                direction = reflect(unit_direction, rec.normal)
            else:
                direction = refract(unit_direction, rec.normal, ri)
            dielectric_p = rec.p + ti.select(ti.math.dot(direction, rec.normal) > 0.0, rec.normal, -rec.normal) * 0.0001
            scattered[i, j] = Ray(dielectric_p, direction, r_in.tm, 1)
            inside = 1
            ret = 1
        return ti.select(is_light, ti.cast(0, ti.i32), ti.select(inside, ret, 0))

def create_solid_texture(color):
    return Texture(texture_type=0, color1=color, color2=ti.math.vec3(0, 0, 0), scale=1.0)

def create_checker_texture(color1, color2, scale):
    return Texture(texture_type=1, color1=color1, color2=color2, scale=scale)

def create_lambertian(texture, emitted=None):
    if emitted is None:
        emitted = ti.math.vec3(0.0, 0.0, 0.0)
    return Material(material_type=0, texture=texture, fuzz=0.0, refractive_index=1.0, emitted=emitted)

def create_metal(albedo, fuzz):
    return Material(material_type=1, texture=create_solid_texture(albedo), fuzz=min(fuzz, 1.0), refractive_index=1.0, emitted=ti.math.vec3(0.0, 0.0, 0.0))

def create_dielectric(refractive_index):
    return Material(material_type=2, texture=create_solid_texture(ti.math.vec3(1, 1, 1)), fuzz=0.0, refractive_index=refractive_index, emitted=ti.math.vec3(0.0, 0.0, 0.0))

def create_diffuse_light(emitted_color):
    return Material(material_type=3, texture=create_solid_texture(ti.math.vec3(0.0, 0.0, 0.0)), fuzz=0.0, refractive_index=1.0, emitted=emitted_color)

def create_apple_peel_texture(scale: float=3.5):
    return Texture(texture_type=3, color1=ti.math.vec3(0.78, 0.06, 0.08), color2=ti.math.vec3(0.42, 0.62, 0.14), scale=scale)

def create_uv_checker_texture(color1, color2, scale):
    return Texture(texture_type=2, color1=color1, color2=color2, scale=scale)

def create_gradient_stripe_texture(color1, color2, scale):
    return Texture(texture_type=4, color1=color1, color2=color2, scale=scale)

def create_marble_texture(color1, color2, scale):
    return Texture(texture_type=5, color1=color1, color2=color2, scale=scale)