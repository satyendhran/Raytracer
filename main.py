import math
import random
from time import perf_counter as pf

import taichi as ti
import taichi.math as tm
from taichi import loop_config

from AABB import AABB
from BVH import BVH
from Color import color, float_to_rgb8
from interval import Interval
from materials import Material
from Objects import Hittable_list, Sphere, hit_record
from Rand import (
    random_double,
    random_in_unit_disk,
)
from Ray import Ray

ti.init(arch=ti.gpu, default_fp=ti.f32)

# Aliases
vec3 = tm.vec3
point3 = tm.vec3


def create_sphere(c1, c2, r, mat_id):
    """
    Create a sphere with optional animation between two positions.

    Constructs a sphere that can move from c1 to c2, along with its
    bounding box that encompasses both positions.

    Parameters
    ----------
    c1 : vec3
        Initial center position.
    c2 : vec3
        Final center position (for animated spheres).
    r : float
        Radius of the sphere.
    mat_id : int
        Material ID index.

    Returns
    -------
    Sphere
        Sphere object with AABB.
    """
    v = c2 - c1
    rvec = vec3(r, r, r)

    # Create AABB in Python
    min1, max1 = c1 - rvec, c1 + rvec
    aabb_min = vec3(min(min1[0], max1[0]), min(min1[1], max1[1]), min(min1[2], max1[2]))
    aabb_max = vec3(max(min1[0], max1[0]), max(min1[1], max1[1]), max(min1[2], max1[2]))

    if c1[0] != c2[0] or c1[1] != c2[1] or c1[2] != c2[2]:
        min2, max2 = c2 - rvec, c2 + rvec
        aabb_min = vec3(
            min(aabb_min[0], min2[0], max2[0]),
            min(aabb_min[1], min2[1], max2[1]),
            min(aabb_min[2], min2[2], max2[2]),
        )
        aabb_max = vec3(
            max(aabb_max[0], min2[0], max2[0]),
            max(aabb_max[1], min2[1], max2[1]),
            max(aabb_max[2], min2[2], max2[2]),
        )

    # Create AABB using Interval
    from interval import Interval

    bbox = AABB(
        Interval(aabb_min[0], aabb_max[0]),
        Interval(aabb_min[1], aabb_max[1]),
        Interval(aabb_min[2], aabb_max[2]),
    )

    return Sphere(Ray(c1, v, 0), r, mat_id, bbox)


@ti.kernel
def length(vec: vec3) -> ti.f32:
    """
    Compute the length of a vector.

    Parameters
    ----------
    vec : vec3
        Input vector.

    Returns
    -------
    ti.f32
        Length of the vector.
    """
    return ti.math.length(vec)


@ti.kernel
def cross(v1: vec3, v2: vec3) -> vec3:
    """
    Compute the cross product of two vectors.

    Parameters
    ----------
    v1 : vec3
        First vector.
    v2 : vec3
        Second vector.

    Returns
    -------
    vec3
        Cross product v1 × v2.
    """
    return ti.math.cross(v1, v2)


@ti.kernel
def normalize(v: vec3) -> vec3:
    """
    Normalize a vector to unit length.

    Parameters
    ----------
    v : vec3
        Input vector.

    Returns
    -------
    vec3
        Normalized vector.
    """
    return ti.math.normalize(v)


@ti.data_oriented
class Camera:
    """
    Perspective camera for ray tracing with depth of field support.

    This class encapsulates image resolution, camera geometry, ray
    generation, and rendering logic. The camera owns the output image
    buffer and a hit record buffer and renders a given hittable world.

    Parameters
    ----------
    world : Hittable_list
        Scene representation containing all hittable objects.
    bvh : BVH
        BVH acceleration structure for the scene.
    image_width : int
        Width of the output image in pixels.
    aspect_ratio : float
        Aspect ratio of the output image (width / height).
    vfov : float
        Vertical field of view in degrees.
    lookfrom : vec3
        Camera position in world space.
    lookat : vec3
        Point the camera is looking at.
    vup : vec3
        Up vector for camera orientation.
    defocus_angle : float
        Angle of defocus cone for depth of field (0 for pinhole).
    focus_distance : float
        Distance to the focus plane.
    """

    def __init__(
        self,
        world,
        bvh,
        image_width,
        aspect_ratio,
        vfov,
        lookfrom,
        lookat,
        vup,
        defocus_angle,
        focus_distance,
    ):
        """
        Initialize camera geometry, image buffers, and scene reference.

        Parameters
        ----------
        world : Hittable_list
            Scene containing hittable objects.
        bvh : BVH
            BVH acceleration structure for the scene.
        image_width : int
            Width of the rendered image in pixels.
        aspect_ratio : float
            Aspect ratio of the rendered image.
        vfov : float
            Vertical field of view in degrees.
        lookfrom : vec3
            Camera position.
        lookat : vec3
            Point the camera looks at.
        vup : vec3
            Up vector.
        defocus_angle : float
            Defocus angle for depth of field.
        focus_distance : float
            Focus distance.
        """
        # ----------------------------
        # Image parameters
        # ----------------------------
        self.samples_per_pixel = 512
        self.pixel_sample_scale = 1 / self.samples_per_pixel
        self.image_width = image_width
        self.image_height = max(1, int(image_width / aspect_ratio))
        self.img = ti.Vector.field(
            3, ti.u8, shape=(self.image_width, self.image_height)
        )
        self.max_bounce_per_ray = 50

        # ----------------------------
        # Scene
        # ----------------------------
        self.world = world
        self.bvh = bvh
        self.hit_rec = hit_record.field(shape=(self.image_width, self.image_height))
        self.temp_rec = hit_record.field(shape=())

        # ----------------------------
        # Camera properties
        # ----------------------------
        self.vfov = vfov
        theta = math.radians(self.vfov)
        h = math.tan(theta / 2)
        self.focal_distance = focus_distance
        self.viewport_height = 2.0 * h * self.focal_distance
        self.viewport_width = self.viewport_height * aspect_ratio

        self.w = normalize(lookfrom - lookat)
        self.u = normalize(cross(vup, self.w))
        self.v = cross(self.w, self.u)

        self.camera_center = lookfrom

        self.viewport_u = self.viewport_width * self.u
        self.viewport_v = -self.viewport_height * self.v

        self.pixel_delta_u = self.viewport_u / self.image_width
        self.pixel_delta_v = self.viewport_v / self.image_height

        self.viewport_upper_left = (
            self.camera_center
            - self.focal_distance * self.w
            - 0.5 * self.viewport_u
            - 0.5 * self.viewport_v
        )

        self.pixel00_loc = self.viewport_upper_left + 0.5 * (
            self.pixel_delta_u + self.pixel_delta_v
        )
        self.atten = self.atten = ti.Vector.field(
            3, dtype=ti.f32, shape=(self.image_width, self.image_height)
        )

        self.scattered = Ray.field(shape=(self.image_width, self.image_height))
        self.defocus_angle = defocus_angle
        defocus_radius = focus_distance * math.tan(math.radians(defocus_angle / 2))
        self.defocus_disk_u = self.u * defocus_radius
        self.defocus_disk_v = self.v * defocus_radius

    @ti.func
    def ray_color(self, r: Ray, i: int, j: int) -> vec3:
        """
        Compute the color returned by a ray using path tracing.

        The ray is traced through the scene, bouncing off surfaces
        according to material properties, accumulating color contributions
        until it exits to the sky or reaches maximum depth.

        Parameters
        ----------
        r : Ray
            Ray cast from the camera through the pixel.
        i : int
            Pixel x-index used for hit record storage.
        j : int
            Pixel y-index used for hit record storage.

        Returns
        -------
        ti.math.vec3
            RGB color in normalized [0, 1] range.
        """
        self.atten[i, j] = vec3(1, 1, 1)
        ray = r
        attenuation = tm.vec3(1.0, 1.0, 1.0)
        exitted = 0
        hit_color = vec3(0, 0, 0)
        loop_config(serialize=True)
        for _ in range(self.max_bounce_per_ray):
            # Use BVH for intersection testing
            if self.bvh.hit(
                ray, Interval(1e-3, tm.inf), self.hit_rec, self.temp_rec, i, j
            ):
                if materials[self.hit_rec[i, j].mat_id].scatter(
                    ray, self.hit_rec[i, j], self.atten, self.scattered, i, j
                ):
                    attenuation *= self.atten[i, j]
                    ray = self.scattered[i, j]
                else:
                    break
            else:
                # Sky gradient background
                unit_dir = tm.normalize(ray.direction)
                a = 0.5 * (unit_dir.y + 1.0)
                sky = (1.0 - a) * tm.vec3(1.0) + a * tm.vec3(0.5, 0.7, 1.0)
                exitted = 1
                hit_color = attenuation * sky
                break

        return ti.select(exitted, hit_color, color(0, 0, 0))

    @ti.func
    def get_sq(self):
        """
        Generate a random offset within a pixel for antialiasing.

        Returns
        -------
        vec3
            Random offset in [-0.5, 0.5] for x and y, z=0.
        """
        return vec3(random_double(0, 1) - 0.5, random_double(0, 1) - 0.5, 0)

    @ti.func
    def defocus_disk_sample(self):
        """
        Sample a point on the defocus disk for depth of field.

        Returns
        -------
        vec3
            Random point on the lens/defocus disk.
        """
        p = random_in_unit_disk()
        return (
            self.camera_center + p.x * self.defocus_disk_u + p.y * self.defocus_disk_v
        )

    @ti.func
    def get_ray(self, i, j):
        """
        Generate a ray from the camera through pixel (i, j).

        Applies random offset for antialiasing and depth of field.

        Parameters
        ----------
        i : int
            Pixel x-coordinate.
        j : int
            Pixel y-coordinate.

        Returns
        -------
        Ray
            Ray from camera origin through randomized pixel sample.
        """
        offset = self.get_sq()
        pixel_sample = (
            self.pixel00_loc
            + ((i + offset.x) * self.pixel_delta_u)
            + ((j + offset.y) * self.pixel_delta_v)
        )
        ray_origin = ti.select(
            self.defocus_angle <= 0, self.camera_center, self.defocus_disk_sample()
        )
        ray_direction = pixel_sample - ray_origin
        ray_time = random_double(0, 1)
        return Ray(ray_origin, ray_direction, ray_time)

    @ti.kernel
    def render(self):
        """
        Render the scene with multi-sample antialiasing.

        For each pixel, multiple rays are generated with random offsets
        to accumulate color samples. The averaged result is gamma-corrected
        and written to the image buffer.
        """
        for i, j in self.img:
            pixel_color = color(0, 0, 0)
            for sample in range(self.samples_per_pixel):
                r = self.get_ray(i, j)
                pixel_color += self.ray_color(r, i, j)
            self.img[i, self.image_height - 1 - j] = float_to_rgb8(
                pixel_color * self.pixel_sample_scale
            )


# ----------------------------
# Scene objects
# ----------------------------
print("Scene Setup Started")
materials = Material.field(shape=1 + 22 * 22 + 3)

# Ground material (large Lambertian sphere)
materials[0] = Material(0, vec3(0.5, 0.5, 0.5), 0.0, 1.0)

world = Hittable_list(1 + 22 * 22 + 3)

# Ground sphere
world.add_sphere(create_sphere(point3(0, -1000, 0), point3(0, -1000, 0), 1000.0, 0))

mat_id = 1

# Generate random spheres in grid pattern
for a in range(-11, 11):
    for b in range(-11, 11):
        choose_mat = random.uniform(0.0, 1.0)

        center = point3(
            a + 0.9 * random.uniform(0.0, 1.0),
            0.2,
            b + 0.9 * random.uniform(0.0, 1.0),
        )

        dx = center.x - 4.0
        dy = center.y - 0.2
        dz = center.z - 0.0

        # Skip spheres too close to the large spheres
        if math.sqrt(dx * dx + dy * dy + dz * dz) > 0.9:
            if choose_mat < 0.8:
                # Diffuse material
                albedo = vec3(
                    random.uniform(0.0, 1.0) * random.uniform(0.0, 1.0),
                    random.uniform(0.0, 1.0) * random.uniform(0.0, 1.0),
                    random.uniform(0.0, 1.0) * random.uniform(0.0, 1.0),
                )
                centre2 = center
                materials[mat_id] = Material(0, albedo, 0.0, 1.0)

            elif choose_mat < 0.95:
                # Metal material
                albedo = vec3(
                    random.uniform(0.5, 1.0),
                    random.uniform(0.5, 1.0),
                    random.uniform(0.5, 1.0),
                )
                fuzz = random.uniform(0.0, 0.5)
                centre2 = center
                materials[mat_id] = Material(1, albedo, fuzz, 1.0)

            else:
                # Glass material
                materials[mat_id] = Material(2, vec3(1, 1, 1), 0.0, 1.5)
                centre2 = center

            world.add_sphere(create_sphere(center, centre2, 0.2, mat_id))
            mat_id += 1

# Large glass sphere
materials[mat_id] = Material(2, vec3(1, 1, 1), 0.0, 1.5)
world.add_sphere(create_sphere(point3(0, 1, 0), point3(0, 1, 0), 1.0, mat_id))
mat_id += 1

# Large diffuse sphere
materials[mat_id] = Material(0, vec3(0.4, 0.2, 0.1), 0.0, 1.0)
world.add_sphere(create_sphere(point3(-4, 1, 0), point3(-4, 1, 0), 1.0, mat_id))
mat_id += 1

# Large metal sphere
materials[mat_id] = Material(1, vec3(0.7, 0.6, 0.5), 0.0, 1.0)
world.add_sphere(create_sphere(point3(4, 1, 0), point3(4, 1, 0), 1.0, mat_id))

print("Scene Initialised")

# ----------------------------
# Build BVH
# ----------------------------
print("Building BVH...")
bvh = BVH(world.get_spheres_field(), world.get_sphere_count())
print(f"BVH built with {bvh.node_count} nodes, {bvh.prim_count} primitives")

# ----------------------------
# Camera + render
# ----------------------------
lookfrom = point3(13, 2, 3)
lookat = point3(0, 0, 0)
vup = vec3(0, 1, 0)
print("Taichi Started")
c = pf()
camera = Camera(
    world,
    bvh,
    image_width=3840,
    aspect_ratio=16.0 / 9.0,
    vfov=20,
    lookfrom=lookfrom,
    lookat=lookat,
    vup=vup,
    defocus_angle=0.6,
    focus_distance=10.0,
)
ti.sync()
d = pf()
print("Initialisation Complete")
a = pf()
camera.render()
ti.sync()
b = pf()
print("Taichi Completed")
ti.tools.imwrite(camera.img.to_numpy(), "gradient2.png")
print(f"Image Saved and render took {b - a}s and Initialisation took {d - c}s")
