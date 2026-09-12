import numpy as np
import taichi as ti
import taichi.math as tm


_POLYNOMIALS = (
    1, 3, 7, 11, 13, 19, 25, 37, 41, 47, 55, 59, 61, 67, 91, 97,
    103, 109, 115, 131, 137, 143, 145, 157, 167, 171, 185, 191, 193, 203, 211, 213,
)
_INITIAL_DIRECTIONS = (
    (1,), (1,), (1, 3), (1, 3, 1), (1, 1, 1), (1, 1, 3, 3),
    (1, 3, 5, 13), (1, 1, 5, 5, 17), (1, 1, 5, 5, 5),
    (1, 1, 7, 11, 19), (1, 1, 5, 1, 1), (1, 1, 1, 3, 11),
    (1, 3, 5, 5, 31), (1, 3, 3, 9, 7, 49), (1, 1, 1, 15, 21, 21),
    (1, 3, 1, 13, 27, 49), (1, 1, 1, 15, 7, 5), (1, 3, 1, 15, 13, 25),
    (1, 1, 5, 5, 19, 61), (1, 3, 7, 11, 23, 15, 103),
    (1, 3, 7, 13, 13, 15, 69), (1, 1, 3, 13, 7, 35, 63),
    (1, 3, 5, 9, 1, 25, 53), (1, 3, 1, 13, 9, 35, 107),
    (1, 3, 1, 5, 27, 61, 31), (1, 1, 5, 11, 19, 41, 61),
    (1, 3, 5, 3, 3, 13, 69), (1, 1, 7, 13, 1, 19, 1),
    (1, 3, 7, 5, 13, 19, 59), (1, 1, 3, 9, 25, 29, 41),
    (1, 3, 5, 13, 23, 1, 55), (1, 3, 7, 3, 13, 59, 17),
)


def sobol_directions():
    directions = np.zeros((32, 32), dtype=np.uint32)
    directions[0] = np.array([1 << (31 - bit) for bit in range(32)], dtype=np.uint32)
    for dimension in range(1, 32):
        polynomial = _POLYNOMIALS[dimension]
        degree = polynomial.bit_length() - 1
        for bit, value in enumerate(_INITIAL_DIRECTIONS[dimension]):
            directions[dimension, bit] = value << (31 - bit)
        for bit in range(degree, 32):
            value = int(directions[dimension, bit - degree])
            value ^= value >> degree
            for offset in range(1, degree):
                if (polynomial >> (degree - offset)) & 1:
                    value ^= int(directions[dimension, bit - offset])
            directions[dimension, bit] = value
    return directions


@ti.func
def hash_u32(value: ti.u32) -> ti.u32:
    value ^= value >> 16
    value *= ti.u32(0x7FEB352D)
    value ^= value >> 15
    value *= ti.u32(0x846CA68B)
    value ^= value >> 16
    return value


@ti.func
def reverse_bits(value: ti.u32) -> ti.u32:
    value = ((value & ti.u32(0x55555555)) << 1) | ((value >> 1) & ti.u32(0x55555555))
    value = ((value & ti.u32(0x33333333)) << 2) | ((value >> 2) & ti.u32(0x33333333))
    value = ((value & ti.u32(0x0F0F0F0F)) << 4) | ((value >> 4) & ti.u32(0x0F0F0F0F))
    value = ((value & ti.u32(0x00FF00FF)) << 8) | ((value >> 8) & ti.u32(0x00FF00FF))
    return (value << 16) | (value >> 16)


@ti.func
def scramble_bits(value: ti.u32, seed: ti.u32) -> ti.u32:
    value = reverse_bits(value)
    value ^= value * ti.u32(0x3D20ADEA)
    value += seed
    value *= (seed >> 16) | ti.u32(1)
    value ^= value * ti.u32(0x05526C56)
    value ^= value * ti.u32(0x53A22864)
    return reverse_bits(value)


@ti.func
def cosine_direction(normal: tm.vec3, u: ti.f32, v: ti.f32) -> tm.vec3:
    phi = 2.0 * ti.math.pi * u
    radius = ti.sqrt(v)
    local = tm.vec3(radius * ti.cos(phi), radius * ti.sin(phi), ti.sqrt(1.0 - v))
    sign = ti.select(normal.z >= 0.0, 1.0, -1.0)
    a = -1.0 / (sign + normal.z)
    b = normal.x * normal.y * a
    tangent = tm.vec3(1.0 + sign * normal.x * normal.x * a, sign * b, -sign * normal.x)
    bitangent = tm.vec3(b, sign + normal.y * normal.y * a, -normal.y)
    return tm.normalize(local.x * tangent + local.y * bitangent + local.z * normal)


@ti.data_oriented
class PathSampler:
    def __init__(self):
        self.directions = ti.field(ti.u32, shape=(32, 32))
        self.directions.from_numpy(sobol_directions())
        self.seed = ti.field(ti.u32, shape=())
        self.seed[None] = 0
        self.mode = ti.field(ti.i32, shape=())
        self.mode[None] = 1

    @ti.func
    def sample(self, pixel: ti.i32, index: ti.i32, dimension: ti.i32, stream: ti.i32) -> ti.f32:
        seed = hash_u32(ti.cast(pixel, ti.u32) ^ self.seed[None])
        seed = hash_u32(seed ^ hash_u32(ti.cast(dimension, ti.u32) + ti.u32(0x9E3779B9)))
        seed = hash_u32(seed ^ hash_u32(ti.cast(stream, ti.u32)))
        value = hash_u32(ti.cast(index, ti.u32) ^ seed)
        if self.mode[None] != 0:
            gray = ti.cast(index, ti.u32) ^ (ti.cast(index, ti.u32) >> 1)
            value = ti.u32(0)
            bit = 0
            while gray != 0:
                if (gray & ti.u32(1)) != 0:
                    value ^= self.directions[dimension % 32, bit]
                gray >>= 1
                bit += 1
            value = scramble_bits(value, seed)
        return (ti.cast(value >> 9, ti.f32) + 0.5) * (1.0 / 8388608.0)
