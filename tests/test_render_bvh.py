import os
import sys
import unittest

import numpy as np
import taichi as ti
import taichi.math as tm


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import main
from src.accel.scene_accel import PrimBVH, ScenePrimitive, scene_primitive_sphere
from src.core.interval import Interval
from src.core.Ray import Ray
from src.geometry.Objects import hit_record


@ti.kernel
def _median_sphere_rays(bvh: ti.template(), records: ti.template(), output: ti.template()):
    for slot in range(2):
        ray = Ray(
            tm.vec3(2.0 * ti.cast(slot, ti.f32), 0.0, 0.0),
            tm.vec3(0.0, 0.0, -1.0), 0.5, main.RAY_PRIMARY,
        )
        records[slot, 0].prim_id = -1
        records[slot, 0].t = -1.0
        output[slot] = bvh.hit(ray, Interval(0.001, 10.0), records, slot, 0)


class TestMedianBVH(unittest.TestCase):
    def test_single_analytic_sphere_center_hit_and_miss(self):
        primitives = ScenePrimitive.field(shape=1)
        center = tm.vec3(0.0, 0.0, -2.0)
        primitives[0] = scene_primitive_sphere(center, center, 0.5, 7)
        bvh = PrimBVH(primitives, prim_count=1, split="median")
        records = hit_record.field(shape=(2, 1))
        output = ti.field(ti.i32, shape=2)
        _median_sphere_rays(bvh, records, output)
        self.assertEqual(bvh.prim_count, 1)
        np.testing.assert_array_equal(output.to_numpy(), np.array([1, 0], np.int32))
        self.assertAlmostEqual(float(records[0, 0].t), 1.5, places=6)
        np.testing.assert_allclose(records[0, 0].p, [0.0, 0.0, -1.5], rtol=0, atol=1e-6)
        np.testing.assert_allclose(records[0, 0].normal, [0.0, 0.0, 1.0], rtol=0, atol=1e-6)
        self.assertEqual(int(records[0, 0].front_face), 1)
        self.assertEqual(int(records[0, 0].mat_id), 7)
        self.assertEqual(int(records[0, 0].prim_id), 0)


if __name__ == "__main__":
    unittest.main()
