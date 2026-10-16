import argparse
import os
import sys
import unittest
from unittest import mock

import numpy as np
import taichi as ti
import taichi.math as tm


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import main
from src.core.Ray import Ray
from src.core.sampling import PathSampler, sobol_directions


@ti.data_oriented
class _UnoccludedBVH:
    @ti.func
    def any_hit(self, ray, interval, ignore_prim):
        return 0


@ti.data_oriented
class _SyntheticAdaptiveCamera(main.Camera):
    @ti.func
    def trace_path(self, ray: Ray, i: ti.i32, j: ti.i32, sample_index: ti.i32) -> tm.vec3:
        radiance = tm.vec3(1.0 + 0.125 * i, 2.0 + 0.25 * j, 4.0)
        if (i + j) % 2 != 0:
            sign = 2.0 * ti.cast(sample_index % 2, ti.f32) - 1.0
            radiance += sign * tm.vec3(0.5, 1.0, 2.0)
        return radiance


@ti.data_oriented
class _SyntheticFixedBatchCamera(main.Camera):
    @ti.func
    def trace_path(self, ray: Ray, i: ti.i32, j: ti.i32, sample_index: ti.i32) -> tm.vec3:
        return ti.cast(sample_index, ti.f32) * tm.vec3(0.25, 0.5, 2.0)


@ti.kernel
def _sample_requests(
    sampler: ti.template(),
    requests: ti.types.ndarray(dtype=ti.i32, ndim=2),
    output: ti.types.ndarray(dtype=ti.f32, ndim=1),
    start: ti.i32,
    stop: ti.i32,
):
    for row in range(start, stop):
        output[row] = sampler.sample(
            requests[row, 0], requests[row, 1], requests[row, 2], requests[row, 3]
        )


@ti.kernel
def _power_weights(
    camera: ti.template(),
    pdfs: ti.types.ndarray(dtype=ti.f32, ndim=2),
    output: ti.types.ndarray(dtype=ti.f32, ndim=2),
):
    for row in range(pdfs.shape[0]):
        output[row, 0] = camera.power_heuristic(pdfs[row, 0], pdfs[row, 1])
        output[row, 1] = camera.power_heuristic(pdfs[row, 1], pdfs[row, 0])


@ti.kernel
def _area_light_samples(
    camera: ti.template(),
    output: ti.types.ndarray(dtype=ti.f32, ndim=3),
    start: ti.i32,
    stop: ti.i32,
):
    for index in range(start, stop):
        p = tm.vec3(0.0, 0.0, 0.0)
        albedo = tm.vec3(0.8, 0.5, 0.25)
        above = camera.sample_area_light(
            p, tm.vec3(0.0, 0.0, 1.0), albedo,
            index % 8, (index // 8) % 8, 0.375, index, index % 4,
        )
        below = camera.sample_area_light(
            p, tm.vec3(0.0, 0.0, -1.0), albedo,
            index % 8, (index // 8) % 8, 0.375, index, index % 4,
        )
        for channel in ti.static(range(3)):
            output[index, 0, channel] = above[channel]
            output[index, 1, channel] = below[channel]


@ti.kernel
def _area_light_estimates(
    camera: ti.template(),
    output: ti.types.ndarray(dtype=ti.f32, ndim=2),
):
    for index in range(output.shape[0]):
        value = camera.sample_area_light(
            tm.vec3(0.0), tm.vec3(0.0, 0.0, 1.0), tm.vec3(0.8, 0.5, 0.25),
            0, 0, 0.375, index, 0,
        )
        for channel in ti.static(range(3)):
            output[index, channel] = value[channel]


@ti.kernel
def _endpoint_weight(
    camera: ti.template(), hit_z: ti.f32, direction_scale: ti.f32
) -> ti.f32:
    ray = Ray(
        tm.vec3(0.0, 0.0, 0.0002),
        tm.vec3(0.0, 0.0, direction_scale),
        0.375,
        main.RAY_SECONDARY,
    )
    return camera.bsdf_light_mis_weight(
        ray, tm.vec3(0.0), tm.vec3(0.0, 0.0, 1.0), tm.vec3(0.0, 0.0, hit_z)
    )


@ti.kernel
def _indexed_endpoint_weight(
    camera: ti.template(), hit_z: ti.f32, hit_prim_id: ti.i32
) -> ti.f32:
    ray = Ray(
        tm.vec3(0.0, 0.0, 0.0002), tm.vec3(0.0, 0.0, 1.0), 0.375, main.RAY_SECONDARY,
    )
    return camera.bsdf_light_mis_weight(
        ray, tm.vec3(0.0), tm.vec3(0.0, 0.0, 1.0),
        tm.vec3(0.0, 0.0, hit_z), hit_prim_id,
    )


def _quad(z, half_extent=0.5):
    return {
        "Q": (-half_extent, -half_extent, z),
        "u": (2.0 * half_extent, 0.0, 0.0),
        "v": (0.0, 2.0 * half_extent, 0.0),
        "n": (0.0, 0.0, -1.0),
        "area": (2.0 * half_extent) ** 2,
        "Le": (5.0, 4.0, 3.0),
    }


def _sobol_points(directions, count):
    points = np.zeros((count, directions.shape[0]), dtype=np.uint32)
    for index in range(count):
        gray = index ^ (index >> 1)
        for bit in range(gray.bit_length()):
            if gray & (1 << bit):
                points[index] ^= directions[:, bit]
    return points.astype(np.float64) / (1 << 32)


def _area_light_midpoint_integral(samples, resolution=256, competing_bsdf=True):
    coordinates = (np.arange(resolution, dtype=np.float64) + 0.5) * (2.0 / resolution) - 1.0
    x, y = np.meshgrid(coordinates, coordinates, indexing="ij")
    distance_squared = x * x + y * y + 4.0
    cosine = 2.0 / np.sqrt(distance_squared)
    light_pdf = distance_squared / (cosine * 4.0)
    bsdf_pdf = cosine / np.pi
    weighted_light_pdf_squared = np.square(samples * light_pdf)
    mis_weight = weighted_light_pdf_squared / (weighted_light_pdf_squared + np.square(bsdf_pdf))
    if not competing_bsdf:
        mis_weight = np.ones_like(mis_weight)
    geometry = cosine * cosine * 4.0 / distance_squared
    return np.array([0.8, 0.5, 0.25]) * np.array([5.0, 4.0, 3.0]) / np.pi * np.mean(
        geometry * mis_weight
    )


class TestSobolDirections(unittest.TestCase):
    def test_portable_direction_words_match_known_dimensions(self):
        directions = sobol_directions()
        self.assertEqual(directions.shape, (32, 32))
        self.assertEqual(directions.dtype, np.dtype(np.uint32))
        np.testing.assert_array_equal(
            directions[0], np.array([1 << bit for bit in range(31, -1, -1)], np.uint32)
        )
        expected = np.array([
            [0x80000000, 0xC0000000, 0xA0000000, 0xF0000000,
             0x88000000, 0xCC000000, 0xAA000000, 0xFF000000],
            [0x80000000, 0xC0000000, 0x60000000, 0x90000000,
             0xE8000000, 0x5C000000, 0x8E000000, 0xC5000000],
            [0x80000000, 0xC0000000, 0x20000000, 0x50000000,
             0xF8000000, 0x74000000, 0xA2000000, 0x93000000],
        ], dtype=np.uint32)
        np.testing.assert_array_equal(directions[1:4, :8], expected)

    def test_gray_code_points_match_known_first_dimensions(self):
        expected = np.array([
            [0, 0, 0, 0], [4, 4, 4, 4], [6, 2, 2, 2], [2, 6, 6, 6],
            [3, 3, 5, 7], [7, 7, 1, 3], [5, 1, 7, 5], [1, 5, 3, 1],
        ], dtype=np.float64) / 8.0
        np.testing.assert_array_equal(_sobol_points(sobol_directions()[:4], 8), expected)

    def test_all_dimensions_match_scipy_when_available(self):
        try:
            from scipy.stats import qmc
        except ImportError:
            self.skipTest("SciPy is optional; fixed Sobol references remain covered")
        expected = qmc.Sobol(d=32, scramble=False, bits=32).random_base2(8)
        np.testing.assert_array_equal(_sobol_points(sobol_directions(), 256), expected)


class TestPathSampler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sampler = PathSampler()

    def setUp(self):
        self.sampler.mode[None] = 1
        self.sampler.seed[None] = 0

    def test_power_of_two_counts_have_uniform_2d_histograms(self):
        for mode in (1, 0):
            self.sampler.mode[None] = mode
            for seed, pixel, stream in ((0, 0, 0), (0xFFFFFFFF, 63, 7)):
                self.sampler.seed[None] = seed
                for count in (64, 256, 1024, 4096):
                    with self.subTest(mode=mode, seed=seed, count=count):
                        requests = np.empty((count, 2, 4), dtype=np.int32)
                        requests[:, :, 0] = pixel
                        requests[:, :, 1] = np.arange(count)[:, None]
                        requests[:, :, 2] = (0, 1)
                        requests[:, :, 3] = stream
                        output = np.full(count * 2, np.nan, dtype=np.float32)
                        _sample_requests(self.sampler, requests.reshape(-1, 4), output, 0, count * 2)
                        points = output.reshape(count, 2)
                        self.assertTrue(np.isfinite(points).all())
                        self.assertTrue(((points > 0.0) & (points < 1.0)).all())
                        histogram, _, _ = np.histogram2d(
                            points[:, 0], points[:, 1], bins=8, range=((0, 1), (0, 1))
                        )
                        expected = count / 64
                        if mode == 1:
                            np.testing.assert_array_equal(histogram, np.full((8, 8), expected))
                        else:
                            chi_squared = np.square(histogram - expected).sum() / expected
                            self.assertLess(chi_squared, 160.0)

    def test_samples_are_bitwise_reproducible_across_batches_and_order(self):
        count = 513
        requests = np.empty((count, 4), dtype=np.int32)
        row = np.arange(count, dtype=np.int32)
        requests[:, 0] = (row * 17) % 64
        requests[:, 1] = row * 65537
        requests[-1, 1] = np.iinfo(np.int32).max
        requests[:, 2] = row % 97
        requests[:, 3] = row % 11
        ranges = ((0, 1), (1, 7), (7, 64), (64, 129), (129, 512), (512, 513))
        for mode in (1, 0):
            for seed in (0, 0xDEADBEEF, 0xFFFFFFFF):
                with self.subTest(mode=mode, seed=seed):
                    self.sampler.mode[None] = mode
                    self.sampler.seed[None] = seed
                    whole = np.full(count, np.nan, dtype=np.float32)
                    batched = np.full_like(whole, np.nan)
                    _sample_requests(self.sampler, requests, whole, 0, count)
                    for start, stop in reversed(ranges):
                        _sample_requests(self.sampler, requests, batched, start, stop)
                    self.assertTrue(np.isfinite(whole).all())
                    self.assertTrue(((whole > 0.0) & (whole < 1.0)).all())
                    np.testing.assert_array_equal(batched, whole)
                    permutation = np.random.default_rng(23).permutation(count)
                    permuted = np.ascontiguousarray(requests[permutation])
                    _sample_requests(self.sampler, permuted, batched, 0, count)
                    np.testing.assert_array_equal(batched, whole[permutation])

    def test_seed_pixel_dimension_stream_and_mode_change_the_sequence(self):
        count = 256
        baseline_requests = np.zeros((count, 4), dtype=np.int32)
        baseline_requests[:, 1] = np.arange(count)
        for mode in (1, 0):
            with self.subTest(mode=mode):
                self.sampler.mode[None] = mode
                self.sampler.seed[None] = 0
                baseline = np.empty(count, dtype=np.float32)
                changed = np.empty_like(baseline)
                _sample_requests(self.sampler, baseline_requests, baseline, 0, count)
                for column, value in ((0, 17), (2, 1), (2, 32), (3, 9)):
                    requests = baseline_requests.copy()
                    requests[:, column] = value
                    _sample_requests(self.sampler, requests, changed, 0, count)
                    self.assertGreater(np.count_nonzero(changed != baseline), count * 0.95)
                self.sampler.seed[None] = 0x80000000
                _sample_requests(self.sampler, baseline_requests, changed, 0, count)
                self.assertGreater(np.count_nonzero(changed != baseline), count * 0.95)
                self.sampler.seed[None] = 0
                _sample_requests(self.sampler, baseline_requests, changed, 0, count)
                np.testing.assert_array_equal(changed, baseline)
                self.sampler.mode[None] = 1 - mode
                _sample_requests(self.sampler, baseline_requests, changed, 0, count)
                self.assertGreater(np.count_nonzero(changed != baseline), count * 0.95)


class TestCameraSampling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.camera = main.Camera(
            _UnoccludedBVH(), image_width=8, aspect_ratio=1.0, vfov=40.0,
            lookfrom=(0.0, 0.0, 3.0), lookat=(0.0, 0.0, 0.0),
            vup=(0.0, 1.0, 0.0), defocus_angle=0.0, focus_distance=3.0,
        )

    def setUp(self):
        self.camera.configure_nee([], enabled=False)
        self.camera.light_samples[None] = 1
        self.camera.shadow_epsilon[None] = 0.0002
        self.camera.max_bounces[None] = 12

    def test_power_heuristic_is_scale_invariant_and_complementary_for_tiny_pdfs(self):
        scales = np.array([1e-28, 1e-20, 1e-9, 1.0, 1e20, 1e30], np.float64)
        ratios = np.array([[1, 1], [1, 3], [3, 1], [0, 1], [1, 0]], np.float64)
        pdfs = (scales[:, None, None] * ratios).reshape(-1, 2).astype(np.float32)
        output = np.empty_like(pdfs)
        _power_weights(self.camera, pdfs, output)
        squared = np.square(pdfs.astype(np.float64))
        expected = squared / squared.sum(axis=1, keepdims=True)
        self.assertTrue(np.isfinite(output).all())
        np.testing.assert_allclose(output, expected, rtol=2e-6, atol=2e-7)
        np.testing.assert_allclose(output.sum(axis=1), 1.0, rtol=0, atol=2e-7)
        zeros = np.zeros((1, 2), np.float32)
        zero_output = np.empty_like(zeros)
        _power_weights(self.camera, zeros, zero_output)
        np.testing.assert_array_equal(zero_output, zeros)

    def test_area_light_below_surface_is_zero_with_positive_control(self):
        self.assertIsInstance(self.camera.sampler, PathSampler)
        self.camera.configure_nee([_quad(2.0)])
        for mode in (1, 0):
            self.camera.sampler.mode[None] = mode
            self.camera.sampler.seed[None] = 0xDEADBEEF
            for samples in (1, 2, 4, 8):
                with self.subTest(mode=mode, light_samples=samples):
                    self.camera.light_samples[None] = samples
                    output = np.full((128, 2, 3), np.nan, np.float32)
                    _area_light_samples(self.camera, output, 0, len(output))
                    self.assertTrue(np.isfinite(output).all())
                    self.assertTrue((output[:, 0] > 0).all())
                    np.testing.assert_array_equal(output[:, 1], np.zeros((128, 3), np.float32))

    def test_area_light_mean_matches_physical_midpoint_quadrature(self):
        self.camera.configure_nee([_quad(2.0, half_extent=1.0)])
        self.camera.shadow_epsilon[None] = 0.0
        for mode in (1, 0):
            self.camera.sampler.mode[None] = mode
            self.camera.sampler.seed[None] = 0xDEADBEEF
            for samples in (1, 2, 4, 8):
                with self.subTest(mode=mode, light_samples=samples):
                    self.camera.light_samples[None] = samples
                    expected = _area_light_midpoint_integral(samples)
                    np.testing.assert_allclose(
                        expected, _area_light_midpoint_integral(samples, 128),
                        rtol=2e-5, atol=0,
                    )
                    output = np.full((32768, 3), np.nan, np.float32)
                    _area_light_estimates(self.camera, output)
                    self.assertTrue(np.isfinite(output).all())
                    self.assertTrue((output > 0.0).all())
                    np.testing.assert_allclose(
                        output.mean(axis=0, dtype=np.float64), expected, rtol=0.01, atol=0,
                    )

    def test_area_light_samples_are_reproducible_across_batches(self):
        self.camera.configure_nee([_quad(2.0)])
        self.camera.light_samples[None] = 4
        for mode in (1, 0):
            with self.subTest(mode=mode):
                self.camera.sampler.mode[None] = mode
                self.camera.sampler.seed[None] = 0xFFFFFFFF
                whole = np.full((128, 2, 3), np.nan, np.float32)
                batched = np.full_like(whole, np.nan)
                _area_light_samples(self.camera, whole, 0, 128)
                for start, stop in ((97, 128), (1, 17), (0, 1), (17, 64), (64, 97)):
                    _area_light_samples(self.camera, batched, start, stop)
                self.assertTrue((whole[:, 0] > 0).all())
                np.testing.assert_array_equal(batched, whole)

    def test_last_bounce_area_light_mean_has_no_bsdf_competitor(self):
        self.camera.configure_nee([_quad(2.0, half_extent=1.0)])
        self.camera.shadow_epsilon[None] = 0.0
        self.camera.max_bounces[None] = 1
        expected = _area_light_midpoint_integral(1, competing_bsdf=False)
        np.testing.assert_allclose(
            expected, _area_light_midpoint_integral(1, 128, competing_bsdf=False),
            rtol=2e-5, atol=0,
        )
        for mode in (1, 0):
            self.camera.sampler.mode[None] = mode
            self.camera.sampler.seed[None] = 0xDEADBEEF
            for samples in (1, 2, 4, 8):
                with self.subTest(mode=mode, light_samples=samples):
                    self.camera.light_samples[None] = samples
                    output = np.full((32768, 3), np.nan, np.float32)
                    _area_light_estimates(self.camera, output)
                    self.assertTrue(np.isfinite(output).all())
                    self.assertTrue((output > 0.0).all())
                    np.testing.assert_allclose(
                        output.mean(axis=0, dtype=np.float64), expected, rtol=0.01, atol=0,
                    )

    def test_close_unregistered_emitter_does_not_match_registered_endpoint(self):
        self.camera.configure_nee([_quad(1.0)])
        for samples in (1, 2, 4, 8):
            self.camera.light_samples[None] = samples
            for direction_scale in (1.0, 7.0):
                with self.subTest(light_samples=samples, direction_scale=direction_scale):
                    self.assertEqual(_endpoint_weight(self.camera, 0.9995, direction_scale), 1.0)
                    expected = (1.0 / np.pi) ** 2 / ((1.0 / np.pi) ** 2 + samples ** 2)
                    self.assertAlmostEqual(
                        _endpoint_weight(self.camera, 1.0, direction_scale), expected, delta=2e-7,
                    )

    def test_indexed_endpoint_mis_requires_matching_primitive_identity(self):
        self.camera.configure_nee([_quad(1.0)])
        self.camera.nee_light_indices[0] = 23
        for samples in (1, 2, 4, 8):
            with self.subTest(light_samples=samples):
                self.camera.light_samples[None] = samples
                self.assertEqual(_indexed_endpoint_weight(self.camera, 1.0, 24), 1.0)
                self.assertEqual(_indexed_endpoint_weight(self.camera, 0.9995, 24), 1.0)
                expected = (1.0 / np.pi) ** 2 / ((1.0 / np.pi) ** 2 + samples ** 2)
                self.assertAlmostEqual(
                    _indexed_endpoint_weight(self.camera, 1.0, 23), expected, delta=2e-7,
                )

    def test_endpoint_mis_ignores_collinear_light_behind_actual_hit(self):
        actual = _quad(2.0, half_extent=1.0)
        behind = _quad(4.0)
        for samples in (1, 2, 4, 8):
            self.camera.light_samples[None] = samples
            for direction_scale in (1.0, 7.0):
                with self.subTest(light_samples=samples, direction_scale=direction_scale):
                    expected = (1.0 / np.pi) ** 2 / ((1.0 / np.pi) ** 2 + samples ** 2)
                    self.camera.configure_nee([actual])
                    baseline = _endpoint_weight(self.camera, 2.0, direction_scale)
                    self.assertAlmostEqual(baseline, expected, delta=2e-7)
                    self.assertGreater(baseline, 0.0)
                    self.assertLess(baseline, 1.0)
                    for lights in ([actual, behind], [behind, actual]):
                        self.camera.configure_nee(lights)
                        self.assertAlmostEqual(
                            _endpoint_weight(self.camera, 2.0, direction_scale), baseline, delta=2e-7
                        )
                    self.camera.configure_nee([behind])
                    self.assertEqual(_endpoint_weight(self.camera, 2.0, direction_scale), 1.0)
                    behind_expected = (1.0 / np.pi) ** 2 / (
                        (1.0 / np.pi) ** 2 + (samples * 16.0) ** 2
                    )
                    self.assertAlmostEqual(
                        _endpoint_weight(self.camera, 4.0, direction_scale),
                        behind_expected, delta=2e-7,
                    )
        self.camera.configure_nee([actual], enabled=False)
        self.assertEqual(_endpoint_weight(self.camera, 2.0, 1.0), 1.0)


class TestCameraDenoise(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.camera = main.Camera(
            _UnoccludedBVH(), image_width=32, aspect_ratio=1.0, vfov=40.0,
            lookfrom=(0.0, 0.0, 3.0), lookat=(0.0, 0.0, 0.0),
            vup=(0.0, 1.0, 0.0), defocus_angle=0.0, focus_distance=3.0,
            enable_denoise=True,
        )
        cls.shape = (32, 32)

    def setUp(self):
        self.camera.edge_depth.from_numpy(np.full(self.shape, 2.0, np.float32))
        normals = np.zeros(self.shape + (3,), np.float32)
        normals[:, :, 2] = 1.0
        self.camera.edge_normal.from_numpy(normals)
        self.camera.edge_albedo.from_numpy(np.full(self.shape + (3,), 0.5, np.float32))
        self.camera.denoise_variance.fill(0.0)
        self.camera.denoise_phi_color[None] = 0.07

    def _filter(self, image, step):
        self.camera.denoise_ping.from_numpy(image)
        self.camera.denoise_pong.fill(float("nan"))
        self.camera.atrous_pass(self.camera.denoise_ping, self.camera.denoise_pong, step)
        result = self.camera.denoise_pong.to_numpy()
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_array_equal(self.camera.denoise_ping.to_numpy(), image)
        return result

    def test_atrous_preserves_constant_hdr_at_every_step(self):
        image = np.empty(self.shape + (3,), np.float32)
        image[:] = (0.125, 4.5, 32.0)
        for depth in (2.0, 1e20):
            self.camera.edge_depth.fill(depth)
            if depth > 1e10:
                self.camera.edge_normal.fill(0.0)
            for variance in (0.0, 1.0):
                self.camera.denoise_variance.fill(variance)
                for step in (1, 2, 4, 8):
                    with self.subTest(depth=depth, variance=variance, step=step):
                        result = self._filter(image, step)
                        np.testing.assert_allclose(result, image, rtol=2e-6, atol=1e-5)

    def test_atrous_preserves_albedo_and_depth_edges_at_near_zero_variance(self):
        image = np.empty(self.shape + (3,), np.float32)
        image[:16] = (0.25, 0.5, 2.0)
        image[16:] = (1.5, 2.5, 6.0)
        self.camera.denoise_variance.fill(1e-12)
        self.camera.denoise_phi_color[None] = 10.0
        for guide in ("albedo", "depth"):
            for step in (1, 2, 4):
                with self.subTest(guide=guide, step=step):
                    self.camera.edge_depth.fill(2.0)
                    self.camera.edge_albedo.fill(0.5)
                    unprotected = self._filter(image, step)
                    self.assertGreater(np.max(np.abs(unprotected[15:17] - image[15:17])), 0.1)
                    if guide == "albedo":
                        albedo = np.empty_like(image)
                        albedo[:16] = 0.05
                        albedo[16:] = 0.95
                        self.camera.edge_albedo.from_numpy(albedo)
                    else:
                        depth = np.full(self.shape, 2.0, np.float32)
                        depth[16:] = 20.0
                        self.camera.edge_depth.from_numpy(depth)
                    protected = self._filter(image, step)
                    np.testing.assert_allclose(protected, image, rtol=2e-6, atol=1e-5)

    def test_atrous_reduces_noisy_flat_patch_variance_without_changing_mean(self):
        noise = np.random.default_rng(1729).normal(size=self.shape + (3,))
        noise *= np.array([0.4, 0.5, 0.6])
        noise -= noise.mean(axis=(0, 1), keepdims=True)
        image = (np.array([2.0, 4.0, 8.0]) + noise).astype(np.float32)
        mean = image.mean(axis=(0, 1), dtype=np.float64)
        variance = image.var(axis=(0, 1), dtype=np.float64)
        for step in (1, 2, 4):
            with self.subTest(step=step):
                self.camera.denoise_variance.fill(0.0)
                zero_variance = self._filter(image, step)
                zero_filtered_variance = zero_variance.var(axis=(0, 1), dtype=np.float64)
                self.assertTrue((zero_filtered_variance > variance * 0.9).all())
                self.camera.denoise_variance.fill(1.0)
                filtered = self._filter(image, step)
                filtered_variance = filtered.var(axis=(0, 1), dtype=np.float64)
                self.assertTrue((filtered_variance < variance * 0.35).all())
                np.testing.assert_allclose(
                    filtered.mean(axis=(0, 1), dtype=np.float64), mean, rtol=0.01, atol=0.003,
                )


class TestFixedBatchIntegrator(unittest.TestCase):
    def test_split_batches_advance_sample_indices_and_match_one_batch(self):
        camera = _SyntheticFixedBatchCamera(
            _UnoccludedBVH(), image_width=7, aspect_ratio=7.0 / 5.0, vfov=40.0,
            lookfrom=(0.0, 0.0, 3.0), lookat=(0.0, 0.0, 0.0),
            vup=(0.0, 1.0, 0.0), defocus_angle=0.0, focus_distance=3.0,
            enable_adaptive=False, enable_denoise=False,
        )
        shape = (7, 5, 3)
        camera.sample_offset[None] = 41
        camera.color_accum.fill(99.0)
        camera.clear_accum()
        self.assertEqual(int(camera.sample_offset[None]), 0)
        np.testing.assert_array_equal(camera.color_accum.to_numpy(), np.zeros(shape, np.float32))
        camera.trace_batch(2)
        self.assertEqual(int(camera.sample_offset[None]), 2)
        np.testing.assert_array_equal(
            camera.color_accum.to_numpy(),
            np.broadcast_to(np.array([0.25, 0.5, 2.0], np.float32), shape),
        )
        camera.trace_batch(3)
        self.assertEqual(int(camera.sample_offset[None]), 5)
        split = camera.color_accum.to_numpy()
        expected = np.broadcast_to(np.array([2.5, 5.0, 20.0], np.float32), shape)
        np.testing.assert_array_equal(split, expected)
        camera.clear_accum()
        self.assertEqual(int(camera.sample_offset[None]), 0)
        np.testing.assert_array_equal(camera.color_accum.to_numpy(), np.zeros(shape, np.float32))
        camera.trace_batch(5)
        self.assertEqual(int(camera.sample_offset[None]), 5)
        whole = camera.color_accum.to_numpy()
        np.testing.assert_array_equal(whole, expected)
        np.testing.assert_array_equal(whole, split)


class TestAdaptiveQueues(unittest.TestCase):
    def test_auto_switches_from_dense_tiles_to_sparse_pixels_and_matches_both(self):
        camera = _SyntheticAdaptiveCamera(
            _UnoccludedBVH(), image_width=35, aspect_ratio=35.0 / 19.0, vfov=40.0,
            lookfrom=(0.0, 0.0, 3.0), lookat=(0.0, 0.0, 0.0),
            vup=(0.0, 1.0, 0.0), defocus_angle=0.0, focus_distance=3.0,
            enable_adaptive=True, enable_denoise=False,
        )
        self.assertEqual((camera.image_width, camera.image_height), (35, 19))
        camera.offline_batch_size = 3
        initial_slots = camera.adaptive_tile_count * camera.ADAPTIVE_TILE_SIZE ** 2
        self.assertGreater(35 * 19 / initial_slots, 0.4)
        transitions = []
        compact = camera.compact_active_pixels

        def record_compaction():
            transitions.append((
                int(camera.adaptive_active_pixels[None]),
                int(camera.adaptive_active_tiles[None]) * camera.ADAPTIVE_TILE_SIZE ** 2,
            ))
            return compact()

        results = {}
        for scheduler in ("pixels", "tiles", "auto"):
            with self.subTest(scheduler=scheduler):
                camera.adaptive_scheduler = scheduler
                transition_count = len(transitions)
                with (
                    mock.patch.object(camera, "trace_adaptive_tile_batch", wraps=camera.trace_adaptive_tile_batch) as tiles,
                    mock.patch.object(camera, "trace_adaptive_pixel_batch", wraps=camera.trace_adaptive_pixel_batch) as pixels,
                    mock.patch.object(camera, "compact_active_pixels", new=record_compaction),
                ):
                    camera.render_adaptive(min_spp=4, max_spp=17, threshold=0.02)
                    if scheduler == "auto":
                        self.assertGreater(tiles.call_count, 0)
                        self.assertGreater(pixels.call_count, 0)
                        self.assertEqual(len(transitions), transition_count + 1)
                    else:
                        self.assertEqual(len(transitions), transition_count)
                counts = camera.adaptive_count.to_numpy()
                mean = camera.adaptive_mean.to_numpy()
                self.assertTrue(((counts >= 4) & (counts <= 17)).all())
                self.assertTrue((counts < 17).any())
                self.assertTrue((counts == 17).any())
                self.assertFalse(camera.adaptive_active.to_numpy().any())
                self.assertEqual(int(camera.adaptive_active_pixels[None]), 0)
                results[scheduler] = (counts, mean)
        self.assertEqual(len(transitions), 1)
        active, slots = transitions[0]
        self.assertGreater(active, 0)
        self.assertLess(active, 35 * 19)
        self.assertLess(active / slots, 0.4)
        for scheduler in ("tiles", "auto"):
            np.testing.assert_array_equal(results[scheduler][0], results["pixels"][0])
            np.testing.assert_allclose(results[scheduler][1], results["pixels"][1], rtol=0, atol=1e-7)

    def test_pixel_and_tile_queues_match_on_partial_tile_and_partial_batch(self):
        camera = _SyntheticAdaptiveCamera(
            _UnoccludedBVH(), image_width=7, aspect_ratio=7.0 / 5.0, vfov=40.0,
            lookfrom=(0.0, 0.0, 3.0), lookat=(0.0, 0.0, 0.0),
            vup=(0.0, 1.0, 0.0), defocus_angle=0.0, focus_distance=3.0,
            enable_adaptive=True, enable_denoise=False,
        )
        self.assertEqual((camera.image_width, camera.image_height), (7, 5))
        camera.offline_batch_size = 3
        i, j = np.indices((7, 5))
        varying = (i + j) % 2 != 0
        expected = np.stack((1.0 + 0.125 * i, 2.0 + 0.25 * j, np.full((7, 5), 4.0)), axis=2)
        results = []
        for scheduler in ("pixels", "tiles"):
            with self.subTest(scheduler=scheduler):
                camera.adaptive_scheduler = scheduler
                camera.clear_adaptive()
                np.testing.assert_array_equal(camera.adaptive_count.to_numpy(), np.zeros((7, 5)))
                camera.render_adaptive(min_spp=4, max_spp=17, threshold=0.02)
                counts = camera.adaptive_count.to_numpy()
                mean = camera.adaptive_mean.to_numpy()
                self.assertTrue(((counts >= 4) & (counts <= 17)).all())
                self.assertTrue((counts[~varying] < 17).all())
                np.testing.assert_array_equal(counts[varying], np.full(varying.sum(), 17))
                self.assertEqual(int(camera.adaptive_active_pixels[None]), 0)
                self.assertFalse(camera.adaptive_active.to_numpy().any())
                if scheduler == "tiles":
                    self.assertEqual(int(camera.adaptive_active_tiles[None]), 0)
                reference = expected.copy()
                reference[varying] -= np.array([0.5, 1.0, 2.0]) / 17.0
                np.testing.assert_allclose(mean, reference, rtol=2e-6, atol=1e-6)
                results.append((counts, mean))
        np.testing.assert_array_equal(results[0][0], results[1][0])
        np.testing.assert_allclose(results[0][1], results[1][1], rtol=0, atol=1e-7)


class TestRenderExportState(unittest.TestCase):
    def test_finish_image_selects_actual_raw_render_and_clears_stale_denoise_result(self):
        camera = _SyntheticFixedBatchCamera(
            _UnoccludedBVH(), image_width=8, aspect_ratio=1.0, vfov=40.0,
            lookfrom=(0.0, 0.0, 3.0), lookat=(0.0, 0.0, 0.0),
            vup=(0.0, 1.0, 0.0), defocus_angle=0.0, focus_distance=3.0,
            enable_adaptive=True, enable_denoise=True,
        )
        fixed = np.full((8, 8, 3), [2.0, 4.0, 8.0], np.float32)
        adaptive = np.full((8, 8, 3), [3.0, 6.0, 12.0], np.float32)
        camera.color_accum.from_numpy(fixed * 4.0)
        camera.pixel_sample_scale = 0.25
        camera.adaptive_mean.from_numpy(adaptive)
        camera.denoise_ping.fill(99.0)
        camera.denoise_enabled[None] = 0
        with mock.patch.object(camera, "prepare_surface_guides") as guides:
            for use_adaptive in (False, True, False):
                with self.subTest(adaptive=use_adaptive):
                    camera.denoised_result = camera.denoise_ping
                    camera.finish_image(adaptive=use_adaptive)
                    self.assertEqual(camera.last_render_adaptive, use_adaptive)
                    self.assertIsNone(camera.denoised_result)
                    np.testing.assert_array_equal(camera.linear_image(), adaptive if use_adaptive else fixed)
            guides.assert_not_called()

    def test_scene_export_path_suffixes_filename_without_changing_relative_parent(self):
        for extension in ("json", "npz"):
            with self.subTest(extension=extension):
                path = os.path.join("nested", f"relative.{extension}")
                expected = os.path.join("nested", f"relative_cornell.{extension}")
                self.assertEqual(main._scene_export_path(path, "cornell"), expected)
        self.assertEqual(main._scene_export_path("relative.json", "checker"), "relative_checker.json")

    def test_empty_scene_export_path_stays_disabled(self):
        self.assertEqual(main._scene_export_path("", "cornell"), "")


class TestQualitySettings(unittest.TestCase):
    def test_explicit_sample_counts_override_quality_defaults(self):
        for quality in ("quick", "normal", "final", "manual"):
            for scene_spp in (32, 256, 2048):
                for spp in (None, 64, 1024, 1):
                    with self.subTest(quality=quality, scene_spp=scene_spp, spp=spp):
                        args = argparse.Namespace(
                            quality=quality, spp=spp, denoise=False, no_adaptive=False,
                            offline_batch_size=8, denoise_iterations=3, denoise_phi_color=0.07,
                        )
                        settings = main._quality_settings(args, scene_spp)
                        expected = spp
                        if expected is None:
                            cap = {"quick": 96, "normal": 320}.get(quality, scene_spp)
                            expected = min(scene_spp, cap)
                        self.assertEqual(settings["max_spp"], expected)
                        self.assertGreaterEqual(settings["min_spp"], 1)
                        self.assertLessEqual(settings["min_spp"], settings["max_spp"])
                        if quality == "final":
                            self.assertEqual(settings["min_spp"], expected)


if __name__ == "__main__":
    unittest.main()
