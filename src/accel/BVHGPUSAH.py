import math

import numpy as np
from numba import njit

DEFAULT_LEAF_SIZE = 2


def _expand_bits(v):
    v = v.astype(np.uint64) & np.uint64(0x3FF)
    v = (v | (v << np.uint64(16))) & np.uint64(0x030000FF)
    v = (v | (v << np.uint64(8))) & np.uint64(0x0300F00F)
    v = (v | (v << np.uint64(4))) & np.uint64(0x030C30C3)
    v = (v | (v << np.uint64(2))) & np.uint64(0x09249249)
    return v.astype(np.uint32)


def _morton3d(x, y, z):
    xx = _expand_bits((np.clip(x, 0.0, 1.0) * 1023.0).astype(np.uint32))
    yy = _expand_bits((np.clip(y, 0.0, 1.0) * 1023.0).astype(np.uint32))
    zz = _expand_bits((np.clip(z, 0.0, 1.0) * 1023.0).astype(np.uint32))
    return (xx << np.uint32(2)) | (yy << np.uint32(1)) | zz


class GPUSAHBVHBuilder:
    def __init__(self, prim_mins, prim_maxs, prim_refs, leaf_size=DEFAULT_LEAF_SIZE):
        self.prim_mins = prim_mins
        self.prim_maxs = prim_maxs
        self.prim_refs = prim_refs
        self.leaf_size = max(1, int(leaf_size))
        self.ordered_prim_indices = []
        self.heap_aabb_min = None
        self.heap_aabb_max = None
        self.heap_left_or_first = None
        self.heap_count = None
        self.max_heap_idx = 0

    def build(self):
        n = self.prim_mins.shape[0]
        if n == 0:
            return
        from time import perf_counter as _pf

        t_total = _pf()
        prim_mins = self.prim_mins
        prim_maxs = self.prim_maxs
        prim_refs = self.prim_refs
        centroids = (prim_mins + prim_maxs) * 0.5
        scene_min = prim_mins.min(axis=0)
        scene_max = prim_maxs.max(axis=0)
        extent = np.maximum(scene_max - scene_min, 1e-10)
        normed = np.clip((centroids - scene_min) / extent, 0.0, 1.0)

        t0 = _pf()
        morton = _morton3d(normed[:, 0], normed[:, 1], normed[:, 2])
        skey = (morton.astype(np.uint64) << np.uint64(32)) | np.arange(
            n, dtype=np.uint64
        )
        order = np.argsort(skey, kind="mergesort").astype(np.int32)
        t_sort = _pf() - t0

        s_mins = prim_mins[order]
        s_maxs = prim_maxs[order]
        s_refs = prim_refs[order]
        s_codes = morton[order]
        self.ordered_prim_indices = s_refs

        leaf_size = self.leaf_size
        nl = (n + leaf_size - 1) // leaf_size
        leaf_starts = np.arange(0, n, leaf_size, dtype=np.int32)
        leaf_ends = np.minimum(leaf_starts + leaf_size, n).astype(np.int32)
        lmn = np.minimum.reduceat(s_mins, leaf_starts, axis=0).astype(np.float32)
        lmx = np.maximum.reduceat(s_maxs, leaf_starts, axis=0).astype(np.float32)
        ls = leaf_starts
        lc_arr = (leaf_ends - leaf_starts).astype(np.int32)
        lcode = s_codes[leaf_starts]

        if nl <= 1:
            self.heap_aabb_min = np.zeros((2, 3), dtype=np.float32)
            self.heap_aabb_max = np.zeros((2, 3), dtype=np.float32)
            self.heap_left_or_first = np.zeros(2, dtype=np.int32)
            self.heap_count = np.zeros(2, dtype=np.int32)
            self.heap_aabb_min[1] = lmn[0]
            self.heap_aabb_max[1] = lmx[0]
            self.heap_left_or_first[1] = 0
            self.heap_count[1] = int(lc_arr[0])
            self.max_heap_idx = 1
            return

        t0 = _pf()
        ni = nl - 1
        nt = ni + nl
        n_left = np.full(ni, -1, dtype=np.int32)
        n_right = np.full(ni, -1, dtype=np.int32)
        n_parent = np.full(nt, -1, dtype=np.int32)
        _karras(lcode, nl, n_left, n_right, n_parent)
        t_hier = _pf() - t0

        t0 = _pf()
        nmn = np.full((nt, 3), 1e30, dtype=np.float32)
        nmx = np.full((nt, 3), -1e30, dtype=np.float32)
        nmn[ni : ni + nl] = lmn
        nmx[ni : ni + nl] = lmx
        vc = np.zeros(nt, dtype=np.int32)
        _refit_bottom_up(nl, ni, n_parent, n_left, n_right, nmn, nmx, vc)
        t_refit = _pf() - t0

        t0 = _pf()
        h_mn = np.zeros((nt + 1, 3), dtype=np.float32)
        h_mx = np.zeros((nt + 1, 3), dtype=np.float32)
        h_lf = np.zeros(nt + 1, dtype=np.int32)
        h_c = np.zeros(nt + 1, dtype=np.int32)
        stack_arr = np.zeros((nt, 2), dtype=np.int32)
        max_hi = _build_compact(
            nmn, nmx, ls, lc_arr, n_left, n_right, ni, h_mn, h_mx, h_lf, h_c, stack_arr
        )

        self.max_heap_idx = max_hi
        sz = max_hi + 1
        self.heap_aabb_min = h_mn[:sz].copy()
        self.heap_aabb_max = h_mx[:sz].copy()
        self.heap_left_or_first = h_lf[:sz].copy()
        self.heap_count = h_c[:sz].copy()
        t_heap = _pf() - t0

        t_all = _pf() - t_total
        print(
            f"  [Karras BVH] {n:,} prims -> {nl:,} leaves"
            f" ({leaf_size}/leaf): sort={t_sort:.3f}s hier={t_hier:.3f}s"
            f" refit={t_refit:.3f}s compact={t_heap:.3f}s total={t_all:.3f}s",
            flush=True,
        )


@njit(cache=True)
def _refit_bottom_up(nl, ni, n_parent, n_left, n_right, nmn, nmx, vc):
    for leaf_i in range(nl):
        idx = n_parent[ni + leaf_i]
        while idx >= 0:
            vc[idx] += 1
            if vc[idx] < 2:
                break
            a = n_left[idx]
            b = n_right[idx]
            for d in range(3):
                nmn[idx, d] = nmn[a, d] if nmn[a, d] < nmn[b, d] else nmn[b, d]
                nmx[idx, d] = nmx[a, d] if nmx[a, d] > nmx[b, d] else nmx[b, d]
            idx = n_parent[idx]


@njit(cache=True)
def _delta(i, j, codes, n):
    if j < 0 or j >= n:
        return -1
    ci = codes[i]
    cj = codes[j]
    if ci == cj:
        xor_ij = i ^ j
        if xor_ij == 0:
            return 62
        return 32 + 31 - int(math.floor(math.log2(float(xor_ij))))
    xor_c = ci ^ cj
    return 31 - int(math.floor(math.log2(float(xor_c))))


@njit(cache=True)
def _karras(codes, n, n_left, n_right, n_parent):
    ni = n - 1
    for i in range(ni):
        dr = _delta(i, i + 1, codes, n)
        dl = _delta(i, i - 1, codes, n)
        d = 1 if dr > dl else -1
        dmin = _delta(i, i - d, codes, n)
        lmax = 2
        while True:
            c = i + lmax * d
            if c < 0 or c >= n or _delta(i, c, codes, n) <= dmin:
                break
            lmax <<= 1
        l = 0
        t = lmax >> 1
        while t >= 1:
            c = i + (l + t) * d
            if 0 <= c < n and _delta(i, c, codes, n) > dmin:
                l += t
            t >>= 1
        j = i + l * d
        dn = _delta(i, j, codes, n)
        s = 0
        t = (abs(j - i) + 1) >> 1
        if t < 1:
            t = 1
        while True:
            c = i + (s + t) * d
            if 0 <= c < n and _delta(i, c, codes, n) > dn:
                s += t
            if t <= 1:
                break
            t = (t + 1) >> 1
        gamma = i + s * d + min(d, 0)
        if min(i, j) == gamma:
            lc = ni + gamma
        else:
            lc = gamma
        if max(i, j) == gamma + 1:
            rc = ni + gamma + 1
        else:
            rc = gamma + 1
        n_left[i] = lc
        n_right[i] = rc
        n_parent[lc] = i
        n_parent[rc] = i


@njit(cache=True)
def _build_compact(nmn, nmx, ls, lc_arr, n_left, n_right, ni, h_mn, h_mx, h_lf, h_c, stack_arr):
    max_hi = 1
    next_free = 2
    stack_ptr = 1
    stack_arr[0, 0] = 0
    stack_arr[0, 1] = 1
    while stack_ptr > 0:
        stack_ptr -= 1
        fi = stack_arr[stack_ptr, 0]
        oi = stack_arr[stack_ptr, 1]
        for d in range(3):
            h_mn[oi, d] = nmn[fi, d]
            h_mx[oi, d] = nmx[fi, d]
        if oi > max_hi:
            max_hi = oi
        if fi >= ni:
            leaf_i = fi - ni
            h_lf[oi] = int(ls[leaf_i])
            h_c[oi] = int(lc_arr[leaf_i])
        else:
            left_slot = next_free
            right_slot = next_free + 1
            next_free += 2
            h_lf[oi] = left_slot
            h_c[oi] = 0
            stack_arr[stack_ptr, 0] = n_right[fi]
            stack_arr[stack_ptr, 1] = right_slot
            stack_ptr += 1
            stack_arr[stack_ptr, 0] = n_left[fi]
            stack_arr[stack_ptr, 1] = left_slot
            stack_ptr += 1
            if right_slot > max_hi:
                max_hi = right_slot
    return max_hi
