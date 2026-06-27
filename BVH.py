import taichi as ti

from interval import Interval
from Ray import Ray

vec3 = ti.math.vec3


def _merge_aabb(mina, maxa, minb, maxb):
    """Host-side AABB union."""
    return vec3(
        min(mina[0], minb[0]),
        min(mina[1], minb[1]),
        min(mina[2], minb[2]),
    ), vec3(
        max(maxa[0], maxb[0]),
        max(maxa[1], maxb[1]),
        max(maxa[2], maxb[2]),
    )


def _aabb_surface_area(mn, mx):
    """Full surface area; epsilon floor for degenerate slabs."""
    dx = max(0.0, float(mx[0] - mn[0]))
    dy = max(0.0, float(mx[1] - mn[1]))
    dz = max(0.0, float(mx[2] - mn[2]))
    return max(2.0 * (dx * dy + dy * dz + dz * dx), 1e-20)

@ti.dataclass
class BVHNode:
    """
    BVH node stored in a flat heap-indexed array.

    Attributes
    ----------
    aabb_min : vec3
        Minimum corner of the bounding box.
    aabb_max : vec3
        Maximum corner of the bounding box.
    is_leaf : ti.i32
        1 if this is a leaf node, 0 otherwise.
    first_prim : ti.i32
        For leaves: index into sphere_indices of first primitive.
        For internal: unused (children are at 2*i and 2*i+1).
    prim_count : ti.i32
        For leaves: number of primitives in this node.
        For internal: 0.
    """

    aabb_min: vec3
    aabb_max: vec3
    is_leaf: ti.i32
    first_prim: ti.i32
    prim_count: ti.i32


@ti.func
def aabb_hit(
    aabb_min: vec3, aabb_max: vec3, r: Ray, t_min: ti.f32, t_max: ti.f32
) -> ti.i32:
    """Branchless slab AABB test — no warp divergence."""
    miss = 0
    for axis in ti.static(range(3)):
        d = r.direction[axis]
        parallel = ti.abs(d) < 1e-12
        outside = parallel & ((r.origin[axis] < aabb_min[axis]) | (r.origin[axis] > aabb_max[axis]))
        miss = ti.select(outside, 1, miss)
        safe_d = ti.select(parallel, ti.select(d < 0.0, -1e-12, 1e-12), d)
        inv_d = 1.0 / safe_d
        t0 = (aabb_min[axis] - r.origin[axis]) * inv_d
        t1 = (aabb_max[axis] - r.origin[axis]) * inv_d
        tnn = ti.select(parallel, t_min, ti.min(t0, t1))
        tff = ti.select(parallel, t_max, ti.max(t0, t1))
        t_min = ti.max(tnn, t_min)
        t_max = ti.min(tff, t_max)
    return ti.select((miss == 0) & (t_max >= ti.max(0.0, t_min)), 1, 0)


@ti.func
def aabb_ray_t_enter(
    aabb_min: vec3, aabb_max: vec3, r: Ray, t_min: ti.f32, t_max: ti.f32
) -> ti.f32:
    """Branchless ray-entry t for near-child ordering."""
    t0 = t_min
    t1 = t_max
    miss = 0
    for axis in ti.static(range(3)):
        d = r.direction[axis]
        parallel = ti.abs(d) < 1e-12
        outside = parallel & ((r.origin[axis] < aabb_min[axis]) | (r.origin[axis] > aabb_max[axis]))
        miss = ti.select(outside, 1, miss)
        safe_d = ti.select(parallel, ti.select(d < 0.0, -1e-12, 1e-12), d)
        inv_d = 1.0 / safe_d
        tn = (aabb_min[axis] - r.origin[axis]) * inv_d
        tf = (aabb_max[axis] - r.origin[axis]) * inv_d
        t0 = ti.max(ti.select(parallel, t0, ti.min(tn, tf)), t0)
        t1 = ti.min(ti.select(parallel, t1, ti.max(tn, tf)), t1)
    return ti.select((miss != 0) | (t1 < t0), 1e30, t0)







class PythonBVHNode:
    """
    Python-side BVH node for tree construction.

    Used during BVH building on CPU before conversion to GPU format.
    """

    def __init__(self):
        self.aabb_min = None
        self.aabb_max = None
        self.left = None
        self.right = None
        self.is_leaf = False
        self.first_prim = 0
        self.prim_count = 0


class BVHBuilder:
    """
    Builds a BVH from a list of spheres in Python, then converts to
    a heap-indexed flat array for Taichi GPU traversal.

    layout="sphere" tuples: (cx, cy, cz, radius, sphere_index, aabb_min, aabb_max)
    layout="prim" tuples: (aabb_min, aabb_max, prim_index)
    """

    MAX_LEAF_SIZE = 4

    def __init__(self, primitives_data, layout: str = "sphere", split: str = "sah"):
        """
        Initialize BVH builder with primitive bounds on the host.

        Parameters
        ----------
        primitives_data : list
            Sphere layout: (cx, cy, cz, radius, sphere_index, aabb_min, aabb_max).
            Prim layout: (aabb_min, aabb_max, prim_index).
        layout : str
            "sphere" (default) or "prim".
        split : str
            "sah" (surface-area heuristic, default) or "median" (centroid median).
        """
        self.layout = layout
        self.split = split
        self.primitives = primitives_data
        self.ordered_prim_indices = []
        self.nodes = []
        self.root = None

    def _aabb_of(self, idx):
        prim = self.primitives[idx]
        if self.layout == "sphere":
            return prim[5], prim[6]
        return prim[0], prim[1]

    def _prim_storage_index(self, idx):
        prim = self.primitives[idx]
        if self.layout == "sphere":
            return prim[4]
        return prim[2]

    def _morton_code(self, idx, scene_min, inv_extent):
        c = self._compute_centroid(idx)
        nx = max(0.0, min(1.0, (float(c[0]) - scene_min[0]) * inv_extent[0]))
        ny = max(0.0, min(1.0, (float(c[1]) - scene_min[1]) * inv_extent[1]))
        nz = max(0.0, min(1.0, (float(c[2]) - scene_min[2]) * inv_extent[2]))

        def expand_bits(v):
            v = int(v) & 0x3FF
            v = (v | (v << 16)) & 0x030000FF
            v = (v | (v << 8)) & 0x0300F00F
            v = (v | (v << 4)) & 0x030C30C3
            v = (v | (v << 2)) & 0x09249249
            return v

        xx = expand_bits(int(nx * 1023.0))
        yy = expand_bits(int(ny * 1023.0))
        zz = expand_bits(int(nz * 1023.0))
        return (xx << 2) | (yy << 1) | zz

    def _presort_morton(self, indices):
        if len(indices) <= 1:
            return indices
        scene_min, scene_max = self._compute_bounds(indices)
        inv_extent = [
            1.0 / max(float(scene_max[0] - scene_min[0]), 1e-10),
            1.0 / max(float(scene_max[1] - scene_min[1]), 1e-10),
            1.0 / max(float(scene_max[2] - scene_min[2]), 1e-10),
        ]
        return sorted(indices, key=lambda i: self._morton_code(i, scene_min, inv_extent))

    def build(self):
        """Build the BVH tree and convert to heap array."""
        if len(self.primitives) == 0:
            return

        indices = self._presort_morton(list(range(len(self.primitives))))
        self.root = self._build_recursive(indices)

        self._convert_to_heap_array()

    def _compute_bounds(self, indices):
        """
        Compute the bounding box encompassing all primitives at given indices.

        Args:
            indices: List of primitive indices to bound.

        Returns:
            Tuple of (min_corner, max_corner) vectors.
        """
        if not indices:
            return vec3(0, 0, 0), vec3(0, 0, 0)

        min_corner = vec3(float("inf"), float("inf"), float("inf"))
        max_corner = vec3(float("-inf"), float("-inf"), float("-inf"))

        for idx in indices:
            aabb_min, aabb_max = self._aabb_of(idx)

            min_corner = vec3(
                min(min_corner[0], aabb_min[0]),
                min(min_corner[1], aabb_min[1]),
                min(min_corner[2], aabb_min[2]),
            )
            max_corner = vec3(
                max(max_corner[0], aabb_max[0]),
                max(max_corner[1], aabb_max[1]),
                max(max_corner[2], aabb_max[2]),
            )

        return min_corner, max_corner

    def _compute_centroid(self, idx):
        """
        Compute the centroid of a primitive's AABB.

        Args:
            idx: Primitive index.

        Returns:
            Centroid position as vec3.
        """
        aabb_min, aabb_max = self._aabb_of(idx)
        return vec3(
            (aabb_min[0] + aabb_max[0]) * 0.5,
            (aabb_min[1] + aabb_max[1]) * 0.5,
            (aabb_min[2] + aabb_max[2]) * 0.5,
        )

    def _sah_partition(self, indices):
        """
        Partition primitive indices by SAH along the best axis / split index.

        Cost model: sum over children of (primitive count * child AABB surface area),
        matching the usual greedy SAH used in interactive ray tracers.
        """
        n = len(indices)
        if n <= 1:
            return indices[:], indices[:]

        best_cost = float("inf")
        best_left = None
        best_right = None

        for axis in range(3):
            sorted_idx = sorted(indices, key=lambda i: self._compute_centroid(i)[axis])
            prefix_mn = []
            prefix_mx = []
            cmn = vec3(float("inf"), float("inf"), float("inf"))
            cmx = vec3(float("-inf"), float("-inf"), float("-inf"))
            for idx in sorted_idx:
                amin, amax = self._aabb_of(idx)
                cmn, cmx = _merge_aabb(cmn, cmx, amin, amax)
                prefix_mn.append(vec3(cmn[0], cmn[1], cmn[2]))
                prefix_mx.append(vec3(cmx[0], cmx[1], cmx[2]))

            suffix_mn = [None] * n
            suffix_mx = [None] * n
            cmn = vec3(float("inf"), float("inf"), float("inf"))
            cmx = vec3(float("-inf"), float("-inf"), float("-inf"))
            for j in range(n - 1, -1, -1):
                idx = sorted_idx[j]
                amin, amax = self._aabb_of(idx)
                cmn, cmx = _merge_aabb(cmn, cmx, amin, amax)
                suffix_mn[j] = vec3(cmn[0], cmn[1], cmn[2])
                suffix_mx[j] = vec3(cmx[0], cmx[1], cmx[2])

            for k in range(1, n):
                lmn, lmx = prefix_mn[k - 1], prefix_mx[k - 1]
                rmn, rmx = suffix_mn[k], suffix_mx[k]
                cost = k * _aabb_surface_area(lmn, lmx) + (n - k) * _aabb_surface_area(
                    rmn, rmx
                )
                if cost < best_cost:
                    best_cost = cost
                    best_left = sorted_idx[:k]
                    best_right = sorted_idx[k:]

        if best_left is None or len(best_left) == 0 or len(best_right) == 0:
            return self._median_partition(indices)

        return best_left, best_right

    def _median_partition(self, indices):
        """Longest-axis centroid median split (legacy)."""
        node_mn, node_mx = self._compute_bounds(indices)
        extent = vec3(
            node_mx[0] - node_mn[0],
            node_mx[1] - node_mn[1],
            node_mx[2] - node_mn[2],
        )
        if extent[0] >= extent[1] and extent[0] >= extent[2]:
            axis = 0
        elif extent[1] >= extent[2]:
            axis = 1
        else:
            axis = 2
        centroids = [(idx, self._compute_centroid(idx)[axis]) for idx in indices]
        centroids.sort(key=lambda x: x[1])
        sorted_indices = [c[0] for c in centroids]
        mid = len(indices) // 2
        return sorted_indices[:mid], sorted_indices[mid:]

    def _build_recursive(self, indices):
        """
        Recursively build the BVH tree using SAH (default) or median split.

        Args:
            indices: List of primitive indices to partition.

        Returns:
            PythonBVHNode representing the root of this subtree.
        """
        node = PythonBVHNode()
        node.aabb_min, node.aabb_max = self._compute_bounds(indices)

        n = len(indices)

        if n <= self.MAX_LEAF_SIZE:
            node.is_leaf = True
            node.first_prim = len(self.ordered_prim_indices)
            node.prim_count = n
            for idx in indices:
                self.ordered_prim_indices.append(self._prim_storage_index(idx))
            return node

        if self.split == "median":
            left_indices, right_indices = self._median_partition(indices)
        else:
            left_indices, right_indices = self._sah_partition(indices)

        node.left = self._build_recursive(left_indices)
        node.right = self._build_recursive(right_indices)
        node.is_leaf = False

        return node

    def _convert_to_heap_array(self):
        """Convert the tree to a heap-indexed flat array (1-indexed)."""
        if self.root is None:
            return

        
        max_depth = self._get_depth(self.root)
        max_nodes = (1 << (max_depth + 1)) - 1  

        
        self.nodes = [None] * (max_nodes + 1)  

        
        self._assign_heap_indices(self.root, 1)
        self._refit_heap_bounds()

    def _refit_heap_bounds(self):
        """Recompute internal heap-node bounds from children after flattening."""
        for idx in range(len(self.nodes) - 1, 0, -1):
            node = self.nodes[idx]
            if node is None or node.is_leaf:
                continue
            left = self.nodes[2 * idx] if 2 * idx < len(self.nodes) else None
            right = self.nodes[2 * idx + 1] if 2 * idx + 1 < len(self.nodes) else None
            if left is not None and right is not None:
                node.aabb_min, node.aabb_max = _merge_aabb(
                    left.aabb_min, left.aabb_max, right.aabb_min, right.aabb_max
                )
            elif left is not None:
                node.aabb_min = vec3(float(left.aabb_min[0]), float(left.aabb_min[1]), float(left.aabb_min[2]))
                node.aabb_max = vec3(float(left.aabb_max[0]), float(left.aabb_max[1]), float(left.aabb_max[2]))
            elif right is not None:
                node.aabb_min = vec3(float(right.aabb_min[0]), float(right.aabb_min[1]), float(right.aabb_min[2]))
                node.aabb_max = vec3(float(right.aabb_max[0]), float(right.aabb_max[1]), float(right.aabb_max[2]))

    def _get_depth(self, node):
        """
        Get the maximum depth of the tree.

        Args:
            node: Root node of subtree to measure.

        Returns:
            Maximum depth from this node to any leaf.
        """
        if node is None or node.is_leaf:
            return 0
        return 1 + max(self._get_depth(node.left), self._get_depth(node.right))

    def _assign_heap_indices(self, node, idx):
        """
        Assign heap indices to nodes using breadth-first ordering.

        Args:
            node: Current node to assign.
            idx: Heap index to assign (children at 2*idx and 2*idx+1).
        """
        if node is None or idx >= len(self.nodes):
            return

        self.nodes[idx] = node

        if not node.is_leaf:
            self._assign_heap_indices(node.left, 2 * idx)
            self._assign_heap_indices(node.right, 2 * idx + 1)

    def get_node_count(self):
        """
        Return the number of valid nodes.

        Returns:
            Count of non-None nodes in the heap array.
        """
        return sum(1 for n in self.nodes if n is not None)

    def get_max_index(self):
        """
        Return the maximum valid heap index.

        Returns:
            Highest index containing a valid node.
        """
        for i in range(len(self.nodes) - 1, 0, -1):
            if self.nodes[i] is not None:
                return i
        return 0







@ti.data_oriented
class BVH:
    """
    GPU-resident BVH for fast ray intersection queries.

    Uses iterative stack-based traversal (no recursion on GPU).
    """

    MAX_STACK_DEPTH = 64  

    def __init__(self, spheres_field, sphere_count):
        """
        Initialize BVH from sphere field.

        Parameters
        ----------
        spheres_field : ti.Field
            Taichi field containing Sphere structs.
        sphere_count : int
            Number of spheres in the field.
        """
        self.spheres = spheres_field
        self.sphere_count = sphere_count

        
        spheres_data = []
        for i in range(sphere_count):
            s = spheres_field[i]
            center = s.center.origin  
            radius = s.radius
            aabb_min = vec3(s.aabb.x.min, s.aabb.y.min, s.aabb.z.min)
            aabb_max = vec3(s.aabb.x.max, s.aabb.y.max, s.aabb.z.max)
            spheres_data.append(
                (center[0], center[1], center[2], radius, i, aabb_min, aabb_max)
            )

        
        builder = BVHBuilder(spheres_data, split="sah")
        builder.build()

        
        max_idx = max(builder.get_max_index(), 1)
        self.nodes = BVHNode.field(shape=(max_idx + 1,))
        self.node_count = max_idx

        
        n_prims = len(builder.ordered_prim_indices)
        self.prim_indices = ti.field(dtype=ti.i32, shape=(max(n_prims, 1),))
        self.prim_count = n_prims

        
        self._upload_to_gpu(builder)

        
        self.temp_rec = None  

    def _upload_to_gpu(self, builder):
        """
        Upload BVH data to GPU fields.

        Args:
            builder: BVHBuilder instance containing the built tree.
        """
        
        for i in range(1, self.node_count + 1):
            node = builder.nodes[i]
            if node is not None:
                self.nodes[i] = BVHNode(
                    aabb_min=node.aabb_min,
                    aabb_max=node.aabb_max,
                    is_leaf=1 if node.is_leaf else 0,
                    first_prim=node.first_prim,
                    prim_count=node.prim_count,
                )
            else:
                
                self.nodes[i] = BVHNode(
                    aabb_min=vec3(0, 0, 0),
                    aabb_max=vec3(0, 0, 0),
                    is_leaf=0,
                    first_prim=0,
                    prim_count=0,
                )

        
        for i, idx in enumerate(builder.ordered_prim_indices):
            self.prim_indices[i] = idx

    @ti.func
    def hit(
        self, r: Ray, ray_t: Interval, rec, temp_rec, x: ti.i32, y: ti.i32
    ) -> ti.i32:
        """
        Find the closest intersection between a ray and any object in the BVH.

        Uses iterative stack-based traversal to avoid GPU recursion overhead.

        Parameters
        ----------
        r : Ray
            The ray to test.
        ray_t : Interval
            Valid t range for intersections.
        rec : hit_record field
            Output hit record field.
        temp_rec : hit_record field
            Temporary hit record for sphere tests.
        x, y : int
            Pixel coordinates for output indexing.

        Returns
        -------
        int
            1 if any object was hit, 0 otherwise.
        """
        did_hit = 0
        closest_so_far = ray_t.max

        
        
        
        stack = ti.Vector([0] * 64, dt=ti.i32)
        stack_ptr = 0

        
        current = 1

        
        step = 0
        while (step < 1000) & ((current > 0) | (stack_ptr > 0)):
            step += 1
            if current > 0 and current <= self.node_count:
                node_idx = current
                node = self.nodes[node_idx]
                current = 0

                
                if aabb_hit(node.aabb_min, node.aabb_max, r, ray_t.min, closest_so_far):
                    if node.is_leaf == 1:
                        
                        for i in range(node.prim_count):
                            prim_idx = self.prim_indices[node.first_prim + i]
                            if self.spheres[prim_idx].hit(
                                r, Interval(ray_t.min, closest_so_far), temp_rec
                            ):
                                did_hit = 1
                                closest_so_far = temp_rec[None].t
                                rec[x, y].t = temp_rec[None].t
                                rec[x, y].p = temp_rec[None].p
                                rec[x,y].u = temp_rec[None].u
                                rec[x,y].v = temp_rec[None].v
                                rec[x, y].normal = temp_rec[None].normal
                                rec[x, y].front_face = temp_rec[None].front_face
                                rec[x, y].mat_id = temp_rec[None].mat_id
                    else:
                        
                        left_child = 2 * node_idx
                        right_child = 2 * node_idx + 1

                        
                        if right_child <= self.node_count and stack_ptr < 63:
                            stack[stack_ptr] = right_child
                            stack_ptr += 1

                        if left_child <= self.node_count:
                            current = left_child
                            continue

            
            if stack_ptr > 0:
                stack_ptr -= 1
                current = stack[stack_ptr]
            else:
                break

        return did_hit


@ti.data_oriented
class GPULBVHBuilder:
    """
    GPU-backed linear BVH using Morton codes.

    Steps:
      1. Compute 30-bit Morton code per primitive centroid
      2. Sort Morton codes and primitive ids with Taichi's GPU parallel sort
      3. Build a balanced heap BVH over Morton-ordered leaves
      4. Refit bounding boxes bottom-up

    Produces the same heap-indexed flat array format as BVHBuilder.
    """

    MAX_LEAF_SIZE = 4

    def __init__(self, primitives_data, layout: str = "prim"):
        self.layout = layout
        self.primitives = primitives_data
        self.ordered_prim_indices = []
        self.nodes = []
        self.root = None

    def _aabb_of(self, idx):
        prim = self.primitives[idx]
        if self.layout == "sphere":
            return prim[5], prim[6]
        return prim[0], prim[1]

    def _prim_storage_index(self, idx):
        prim = self.primitives[idx]
        if self.layout == "sphere":
            return prim[4]
        return prim[2]

    @staticmethod
    def _expand_bits(v):
        v = v & 0x3FF
        v = (v | (v << 16)) & 0x30000FF
        v = (v | (v << 8)) & 0x300F00F
        v = (v | (v << 4)) & 0x30C30C3
        v = (v | (v << 2)) & 0x9249249
        return v

    @staticmethod
    def _morton3d(x, y, z):
        xx = GPULBVHBuilder._expand_bits(int(x * 1023.0))
        yy = GPULBVHBuilder._expand_bits(int(y * 1023.0))
        zz = GPULBVHBuilder._expand_bits(int(z * 1023.0))
        return (xx << 2) | (yy << 1) | zz

    @staticmethod
    def _clz(a, b):
        xor = a ^ b
        if xor == 0:
            return 30
        cnt = 0
        for bit in range(29, -1, -1):
            if xor & (1 << bit):
                return cnt
            cnt += 1
        return 30

    @staticmethod
    @ti.func
    def _expand_bits_gpu(v: ti.u32) -> ti.u32:
        v = v & ti.u32(0x3FF)
        v = (v | (v << ti.u32(16))) & ti.u32(0x030000FF)
        v = (v | (v << ti.u32(8))) & ti.u32(0x0300F00F)
        v = (v | (v << ti.u32(4))) & ti.u32(0x030C30C3)
        v = (v | (v << ti.u32(2))) & ti.u32(0x09249249)
        return v

    @staticmethod
    @ti.func
    def _morton3d_gpu(x: ti.f32, y: ti.f32, z: ti.f32) -> ti.u32:
        xx = GPULBVHBuilder._expand_bits_gpu(ti.cast(ti.math.clamp(x, 0.0, 1.0) * 1023.0, ti.u32))
        yy = GPULBVHBuilder._expand_bits_gpu(ti.cast(ti.math.clamp(y, 0.0, 1.0) * 1023.0, ti.u32))
        zz = GPULBVHBuilder._expand_bits_gpu(ti.cast(ti.math.clamp(z, 0.0, 1.0) * 1023.0, ti.u32))
        return (xx << ti.u32(2)) | (yy << ti.u32(1)) | zz

    @ti.kernel
    def _compute_morton_codes(
        self,
        mins: ti.template(),
        maxs: ti.template(),
        codes: ti.template(),
        values: ti.template(),
        sort_count: ti.i32,
        prim_count: ti.i32,
        scene_min: vec3,
        inv_extent: vec3,
    ):
        for i in range(sort_count):
            if i < prim_count:
                mn = mins[i]
                mx = maxs[i]
                c = (mn + mx) * 0.5
                n = (c - scene_min) * inv_extent
                codes[i] = GPULBVHBuilder._morton3d_gpu(n.x, n.y, n.z)
                values[i] = i
            else:
                codes[i] = ti.u32(0xFFFFFFFF)
                values[i] = i

    @ti.kernel
    def _bitonic_sort_step(
        self,
        codes: ti.template(),
        values: ti.template(),
        sort_count: ti.i32,
        j: ti.i32,
        k: ti.i32,
    ):
        for i in range(sort_count):
            ixj = i ^ j
            if ixj > i:
                ci = codes[i]
                cj = codes[ixj]
                vi = values[i]
                vj = values[ixj]
                ascending = (i & k) == 0
                greater = (ci > cj) | ((ci == cj) & (vi > vj))
                less = (ci < cj) | ((ci == cj) & (vi < vj))
                swap = ti.select(ascending, greater, less)
                if swap:
                    codes[i] = cj
                    codes[ixj] = ci
                    values[i] = vj
                    values[ixj] = vi

    def build(self):
        n = len(self.primitives)
        if n == 0:
            return

        scene_min = [float("inf")] * 3
        scene_max = [float("-inf")] * 3
        centroids = []
        for idx in range(n):
            mn, mx = self._aabb_of(idx)
            cx = 0.5 * (float(mn[0]) + float(mx[0]))
            cy = 0.5 * (float(mn[1]) + float(mx[1]))
            cz = 0.5 * (float(mn[2]) + float(mx[2]))
            centroids.append((cx, cy, cz))
            for a in range(3):
                scene_min[a] = min(scene_min[a], float(mn[a]))
                scene_max[a] = max(scene_max[a], float(mx[a]))

        extent = [max(scene_max[a] - scene_min[a], 1e-10) for a in range(3)]
        mins = ti.Vector.field(3, dtype=ti.f32, shape=n)
        maxs = ti.Vector.field(3, dtype=ti.f32, shape=n)
        sort_count = 1
        while sort_count < max(n, 1):
            sort_count *= 2
        codes = ti.field(dtype=ti.u32, shape=sort_count)
        values = ti.field(dtype=ti.i32, shape=sort_count)
        for idx in range(n):
            mn, mx = self._aabb_of(idx)
            mins[idx] = mn
            maxs[idx] = mx

        self._compute_morton_codes(
            mins,
            maxs,
            codes,
            values,
            sort_count,
            n,
            vec3(scene_min[0], scene_min[1], scene_min[2]),
            vec3(1.0 / extent[0], 1.0 / extent[1], 1.0 / extent[2]),
        )
        k = 2
        while k <= sort_count:
            j = k // 2
            while j > 0:
                self._bitonic_sort_step(codes, values, sort_count, j, k)
                j //= 2
            k *= 2
        ti.sync()
        sorted_indices = [int(i) for i in values.to_numpy()[:n]]

        self.ordered_prim_indices = [
            self._prim_storage_index(idx) for idx in sorted_indices
        ]
        self._build_morton_heap(sorted_indices)

    def _build_morton_heap(self, sorted_indices):
        n = len(sorted_indices)
        leaf_count = max(1, (n + self.MAX_LEAF_SIZE - 1) // self.MAX_LEAF_SIZE)
        base = 1
        while base < leaf_count:
            base *= 2
        self.nodes = [None] * (2 * base)

        empty_min = vec3(1e30, 1e30, 1e30)
        empty_max = vec3(-1e30, -1e30, -1e30)
        for leaf in range(base):
            node = PythonBVHNode()
            start = leaf * self.MAX_LEAF_SIZE
            end = min(start + self.MAX_LEAF_SIZE, n)
            node.is_leaf = True
            node.first_prim = start
            node.prim_count = max(0, end - start)
            if start < end:
                mn, mx = self._aabb_of(sorted_indices[start])
                node.aabb_min = vec3(mn[0], mn[1], mn[2])
                node.aabb_max = vec3(mx[0], mx[1], mx[2])
                for k in range(start + 1, end):
                    kmn, kmx = self._aabb_of(sorted_indices[k])
                    node.aabb_min, node.aabb_max = _merge_aabb(
                        node.aabb_min, node.aabb_max, kmn, kmx
                    )
            else:
                node.aabb_min = empty_min
                node.aabb_max = empty_max
            self.nodes[base + leaf] = node

        for idx in range(base - 1, 0, -1):
            left = self.nodes[2 * idx]
            right = self.nodes[2 * idx + 1]
            node = PythonBVHNode()
            node.left = left
            node.right = right
            node.is_leaf = False
            node.first_prim = 0
            node.prim_count = 0
            node.aabb_min, node.aabb_max = _merge_aabb(
                left.aabb_min, left.aabb_max, right.aabb_min, right.aabb_max
            )
            self.nodes[idx] = node
        self.root = self.nodes[1]

    def _convert_to_heap(self, flat_nodes, internal_count, leaf_count):
        max_depth = self._tree_depth(flat_nodes, 0, internal_count)
        heap_size = (1 << (max_depth + 1))
        self.nodes = [None] * (heap_size + 1)
        self._assign_heap(flat_nodes, 0, 1, internal_count)

    def _tree_depth(self, flat_nodes, idx, internal_count):
        if idx is None or idx < 0:
            return 0
        node = flat_nodes[idx] if idx < len(flat_nodes) else None
        if node is None:
            return 0
        if node.is_leaf:
            return 0
        ld = self._tree_depth(flat_nodes, node.left, internal_count)
        rd = self._tree_depth(flat_nodes, node.right, internal_count)
        return 1 + max(ld, rd)

    def _assign_heap(self, flat_nodes, flat_idx, heap_idx, internal_count):
        if flat_idx is None or flat_idx < 0:
            return
        node = flat_nodes[flat_idx] if flat_idx < len(flat_nodes) else None
        if node is None:
            return
        if heap_idx >= len(self.nodes):
            return
        self.nodes[heap_idx] = node
        if not node.is_leaf:
            self._assign_heap(flat_nodes, node.left, 2 * heap_idx, internal_count)
            self._assign_heap(flat_nodes, node.right, 2 * heap_idx + 1, internal_count)

    def get_node_count(self):
        return sum(1 for n in self.nodes if n is not None)

    def get_max_index(self):
        for i in range(len(self.nodes) - 1, 0, -1):
            if self.nodes[i] is not None:
                return i
        return 0
