import taichi as ti

from interval import Interval
from Ray import Ray

vec3 = ti.math.vec3


# =============================================================================
# Taichi Data Structures for GPU Traversal
# =============================================================================


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
    """
    Test if a ray intersects an AABB within the given t range.

    Uses the slab method optimized for GPU performance.

    Args:
        aabb_min: Minimum corner of the axis-aligned bounding box.
        aabb_max: Maximum corner of the axis-aligned bounding box.
        r: Ray to test for intersection.
        t_min: Minimum valid t parameter for intersection.
        t_max: Maximum valid t parameter for intersection.

    Returns:
        1 if ray intersects AABB within [t_min, t_max], 0 otherwise.
    """
    hit = 1
    for axis in ti.static(range(3)):
        inv_d = 1.0 / r.direction[axis]
        t0 = (aabb_min[axis] - r.origin[axis]) * inv_d
        t1 = (aabb_max[axis] - r.origin[axis]) * inv_d
        if inv_d < 0.0:
            t0, t1 = t1, t0
        t_min = ti.max(t0, t_min)
        t_max = ti.min(t1, t_max)
        if t_max <= t_min:
            hit = 0
    return hit


# =============================================================================
# Python BVH Builder (runs at scene construction time)
# =============================================================================


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
    """

    MAX_LEAF_SIZE = 4  # Maximum primitives per leaf node

    def __init__(self, spheres_data):
        """
        Initialize BVH builder with sphere data.

        Parameters
        ----------
        spheres_data : list of tuples
            Each tuple: (center_x, center_y, center_z, radius, sphere_index, aabb_min, aabb_max)
        """
        self.primitives = spheres_data
        self.ordered_prim_indices = []
        self.nodes = []  # Flat list of nodes in heap order
        self.root = None

    def build(self):
        """Build the BVH tree and convert to heap array."""
        if len(self.primitives) == 0:
            return

        # Build the tree recursively
        indices = list(range(len(self.primitives)))
        self.root = self._build_recursive(indices)

        # Convert to heap-indexed array
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
            prim = self.primitives[idx]
            aabb_min = prim[5]  # aabb_min
            aabb_max = prim[6]  # aabb_max

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
        prim = self.primitives[idx]
        aabb_min = prim[5]
        aabb_max = prim[6]
        return vec3(
            (aabb_min[0] + aabb_max[0]) * 0.5,
            (aabb_min[1] + aabb_max[1]) * 0.5,
            (aabb_min[2] + aabb_max[2]) * 0.5,
        )

    def _build_recursive(self, indices):
        """
        Recursively build the BVH tree using median split strategy.

        Args:
            indices: List of primitive indices to partition.

        Returns:
            PythonBVHNode representing the root of this subtree.
        """
        node = PythonBVHNode()
        node.aabb_min, node.aabb_max = self._compute_bounds(indices)

        n = len(indices)

        # Make a leaf if few enough primitives
        if n <= self.MAX_LEAF_SIZE:
            node.is_leaf = True
            node.first_prim = len(self.ordered_prim_indices)
            node.prim_count = n
            for idx in indices:
                self.ordered_prim_indices.append(
                    self.primitives[idx][4]
                )  # sphere_index
            return node

        # Find the axis with the largest extent
        extent = vec3(
            node.aabb_max[0] - node.aabb_min[0],
            node.aabb_max[1] - node.aabb_min[1],
            node.aabb_max[2] - node.aabb_min[2],
        )

        if extent[0] >= extent[1] and extent[0] >= extent[2]:
            axis = 0
        elif extent[1] >= extent[2]:
            axis = 1
        else:
            axis = 2

        # Sort by centroid along the chosen axis
        centroids = [(idx, self._compute_centroid(idx)[axis]) for idx in indices]
        centroids.sort(key=lambda x: x[1])
        sorted_indices = [c[0] for c in centroids]

        # Split at the median
        mid = n // 2
        left_indices = sorted_indices[:mid]
        right_indices = sorted_indices[mid:]

        # Recursively build children
        node.left = self._build_recursive(left_indices)
        node.right = self._build_recursive(right_indices)
        node.is_leaf = False

        return node

    def _convert_to_heap_array(self):
        """Convert the tree to a heap-indexed flat array (1-indexed)."""
        if self.root is None:
            return

        # Calculate maximum depth to determine array size
        max_depth = self._get_depth(self.root)
        max_nodes = (1 << (max_depth + 1)) - 1  # 2^(depth+1) - 1

        # Initialize nodes array with empty nodes
        self.nodes = [None] * (max_nodes + 1)  # 1-indexed

        # BFS to assign heap indices
        self._assign_heap_indices(self.root, 1)

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


# =============================================================================
# Taichi BVH Class for GPU Traversal
# =============================================================================


@ti.data_oriented
class BVH:
    """
    GPU-resident BVH for fast ray intersection queries.

    Uses iterative stack-based traversal (no recursion on GPU).
    """

    MAX_STACK_DEPTH = 64  # Maximum traversal stack depth

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

        # Extract sphere data for Python builder
        spheres_data = []
        for i in range(sphere_count):
            s = spheres_field[i]
            center = s.center.origin  # Ray origin is the center
            radius = s.radius
            aabb_min = vec3(s.aabb.x.min, s.aabb.y.min, s.aabb.z.min)
            aabb_max = vec3(s.aabb.x.max, s.aabb.y.max, s.aabb.z.max)
            spheres_data.append(
                (center[0], center[1], center[2], radius, i, aabb_min, aabb_max)
            )

        # Build BVH in Python
        builder = BVHBuilder(spheres_data)
        builder.build()

        # Allocate Taichi fields
        max_idx = max(builder.get_max_index(), 1)
        self.nodes = BVHNode.field(shape=(max_idx + 1,))
        self.node_count = max_idx

        # Ordered primitive indices
        n_prims = len(builder.ordered_prim_indices)
        self.prim_indices = ti.field(dtype=ti.i32, shape=(max(n_prims, 1),))
        self.prim_count = n_prims

        # Copy data to Taichi fields
        self._upload_to_gpu(builder)

        # Temporary hit record for traversal
        self.temp_rec = None  # Will be set by hit function

    def _upload_to_gpu(self, builder):
        """
        Upload BVH data to GPU fields.

        Args:
            builder: BVHBuilder instance containing the built tree.
        """
        # Upload nodes
        for i in range(1, len(builder.nodes)):
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
                # Empty node (should not be traversed)
                self.nodes[i] = BVHNode(
                    aabb_min=vec3(0, 0, 0),
                    aabb_max=vec3(0, 0, 0),
                    is_leaf=0,
                    first_prim=0,
                    prim_count=0,
                )

        # Upload primitive indices
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

        # Use a simple iterative approach with explicit loop unrolling control
        # Stack implemented as local variables with fixed max depth
        # Using ti.Vector for local array storage
        stack = ti.Vector([0] * 64, dt=ti.i32)
        stack_ptr = 0

        # Start at root (index 1)
        current = 1

        # Limit iterations to prevent infinite loops
        for _ in range(1000):  # Maximum iteration limit to prevent hangs
            if current > 0 and current <= self.node_count:
                node = self.nodes[current]

                # Check if ray hits this node's AABB
                if aabb_hit(node.aabb_min, node.aabb_max, r, ray_t.min, closest_so_far):
                    if node.is_leaf == 1:
                        # Test all primitives in this leaf
                        for i in range(node.prim_count):
                            prim_idx = self.prim_indices[node.first_prim + i]
                            if self.spheres[prim_idx].hit(
                                r, Interval(ray_t.min, closest_so_far), temp_rec
                            ):
                                did_hit = 1
                                closest_so_far = temp_rec[None].t
                                rec[x, y].t = temp_rec[None].t
                                rec[x, y].p = temp_rec[None].p
                                rec[x, y].normal = temp_rec[None].normal
                                rec[x, y].front_face = temp_rec[None].front_face
                                rec[x, y].mat_id = temp_rec[None].mat_id
                    else:
                        # Internal node: push children to stack
                        left_child = 2 * current
                        right_child = 2 * current + 1

                        # Push right first so left is processed first (LIFO)
                        if right_child <= self.node_count and stack_ptr < 63:
                            stack[stack_ptr] = right_child
                            stack_ptr += 1

                        if left_child <= self.node_count:
                            current = left_child
                            continue

            # Pop from stack
            if stack_ptr > 0:
                stack_ptr -= 1
                current = stack[stack_ptr]
            else:
                break

        return did_hit
