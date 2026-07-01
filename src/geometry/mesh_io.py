from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np
import taichi.math as tm

from tools import debug_tools

vec3 = tm.vec3
_TRIMESH_EXTENSIONS = {".glb", ".gltf", ".stl", ".ply", ".blend", ".dae", ".fbx"}
_DEFAULT_FLIP_WINDING = False


def set_default_flip_winding(enabled: bool) -> None:
    global _DEFAULT_FLIP_WINDING
    _DEFAULT_FLIP_WINDING = bool(enabled)


NORMAL_VALIDATION_DEG = 1.0
NORMAL_VALIDATION_COS = math.cos(math.radians(NORMAL_VALIDATION_DEG))
DEGENERATE_AREA_EPS = 1e-24


@dataclass(frozen=True)
class MeshTriangle:
    p0: vec3
    p1: vec3
    p2: vec3
    n0: vec3
    n1: vec3
    n2: vec3
    uv0: vec3
    uv1: vec3
    uv2: vec3
    material_id: int = 0
    geo_normal: vec3 = vec3(0.0, 0.0, 1.0)


@dataclass
class MeshLoadStats:
    source_path: str
    raw_face_count: int = 0
    triangulated_face_count: int = 0
    valid_face_count: int = 0
    degenerate_skipped: int = 0
    out_of_range_skipped: int = 0
    winding_corrections: int = 0
    flip_flag_applied: int = 0
    normal_validation_errors: int = 0
    bounds_min: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    bounds_max: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def log_summary(self) -> None:
        print(
            f"[mesh load] {os.path.basename(self.source_path)}: raw_faces={self.raw_face_count:,}, triangulated={self.triangulated_face_count:,}, loaded={self.valid_face_count:,}, degenerate_skipped={self.degenerate_skipped:,}, out_of_range_skipped={self.out_of_range_skipped:,}, winding_corrections={self.winding_corrections:,}, flip_flag={self.flip_flag_applied:,}, normal_errors={self.normal_validation_errors:,}, bbox_min=({self.bounds_min[0]:.4f}, {self.bounds_min[1]:.4f}, {self.bounds_min[2]:.4f}), bbox_max=({self.bounds_max[0]:.4f}, {self.bounds_max[1]:.4f}, {self.bounds_max[2]:.4f})",
            file=sys.stderr,
        )


@dataclass
class MeshData:
    triangles: List[MeshTriangle]
    material_names: List[str]
    material_colors: List[Tuple[float, float, float]]
    source_path: str
    stats: MeshLoadStats = field(default_factory=lambda: MeshLoadStats(""))

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.triangles:
            z = np.zeros(3, dtype=np.float64)
            return (z, z)
        pts = []
        for tri in self.triangles:
            pts.extend((tri.p0, tri.p1, tri.p2))
        arr = np.asarray(
            [[float(p[0]), float(p[1]), float(p[2])] for p in pts], dtype=np.float64
        )
        return (arr.min(axis=0), arr.max(axis=0))


def select_best_model(models_dir: str, prefer_obj: bool = False) -> str:
    debug_tools.log(f"scan models in {models_dir}")
    files = [os.path.join(models_dir, f) for f in os.listdir(models_dir)]
    obj_files = [p for p in files if os.path.splitext(p)[1].lower() == ".obj"]
    if prefer_obj and obj_files:
        selected = max(obj_files, key=lambda p: os.path.getsize(p))
        debug_tools.log(
            f"selected OBJ model {selected} ({os.path.getsize(selected):,} bytes)"
        )
        return selected
    candidates = [
        p
        for p in files
        if os.path.splitext(p)[1].lower() in {".obj"} | _TRIMESH_EXTENSIONS
    ]
    if not candidates:
        raise FileNotFoundError(f"No supported model files found in {models_dir}")
    selected = max(
        candidates,
        key=lambda p: (os.path.splitext(p)[1].lower() != ".obj", os.path.getsize(p)),
    )
    debug_tools.log(f"selected model {selected} ({os.path.getsize(selected):,} bytes)")
    return selected


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(vectors, axis=1)
    good = length > 1e-20
    out = np.zeros_like(vectors)
    out[good] = vectors[good] / length[good, None]
    return out


def _transform_vertices(
    vertices: np.ndarray, scale: float, translate: Sequence[float]
) -> np.ndarray:
    t = np.asarray(translate, dtype=np.float64).reshape(3)
    return vertices * float(scale) + t


def _face_normal_ccw(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    return np.cross(p1 - p0, p2 - p0)


def _triangle_area_sq(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    fn = _face_normal_ccw(p0, p1, p2)
    return float(np.dot(fn, fn))


def fit_vertices_to_cornell(
    vertices: np.ndarray, target_height: float = 1.35
) -> np.ndarray:
    mn = vertices.min(axis=0)
    mx = vertices.max(axis=0)
    extent = np.maximum(mx - mn, 1e-12)
    scale = target_height / float(np.max(extent))
    fitted = (vertices - 0.5 * (mn + mx)) * scale
    fitted[:, 1] -= fitted[:, 1].min()
    return fitted


def _enforce_ccw_outward(
    face: List[Tuple[int, int, int]], pos_np: np.ndarray, mesh_centroid: np.ndarray
) -> Tuple[List[Tuple[int, int, int]], bool]:
    ps = [pos_np[c[0]] for c in face]
    fn = _face_normal_ccw(ps[0], ps[1], ps[2])
    fn_len = float(np.linalg.norm(fn))
    if fn_len < 1e-20:
        return (face, False)
    fn = fn / fn_len
    face_center = (ps[0] + ps[1] + ps[2]) / 3.0
    to_centroid = mesh_centroid - face_center
    flipped = False
    if float(np.dot(fn, to_centroid)) > 0.0:
        face = [face[0], face[2], face[1]]
        flipped = True
    return (face, flipped)


def load_obj_mesh(
    path: str,
    scale: float = 1.0,
    translate: Sequence[float] = (0.0, 0.0, 0.0),
    fit_to_cornell: bool = False,
    target_height: float = 1.35,
    flip_winding: bool = False,
) -> MeshData:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    stats = MeshLoadStats(source_path=path)
    debug_tools.log(f"begin load OBJ {path}")
    obj_t0 = debug_tools.now() if debug_tools.enabled() else 0.0
    positions: List[List[float]] = []
    normals: List[List[float]] = []
    uvs: List[List[float]] = []
    raw_faces: List[List[Tuple[int, int, int]]] = []
    faces: List[List[Tuple[int, int, int]]] = []
    parse_t0 = debug_tools.now() if debug_tools.enabled() else 0.0
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            tag = parts[0]
            if tag == "v" and len(parts) >= 4:
                try:
                    positions.append(
                        [float(parts[1]), float(parts[2]), float(parts[3])]
                    )
                except ValueError:
                    continue
            elif tag == "vn" and len(parts) >= 4:
                try:
                    normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
                except ValueError:
                    continue
            elif tag == "vt" and len(parts) >= 3:
                try:
                    uvs.append([float(parts[1]), float(parts[2]), 0.0])
                except ValueError:
                    continue
            elif tag == "f" and len(parts) >= 4:
                corners: List[Tuple[int, int, int]] = []
                for token in parts[1:]:
                    vals = token.split("/")
                    if not vals or not vals[0]:
                        continue
                    try:
                        vi = int(vals[0])
                        ti_idx = int(vals[1]) if len(vals) > 1 and vals[1] else 0
                        ni = int(vals[2]) if len(vals) > 2 and vals[2] else 0
                    except ValueError:
                        continue
                    vi = len(positions) + vi if vi < 0 else vi - 1
                    ti_idx = len(uvs) + ti_idx if ti_idx < 0 else ti_idx - 1
                    ni = len(normals) + ni if ni < 0 else ni - 1
                    corners.append((vi, ti_idx, ni))
                if len(corners) >= 3:
                    stats.raw_face_count += 1
                    raw_faces.append(list(corners))
                    for k in range(1, len(corners) - 1):
                        faces.append([corners[0], corners[k], corners[k + 1]])
    stats.triangulated_face_count = len(faces)
    if debug_tools.enabled():
        debug_tools.log(
            f"parsed OBJ text in {debug_tools.now() - parse_t0:.3f}s: {len(positions):,} vertices, {len(normals):,} normals, {len(uvs):,} uvs, {stats.triangulated_face_count:,} triangulated faces"
        )
    if not positions:
        stats.log_summary()
        return MeshData([], ["default"], [(0.72, 0.7, 0.66)], path, stats)
    normalize_t0 = debug_tools.now() if debug_tools.enabled() else 0.0
    pos_np = np.asarray(positions, dtype=np.float64)
    pos_np = (
        fit_vertices_to_cornell(pos_np, target_height)
        if fit_to_cornell
        else _transform_vertices(pos_np, scale, translate)
    )
    mesh_centroid = pos_np.mean(axis=0)
    valid_faces: List[List[Tuple[int, int, int]]] = []
    for face in faces:
        if not all((0 <= corner[0] < len(pos_np) for corner in face)):
            stats.out_of_range_skipped += 1
            continue
        ps = [pos_np[c[0]] for c in face]
        if _triangle_area_sq(ps[0], ps[1], ps[2]) <= DEGENERATE_AREA_EPS:
            stats.degenerate_skipped += 1
            continue
        face, corrected = _enforce_ccw_outward(face, pos_np, mesh_centroid)
        if corrected:
            stats.winding_corrections += 1
        if flip_winding:
            face = [face[0], face[2], face[1]]
            stats.flip_flag_applied += 1
        ps = [pos_np[c[0]] for c in face]
        if _triangle_area_sq(ps[0], ps[1], ps[2]) <= DEGENERATE_AREA_EPS:
            stats.degenerate_skipped += 1
            continue
        valid_faces.append(face)
    faces = valid_faces
    stats.valid_face_count = len(faces)
    norm_np = (
        _normalize_vectors(np.asarray(normals, dtype=np.float64))
        if normals
        else np.zeros((0, 3), dtype=np.float64)
    )
    needs_generated = not normals or any(
        (
            corner[2] < 0 or corner[2] >= len(norm_np)
            for face in faces
            for corner in face
        )
    )
    generated = np.zeros_like(pos_np)
    if needs_generated:
        accum = np.zeros_like(pos_np)
        for face in faces:
            i0, i1, i2 = (face[0][0], face[1][0], face[2][0])
            fn = _face_normal_ccw(pos_np[i0], pos_np[i1], pos_np[i2])
            for idx in (i0, i1, i2):
                accum[idx] += fn
        generated = _normalize_vectors(accum)
    if debug_tools.enabled():
        debug_tools.log(
            f"normalized/validated OBJ in {debug_tools.now() - normalize_t0:.3f}s: {stats.valid_face_count:,} valid faces"
        )
    uv_np = (
        np.asarray(uvs, dtype=np.float64) if uvs else np.zeros((0, 3), dtype=np.float64)
    )
    material_names = ["default"]
    material_colors = [(0.72, 0.7, 0.66)]
    triangles_t0 = debug_tools.now() if debug_tools.enabled() else 0.0
    n_faces = len(faces)
    face_vi = np.empty((n_faces, 3), dtype=np.int64)
    face_ni = np.empty((n_faces, 3), dtype=np.int64)
    face_ti = np.empty((n_faces, 3), dtype=np.int64)
    for fi, face in enumerate(faces):
        for ci in range(3):
            face_vi[fi, ci] = face[ci][0]
            face_ti[fi, ci] = face[ci][1]
            face_ni[fi, ci] = face[ci][2]
    tri_p = pos_np[face_vi]
    tri_n = np.zeros((n_faces, 3, 3), dtype=np.float64)
    has_file_normals = len(norm_np) > 0
    for ci in range(3):
        valid_mask = (face_ni[:, ci] >= 0) & (face_ni[:, ci] < len(norm_np))
        if has_file_normals and np.any(valid_mask):
            tri_n[valid_mask, ci] = norm_np[face_ni[valid_mask, ci]]
        missing = ~valid_mask
        if np.any(missing):
            tri_n[missing, ci] = generated[face_vi[missing, ci]]
    tri_e1 = tri_p[:, 1] - tri_p[:, 0]
    tri_e2 = tri_p[:, 2] - tri_p[:, 0]
    face_geom_n = np.cross(tri_e1, tri_e2)
    face_geom_len = np.linalg.norm(face_geom_n, axis=1, keepdims=True)
    face_geom_n = np.where(face_geom_len > 1e-20, face_geom_n / face_geom_len, 0.0)
    for ci in range(3):
        norms_sq = np.sum(tri_n[:, ci] ** 2, axis=1)
        zero_mask = norms_sq <= 1e-20
        if np.any(zero_mask):
            tri_n[zero_mask, ci] = face_geom_n[zero_mask]
    for fi in range(n_faces):
        gn = face_geom_n[fi]
        for ci in range(3):
            vn = tri_n[fi, ci]
            vn_len = float(np.linalg.norm(vn))
            if vn_len < 1e-20:
                continue
            vn = vn / vn_len
            if float(np.dot(vn, gn)) < NORMAL_VALIDATION_COS:
                stats.normal_validation_errors += 1
                tri_n[fi, ci] = gn
    has_uvs = len(uv_np) > 0
    tri_uv = np.zeros((n_faces, 3, 3), dtype=np.float64)
    if has_uvs:
        for ci in range(3):
            valid_mask = (face_ti[:, ci] >= 0) & (face_ti[:, ci] < len(uv_np))
            if np.any(valid_mask):
                tri_uv[valid_mask, ci] = uv_np[face_ti[valid_mask, ci]]
    triangles: List[MeshTriangle] = []
    for fi in range(n_faces):
        gn = face_geom_n[fi]
        triangles.append(
            MeshTriangle(
                vec3(
                    float(tri_p[fi, 0, 0]),
                    float(tri_p[fi, 0, 1]),
                    float(tri_p[fi, 0, 2]),
                ),
                vec3(
                    float(tri_p[fi, 1, 0]),
                    float(tri_p[fi, 1, 1]),
                    float(tri_p[fi, 1, 2]),
                ),
                vec3(
                    float(tri_p[fi, 2, 0]),
                    float(tri_p[fi, 2, 1]),
                    float(tri_p[fi, 2, 2]),
                ),
                vec3(
                    float(tri_n[fi, 0, 0]),
                    float(tri_n[fi, 0, 1]),
                    float(tri_n[fi, 0, 2]),
                ),
                vec3(
                    float(tri_n[fi, 1, 0]),
                    float(tri_n[fi, 1, 1]),
                    float(tri_n[fi, 1, 2]),
                ),
                vec3(
                    float(tri_n[fi, 2, 0]),
                    float(tri_n[fi, 2, 1]),
                    float(tri_n[fi, 2, 2]),
                ),
                vec3(float(tri_uv[fi, 0, 0]), float(tri_uv[fi, 0, 1]), 0.0),
                vec3(float(tri_uv[fi, 1, 0]), float(tri_uv[fi, 1, 1]), 0.0),
                vec3(float(tri_uv[fi, 2, 0]), float(tri_uv[fi, 2, 1]), 0.0),
                material_id=0,
                geo_normal=vec3(float(gn[0]), float(gn[1]), float(gn[2])),
            )
        )
    if n_faces > 0:
        all_pts = tri_p.reshape(-1, 3)
        bmin = all_pts.min(axis=0)
        bmax = all_pts.max(axis=0)
        stats.bounds_min = (float(bmin[0]), float(bmin[1]), float(bmin[2]))
        stats.bounds_max = (float(bmax[0]), float(bmax[1]), float(bmax[2]))
    if (
        stats.valid_face_count
        != stats.triangulated_face_count
        - stats.degenerate_skipped
        - stats.out_of_range_skipped
    ):
        raise RuntimeError(
            f"Triangle count mismatch for {path}: loaded={stats.valid_face_count}, expected={stats.triangulated_face_count - stats.degenerate_skipped - stats.out_of_range_skipped}"
        )
    stats.log_summary()
    if debug_tools.enabled():
        debug_tools.log(
            f"built MeshTriangle list in {debug_tools.now() - triangles_t0:.3f}s: {len(triangles):,} triangles"
        )
        debug_tools.log(f"end load OBJ ({debug_tools.now() - obj_t0:.3f}s)")
    return MeshData(triangles, material_names, material_colors, path, stats)


def _load_trimesh_mesh(
    path: str,
    fit_to_cornell: bool = False,
    target_height: float = 1.35,
    flip_winding: bool = False,
) -> MeshData:
    try:
        import trimesh
    except ImportError as exc:
        raise ImportError("trimesh is required for non-OBJ mesh formats") from exc
    stats = MeshLoadStats(source_path=path)
    loaded = trimesh.load(path, force="scene")
    geometries = (
        list(loaded.geometry.values()) if hasattr(loaded, "geometry") else [loaded]
    )
    all_vertices = np.concatenate(
        [np.asarray(g.vertices, dtype=np.float64) for g in geometries], axis=0
    )
    fitted_all = (
        fit_vertices_to_cornell(all_vertices, target_height)
        if fit_to_cornell
        else all_vertices
    )
    mesh_centroid = fitted_all.mean(axis=0)
    offset = 0
    triangles: List[MeshTriangle] = []
    for g in geometries:
        count = len(g.vertices)
        verts = fitted_all[offset : offset + count]
        offset += count
        normals = np.asarray(
            getattr(g, "vertex_normals", np.zeros_like(verts)), dtype=np.float64
        )
        if normals.shape != verts.shape or np.linalg.norm(normals) < 1e-12:
            normals = _normalize_vectors(np.zeros_like(verts))
        normals = _normalize_vectors(normals)
        uv = np.zeros((count, 3), dtype=np.float64)
        visual_uv = getattr(getattr(g, "visual", None), "uv", None)
        if visual_uv is not None and len(visual_uv) == count:
            uv[:, :2] = np.asarray(visual_uv, dtype=np.float64)[:, :2]
        for face in np.asarray(g.faces, dtype=np.int64):
            stats.triangulated_face_count += 1
            idx = [int(face[0]), int(face[1]), int(face[2])]
            ps = [verts[i] for i in idx]
            if _triangle_area_sq(ps[0], ps[1], ps[2]) <= DEGENERATE_AREA_EPS:
                stats.degenerate_skipped += 1
                continue
            fn = _face_normal_ccw(ps[0], ps[1], ps[2])
            fn_len = float(np.linalg.norm(fn))
            if fn_len < 1e-20:
                stats.degenerate_skipped += 1
                continue
            fn = fn / fn_len
            fc = (ps[0] + ps[1] + ps[2]) / 3.0
            if float(np.dot(fn, mesh_centroid - fc)) > 0.0:
                idx = [idx[0], idx[2], idx[1]]
                ps = [verts[i] for i in idx]
                fn = _face_normal_ccw(ps[0], ps[1], ps[2])
                fn = fn / max(float(np.linalg.norm(fn)), 1e-20)
                stats.winding_corrections += 1
            if flip_winding:
                idx = [idx[0], idx[2], idx[1]]
                ps = [verts[i] for i in idx]
                fn = -fn
                stats.flip_flag_applied += 1
            ns = [normals[i] for i in idx]
            for ni, n in enumerate(ns):
                n_arr = np.asarray(n, dtype=np.float64)
                n_len = float(np.linalg.norm(n_arr))
                if (
                    n_len > 1e-20
                    and float(np.dot(n_arr / n_len, fn)) < NORMAL_VALIDATION_COS
                ):
                    stats.normal_validation_errors += 1
                    ns[ni] = fn
            us = [uv[i] for i in idx]
            triangles.append(
                MeshTriangle(
                    *(vec3(float(p[0]), float(p[1]), float(p[2])) for p in ps),
                    *(vec3(float(n[0]), float(n[1]), float(n[2])) for n in ns),
                    *(vec3(float(u[0]), float(u[1]), 0.0) for u in us),
                    material_id=0,
                    geo_normal=vec3(float(fn[0]), float(fn[1]), float(fn[2])),
                )
            )
            stats.valid_face_count += 1
    if triangles:
        pts = np.asarray(
            [[float(t.p0[0]), float(t.p0[1]), float(t.p0[2])] for t in triangles]
            + [[float(t.p1[0]), float(t.p1[1]), float(t.p1[2])] for t in triangles]
            + [[float(t.p2[0]), float(t.p2[1]), float(t.p2[2])] for t in triangles],
            dtype=np.float64,
        )
        bmin = pts.min(axis=0)
        bmax = pts.max(axis=0)
        stats.bounds_min = (float(bmin[0]), float(bmin[1]), float(bmin[2]))
        stats.bounds_max = (float(bmax[0]), float(bmax[1]), float(bmax[2]))
    stats.log_summary()
    return MeshData(triangles, ["default"], [(0.72, 0.7, 0.66)], path, stats)


def load_mesh_auto(
    path: str,
    scale: float = 1.0,
    translate: Sequence[float] = (0.0, 0.0, 0.0),
    fit_to_cornell: bool = False,
    target_height: float = 1.35,
    flip_winding: bool = False,
) -> MeshData:
    if flip_winding is False:
        flip_winding = _DEFAULT_FLIP_WINDING
    ext = os.path.splitext(path)[1].lower()
    if ext == ".obj":
        return load_obj_mesh(
            path, scale, translate, fit_to_cornell, target_height, flip_winding
        )
    if ext in _TRIMESH_EXTENSIONS:
        return _load_trimesh_mesh(path, fit_to_cornell, target_height, flip_winding)
    return _load_trimesh_mesh(path, fit_to_cornell, target_height, flip_winding)


def load_obj_triangles(
    path: str, scale: float = 1.0, translate: Sequence[float] = (0.0, 0.0, 0.0)
):
    return [(t.p0, t.p1, t.p2) for t in load_obj_mesh(path, scale, translate).triangles]
