"""Mesh containers and mesh-loading helpers for the solver."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Iterable

import jax
import numpy as np


@dataclasses.dataclass(frozen=True, slots=True)
class Mesh:
    """Store the mesh arrays used by the solver.

    Attributes:
        nodes: Node coordinates with shape `(n_nodes, 2)`.
        cells: Cell-to-node connectivity as a list of node-index arrays.
        face_nodes: Node indices for each face with shape `(n_faces, 2)`.
        legacy_mesh2d: Original legacy mesh object when the mesh was converted
            from `compressible_2d_`.
        cell_centroids: Cell centroid coordinates with shape `(n_cells, 2)`.
        cell_areas: Cell measures with shape `(n_cells,)`.
        cell_r: Radial coordinate used for axisymmetric weighting per cell.
        face_left: Left-cell index for each face.
        face_right: Right-cell index for each face, or `-1` on boundaries.
        face_normals: Unit normals pointing from left to right cell.
        face_areas: Face measures with shape `(n_faces,)`.
        face_centroids: Face centroid coordinates with shape `(n_faces, 2)`.
        face_r: Radial coordinate used for axisymmetric weighting per face.
        boundary_tags: Boundary tag for each face, or `-1` for interior faces.
        muscl_ll: Far-left stencil index used by MUSCL reconstruction.
        muscl_rr: Far-right stencil index used by MUSCL reconstruction.
        axisymmetric: Whether geometry weighting should use the radial
            coordinates.
    """

    # Construction-time / plotting metadata (not used by solver kernels)
    nodes: np.ndarray  # (n_nodes, 2)
    cells: list[np.ndarray]
    face_nodes: np.ndarray  # (n_faces, 2)
    legacy_mesh2d: object | None  # original compressible_2d_ mesh when available

    # Cell data
    cell_centroids: np.ndarray  # (n_cells, 2)
    cell_areas: np.ndarray  # (n_cells,)
    cell_r: np.ndarray  # (n_cells,)  radial pos; 1.0 for Cartesian

    # Face data
    face_left: np.ndarray  # (n_faces,) int
    face_right: np.ndarray  # (n_faces,) int  (-1 = boundary)
    face_normals: np.ndarray  # (n_faces, 2) unit outward normals
    face_areas: np.ndarray  # (n_faces,)
    face_centroids: np.ndarray  # (n_faces, 2)
    face_r: np.ndarray  # (n_faces,)  radial pos; 1.0 for Cartesian
    boundary_tags: np.ndarray  # (n_faces,) int  (-1 = interior)

    # MUSCL stencil (-1 where unavailable)
    muscl_ll: np.ndarray  # (n_faces,) "far-left" cell index
    muscl_rr: np.ndarray  # (n_faces,) "far-right" cell index
    axisymmetric: bool = False

    @classmethod
    def from_1d_grid(
        cls,
        x_coords: np.ndarray,
        bc_left_tag: int = 1,
        bc_right_tag: int = 2,
        periodic: bool = False,
    ) -> "Mesh":
        """Build a degenerate 2D mesh for a 1D structured grid.

        Args:
            x_coords: Node positions of length n_cells + 1.
            bc_left_tag: Physical tag assigned to the left boundary face.
            bc_right_tag: Physical tag assigned to the right boundary face.
            periodic: If True, wrap connectivity so face 0 connects to the
                last cell and the last face connects back to cell 0.
                Periodic meshes have no boundary faces (face_right >= 0 always).

        Returns:
            Mesh representing the 1D grid in the solver's unified mesh format.
        """
        x_coords = np.asarray(x_coords, dtype=float)
        n_cells = len(x_coords) - 1
        n_faces = n_cells + 1 if not periodic else n_cells

        nodes = np.column_stack([x_coords, np.zeros(len(x_coords))])

        dx = np.diff(x_coords)
        cell_centroids = np.column_stack(
            [0.5 * (x_coords[:-1] + x_coords[1:]), np.zeros(n_cells)]
        )
        cell_areas = dx
        cell_r = np.ones(n_cells)

        if not periodic:
            # Interfaces: x_coords[0], x_coords[1], ..., x_coords[n_cells]
            face_x = x_coords
        else:
            # Periodic: only n_cells faces (one per cell edge, no duplicates)
            face_x = x_coords[:-1]  # face i is between cell i-1 and cell i

        face_centroids = np.column_stack([face_x, np.zeros(n_faces)])
        face_areas = np.ones(n_faces)
        face_r = np.ones(n_faces)

        # Convention: outward normal from face_left cell toward face_right.
        # For 1D: all faces point in +x direction except the left boundary face
        # which points in -x (outward from cell 0).
        face_normals = np.zeros((n_faces, 2))
        if not periodic:
            face_normals[0, 0] = -1.0  # left BC: outward = -x from cell 0
            face_normals[1:, 0] = 1.0  # all others: outward = +x
        else:
            face_normals[:, 0] = 1.0  # all: +x direction

        if not periodic:
            # face 0: left BC (face_left=0, face_right=-1)
            # face i (1..n_cells-1): interior (face_left=i-1, face_right=i)
            # face n_cells: right BC (face_left=n_cells-1, face_right=-1)
            face_left = np.empty(n_faces, dtype=np.int32)
            face_right = np.empty(n_faces, dtype=np.int32)
            face_left[0] = 0
            face_right[0] = -1
            face_left[1:n_cells] = np.arange(n_cells - 1, dtype=np.int32)
            face_right[1:n_cells] = np.arange(1, n_cells, dtype=np.int32)
            face_left[n_cells] = n_cells - 1
            face_right[n_cells] = -1
        else:
            # face i connects cell (i-1) % n_cells -> cell i
            face_left = (np.arange(n_faces, dtype=np.int32) - 1) % n_cells
            face_right = np.arange(n_faces, dtype=np.int32)

        # muscl_ll[f] = face_left[f] - 1 (or -1 at left boundary)
        # muscl_rr[f] = face_right[f] + 1 (or -1 at right boundary)
        muscl_ll = np.full(n_faces, -1, dtype=np.int32)
        muscl_rr = np.full(n_faces, -1, dtype=np.int32)

        if not periodic:
            # Interior faces (1 .. n_cells-1): full stencil
            for f in range(1, n_cells):
                ll = face_left[f] - 1
                rr = face_right[f] + 1
                if 0 <= ll < n_cells:
                    muscl_ll[f] = ll
                if 0 <= rr < n_cells:
                    muscl_rr[f] = rr
        else:
            # Periodic: full stencil wraps around
            muscl_ll = (np.arange(n_faces, dtype=np.int32) - 2) % n_cells
            muscl_rr = (np.arange(n_faces, dtype=np.int32) + 1) % n_cells

        boundary_tags = np.full(n_faces, -1, dtype=np.int32)
        if not periodic:
            boundary_tags[0] = bc_left_tag
            boundary_tags[n_cells] = bc_right_tag

        return cls(
            nodes=nodes,
            cells=[np.array([i, i + 1], dtype=np.int32) for i in range(n_cells)],
            face_nodes=np.full((n_faces, 2), -1, dtype=np.int32),
            legacy_mesh2d=None,
            cell_centroids=cell_centroids,
            cell_areas=cell_areas,
            cell_r=cell_r,
            face_left=face_left,
            face_right=face_right,
            face_normals=face_normals,
            face_areas=face_areas,
            face_centroids=face_centroids,
            face_r=face_r,
            boundary_tags=boundary_tags,
            muscl_ll=muscl_ll,
            muscl_rr=muscl_rr,
            axisymmetric=False,
        )

    @classmethod
    def from_cells(
        cls,
        nodes: np.ndarray,
        cells: Iterable[Iterable[int]],
        boundary_edges: Iterable[tuple[int, int, int]] | None = None,
    ) -> "Mesh":
        """Build an unstructured 2D mesh from cell connectivity.

        Args:
            nodes: Node coordinates (n_nodes, 2).
            cells: Iterable of node-index lists defining each cell.
            boundary_edges: Iterable of (n1, n2, tag) for boundary edges.

        Returns:
            Mesh assembled from the provided polygonal cell connectivity.
        """
        nodes = np.asarray(nodes, dtype=float)
        cell_list = [np.asarray(c, dtype=int) for c in cells]

        n_cells = len(cell_list)
        cell_centroids = np.zeros((n_cells, 2))
        cell_areas = np.zeros((n_cells,))

        for i, c in enumerate(cell_list):
            pts = nodes[c]
            area, centroid = _polygon_area_centroid(pts)
            cell_centroids[i] = centroid
            cell_areas[i] = area

        face_map: dict[tuple[int, int], int] = {}
        face_nodes: list[tuple[int, int]] = []
        face_left_list: list[int] = []
        face_right_list: list[int] = []
        face_normals_list: list[np.ndarray] = []
        face_areas_list: list[float] = []
        face_centroids_list: list[np.ndarray] = []
        boundary_tags_list: list[int] = []

        edge_tag_map: dict[tuple[int, int], int] = {}
        if boundary_edges is not None:
            for n1, n2, tag in boundary_edges:
                key = (min(n1, n2), max(n1, n2))
                edge_tag_map[key] = tag

        for cell_idx, c in enumerate(cell_list):
            n = len(c)
            for k in range(n):
                n1 = c[k]
                n2 = c[(k + 1) % n]
                key = (min(n1, n2), max(n1, n2))
                if key not in face_map:
                    face_idx = len(face_nodes)
                    face_map[key] = face_idx
                    face_nodes.append((n1, n2))
                    face_left_list.append(cell_idx)
                    face_right_list.append(-1)

                    p1 = nodes[n1]
                    p2 = nodes[n2]
                    edge = p2 - p1
                    length = float(np.linalg.norm(edge))
                    if length < 1e-14:
                        normal = np.array([0.0, 0.0])
                    else:
                        n_cand = np.array([edge[1], -edge[0]]) / length
                        face_center = 0.5 * (p1 + p2)
                        to_cell = cell_centroids[cell_idx] - face_center
                        if np.dot(n_cand, to_cell) > 0.0:
                            n_cand = -n_cand
                        normal = n_cand

                    face_normals_list.append(normal)
                    face_areas_list.append(length)
                    face_centroids_list.append(0.5 * (p1 + p2))
                    boundary_tags_list.append(edge_tag_map.get(key, -1))
                else:
                    face_idx = face_map[key]
                    face_right_list[face_idx] = cell_idx

        n_faces = len(face_nodes)
        face_nodes_arr = np.asarray(face_nodes, dtype=np.int32)
        face_left_arr = np.asarray(face_left_list, dtype=np.int32)
        face_right_arr = np.asarray(face_right_list, dtype=np.int32)
        face_normals_arr = np.asarray(face_normals_list, dtype=float)
        face_areas_arr = np.asarray(face_areas_list, dtype=float)
        face_centroids_arr = np.asarray(face_centroids_list, dtype=float)
        boundary_tags_arr = np.asarray(boundary_tags_list, dtype=np.int32)

        cell_r = cell_centroids[:, 0].copy()
        face_r = face_centroids_arr[:, 0].copy()

        # Unstructured meshes do not support MUSCL reconstruction.
        muscl_ll = np.full(n_faces, -1, dtype=np.int32)
        muscl_rr = np.full(n_faces, -1, dtype=np.int32)

        return cls(
            nodes=nodes,
            cells=cell_list,
            face_nodes=face_nodes_arr,
            legacy_mesh2d=None,
            cell_centroids=cell_centroids,
            cell_areas=cell_areas,
            cell_r=cell_r,
            face_left=face_left_arr,
            face_right=face_right_arr,
            face_normals=face_normals_arr,
            face_areas=face_areas_arr,
            face_centroids=face_centroids_arr,
            face_r=face_r,
            boundary_tags=boundary_tags_arr,
            muscl_ll=muscl_ll,
            muscl_rr=muscl_rr,
            axisymmetric=False,
        )

    @classmethod
    def from_gmsh(cls, path: str | Path) -> "Mesh":
        """Read a Gmsh `.msh` file and return a mesh.

        Args:
            path: Path to the Gmsh mesh file.

        Returns:
            Mesh loaded from the file.
        """
        return _read_gmsh(path)

    @classmethod
    def from_gmsh_wedge(
        cls,
        path: str | Path,
        wedge_plane_tag: int = 4,
        remap_tags: dict[int, int] | None = None,
        axis_tag: int | None = None,
        axis_tol: float = 1e-10,
    ) -> "Mesh":
        """Read a thin-wedge 3D Gmsh v2 mesh and extract the 2D cross-section.

        Args:
            path: Path to the Gmsh mesh file.
            wedge_plane_tag: Physical tag identifying the wedge plane to
                extract.
            remap_tags: Optional mapping applied to the extracted boundary tags.
            axis_tag: Optional physical tag identifying the symmetry axis.
            axis_tol: Tolerance used when detecting axis-aligned entities.

        Returns:
            Mesh representing the extracted 2D wedge plane.
        """
        from compressible_2d_.mesh_gmsh import read_gmsh_v2_wedge_plane

        mesh2d = read_gmsh_v2_wedge_plane(
            path,
            wedge_plane_tag=wedge_plane_tag,
            remap_tags=remap_tags,
            axis_tag=axis_tag,
            axis_tol=axis_tol,
        )
        return _mesh2d_to_mesh(mesh2d)


def _mesh2d_to_mesh(m: object) -> Mesh:
    """Convert a legacy 2D mesh to the unified mesh format."""
    n_faces = m.face_left.shape[0]
    muscl_ll = np.full(n_faces, -1, dtype=np.int32)
    muscl_rr = np.full(n_faces, -1, dtype=np.int32)
    return Mesh(
        nodes=m.nodes,
        cells=[np.asarray(c, dtype=np.int32) for c in m.cells],
        face_nodes=m.face_nodes.astype(np.int32),
        legacy_mesh2d=m,
        cell_centroids=m.cell_centroids,
        cell_areas=m.cell_areas,
        cell_r=m.cell_r,
        face_left=m.face_left.astype(np.int32),
        face_right=m.face_right.astype(np.int32),
        face_normals=m.face_normals,
        face_areas=m.face_areas,
        face_centroids=m.face_centroids,
        face_r=m.face_r,
        boundary_tags=m.boundary_tags.astype(np.int32),
        muscl_ll=muscl_ll,
        muscl_rr=muscl_rr,
        axisymmetric=getattr(m, "axisymmetric", False),
    )


def _polygon_area_centroid(points: np.ndarray) -> tuple[float, np.ndarray]:
    """Compute the area and centroid of a polygon."""
    x = points[:, 0]
    y = points[:, 1]
    shift_x = np.roll(x, -1)
    shift_y = np.roll(y, -1)
    cross = x * shift_y - shift_x * y
    area = 0.5 * np.sum(cross)
    if np.abs(area) < 1e-14:
        centroid = np.mean(points, axis=0)
        return 0.0, centroid
    cx = np.sum((x + shift_x) * cross) / (6.0 * area)
    cy = np.sum((y + shift_y) * cross) / (6.0 * area)
    return np.abs(area), np.array([cx, cy])


def _read_gmsh(path: str | Path) -> Mesh:
    """Read a Gmsh file into the unified mesh format."""
    from compressible_2d_.mesh_gmsh import read_gmsh

    mesh2d = read_gmsh(path)
    return _mesh2d_to_mesh(mesh2d)


def _mesh_flatten(mesh: Mesh) -> tuple[list[np.ndarray], bool]:
    """Flatten a mesh for JAX pytree registration."""
    leaves = [
        mesh.nodes,
        mesh.cell_centroids,
        mesh.cell_areas,
        mesh.cell_r,
        mesh.face_left,
        mesh.face_right,
        mesh.face_normals,
        mesh.face_areas,
        mesh.face_centroids,
        mesh.face_r,
        mesh.boundary_tags,
        mesh.muscl_ll,
        mesh.muscl_rr,
    ]
    return leaves, mesh.axisymmetric


def _mesh_unflatten(axisymmetric: bool, leaves: list[np.ndarray]) -> Mesh:
    """Rebuild a mesh from JAX pytree leaves."""
    (
        nodes,
        cell_centroids,
        cell_areas,
        cell_r,
        face_left,
        face_right,
        face_normals,
        face_areas,
        face_centroids,
        face_r,
        boundary_tags,
        muscl_ll,
        muscl_rr,
    ) = leaves
    return Mesh(
        nodes=nodes,
        cells=[],
        face_nodes=np.zeros((0, 2), dtype=np.int32),
        legacy_mesh2d=None,
        cell_centroids=cell_centroids,
        cell_areas=cell_areas,
        cell_r=cell_r,
        face_left=face_left,
        face_right=face_right,
        face_normals=face_normals,
        face_areas=face_areas,
        face_centroids=face_centroids,
        face_r=face_r,
        boundary_tags=boundary_tags,
        muscl_ll=muscl_ll,
        muscl_rr=muscl_rr,
        axisymmetric=axisymmetric,
    )


jax.tree_util.register_pytree_node(Mesh, _mesh_flatten, _mesh_unflatten)
