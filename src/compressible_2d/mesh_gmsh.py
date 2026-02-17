from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclasses.dataclass(frozen=True, slots=True)
class Mesh2D:
    nodes: np.ndarray  # (n_nodes, 2)
    cells: list[np.ndarray]  # list of node index arrays
    cell_centroids: np.ndarray  # (n_cells, 2)
    cell_areas: np.ndarray  # (n_cells,)
    face_nodes: np.ndarray  # (n_faces, 2)
    face_left: np.ndarray  # (n_faces,)
    face_right: np.ndarray  # (n_faces,)
    face_normals: np.ndarray  # (n_faces, 2) unit normals from left->right
    face_areas: np.ndarray  # (n_faces,) length in 2D
    face_centroids: np.ndarray  # (n_faces, 2)
    boundary_tags: np.ndarray  # (n_faces,) tag or -1
    cell_r: np.ndarray  # (n_cells,) radius coordinate (r)
    face_r: np.ndarray  # (n_faces,) radius coordinate (r)

    @staticmethod
    def _polygon_area_centroid(points: np.ndarray) -> tuple[float, np.ndarray]:
        # Shoelace formula for area and centroid
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

    @classmethod
    def from_cells(
        cls,
        nodes: np.ndarray,
        cells: Iterable[Iterable[int]],
        boundary_edges: Iterable[tuple[int, int, int]] | None = None,
    ) -> "Mesh2D":
        nodes = np.asarray(nodes, dtype=float)
        cell_list = [np.asarray(c, dtype=int) for c in cells]

        n_cells = len(cell_list)
        cell_centroids = np.zeros((n_cells, 2))
        cell_areas = np.zeros((n_cells,))

        for i, c in enumerate(cell_list):
            pts = nodes[c]
            area, centroid = cls._polygon_area_centroid(pts)
            cell_centroids[i] = centroid
            cell_areas[i] = area

        # Build face list
        face_map: dict[tuple[int, int], int] = {}
        face_nodes: list[tuple[int, int]] = []
        face_left: list[int] = []
        face_right: list[int] = []
        face_normals: list[np.ndarray] = []
        face_areas: list[float] = []
        face_centroids: list[np.ndarray] = []
        boundary_tags = []

        # Boundary edge tags lookup
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
                    face_left.append(cell_idx)
                    face_right.append(-1)

                    p1 = nodes[n1]
                    p2 = nodes[n2]
                    edge = p2 - p1
                    length = float(np.linalg.norm(edge))
                    if length < 1e-14:
                        normal = np.array([0.0, 0.0])
                    else:
                        # Candidate normal
                        n_cand = np.array([edge[1], -edge[0]]) / length
                        face_center = 0.5 * (p1 + p2)
                        to_cell = cell_centroids[cell_idx] - face_center
                        # If normal points inward, flip
                        if np.dot(n_cand, to_cell) > 0.0:
                            n_cand = -n_cand
                        normal = n_cand

                    face_normals.append(normal)
                    face_areas.append(length)
                    face_centroids.append(0.5 * (p1 + p2))
                    boundary_tags.append(edge_tag_map.get(key, -1))
                else:
                    face_idx = face_map[key]
                    face_right[face_idx] = cell_idx

        face_nodes_arr = np.asarray(face_nodes, dtype=int)
        face_left_arr = np.asarray(face_left, dtype=int)
        face_right_arr = np.asarray(face_right, dtype=int)
        face_normals_arr = np.asarray(face_normals, dtype=float)
        face_areas_arr = np.asarray(face_areas, dtype=float)
        face_centroids_arr = np.asarray(face_centroids, dtype=float)
        boundary_tags_arr = np.asarray(boundary_tags, dtype=int)

        cell_r = cell_centroids[:, 0].copy()
        face_r = face_centroids_arr[:, 0].copy()

        return cls(
            nodes=nodes,
            cells=cell_list,
            cell_centroids=cell_centroids,
            cell_areas=cell_areas,
            face_nodes=face_nodes_arr,
            face_left=face_left_arr,
            face_right=face_right_arr,
            face_normals=face_normals_arr,
            face_areas=face_areas_arr,
            face_centroids=face_centroids_arr,
            boundary_tags=boundary_tags_arr,
            cell_r=cell_r,
            face_r=face_r,
        )


def read_gmsh(path: str | Path) -> Mesh2D:
    """Read a Gmsh .msh file (ASCII v2 or v4) and return Mesh2D."""
    path = Path(path)
    text = path.read_text().splitlines()

    # Detect version
    if "$MeshFormat" not in text:
        raise ValueError("Invalid .msh: missing $MeshFormat")
    idx = text.index("$MeshFormat")
    version_line = text[idx + 1].split()
    version = version_line[0]

    if version.startswith("2."):
        return _read_gmsh_v2(text)
    if version.startswith("4."):
        return _read_gmsh_v4(text)

    raise ValueError(f"Unsupported Gmsh version: {version}")


def _read_gmsh_v2(lines: list[str]) -> Mesh2D:
    # Nodes
    node_start = lines.index("$Nodes") + 1
    n_nodes = int(lines[node_start])
    nodes = np.zeros((n_nodes, 2))
    for i in range(n_nodes):
        parts = lines[node_start + 1 + i].split()
        _, x, y, _z = parts[:4]
        nodes[i] = [float(x), float(y)]

    # Elements
    elem_start = lines.index("$Elements") + 1
    n_elem = int(lines[elem_start])

    cells: list[np.ndarray] = []
    boundary_edges: list[tuple[int, int, int]] = []

    for i in range(n_elem):
        parts = lines[elem_start + 1 + i].split()
        elem_type = int(parts[1])
        num_tags = int(parts[2])
        tags = [int(t) for t in parts[3 : 3 + num_tags]]
        node_ids = [int(n) - 1 for n in parts[3 + num_tags :]]
        physical_tag = tags[0] if tags else -1

        if elem_type == 1:  # line
            if len(node_ids) != 2:
                continue
            boundary_edges.append((node_ids[0], node_ids[1], physical_tag))
        elif elem_type in (2, 3):  # tri or quad
            cells.append(np.array(node_ids, dtype=int))
        else:
            # ignore other element types
            continue

    return Mesh2D.from_cells(nodes, cells, boundary_edges)


def _read_gmsh_v4(lines: list[str]) -> Mesh2D:
    # Nodes block
    node_start = lines.index("$Nodes") + 1
    header = [int(x) for x in lines[node_start].split()]
    num_entity_blocks, num_nodes = header[0], header[1]
    nodes = np.zeros((num_nodes, 2))

    current_line = node_start + 1
    node_tag_to_index: dict[int, int] = {}
    for _ in range(num_entity_blocks):
        ent_info = lines[current_line].split()
        current_line += 1
        num_nodes_in_block = int(ent_info[3])
        # node tags
        tags = []
        while len(tags) < num_nodes_in_block:
            tags.extend([int(x) for x in lines[current_line].split()])
            current_line += 1
        # coordinates
        for t in tags:
            coord = [float(x) for x in lines[current_line].split()[:3]]
            current_line += 1
            idx = len(node_tag_to_index)
            node_tag_to_index[t] = idx
            nodes[idx] = [coord[0], coord[1]]

    # Elements block
    elem_start = lines.index("$Elements") + 1
    header = [int(x) for x in lines[elem_start].split()]
    num_entity_blocks = header[0]

    cells: list[np.ndarray] = []
    boundary_edges: list[tuple[int, int, int]] = []

    current_line = elem_start + 1
    for _ in range(num_entity_blocks):
        ent_info = lines[current_line].split()
        current_line += 1
        entity_dim = int(ent_info[0])
        _entity_tag = int(ent_info[1])
        elem_type = int(ent_info[2])
        num_elem_in_block = int(ent_info[3])

        # For boundaries, the entity tag is the physical tag (in most meshes)
        physical_tag = _entity_tag

        for _ in range(num_elem_in_block):
            parts = lines[current_line].split()
            current_line += 1
            # first entry is element tag
            node_tags = [int(x) for x in parts[1:]]
            node_ids = [node_tag_to_index[t] for t in node_tags]

            if entity_dim == 1 and elem_type == 1 and len(node_ids) == 2:
                boundary_edges.append((node_ids[0], node_ids[1], physical_tag))
            elif entity_dim == 2 and elem_type in (2, 3):
                cells.append(np.array(node_ids, dtype=int))
            else:
                continue

    return Mesh2D.from_cells(nodes, cells, boundary_edges)
