"""Helpers for building boundary condition arrays."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from compressible_core import chemistry_types
from .mesh_gmsh import Mesh2D
from .equation_manager_types import BoundaryConditionArrays2D, BoundaryConditionConfig2D
from .boundary_conditions_types import (
    BC_OUTFLOW,
    BC_INFLOW,
    BC_AXISYMMETRIC,
    BC_WALL,
    BC_WALL_SLIP,
    BC_WALL_EULER,
)


def _build_boundary_arrays(
    mesh: Mesh2D,
    boundary_config: BoundaryConditionConfig2D,
    species: chemistry_types.SpeciesTable,
) -> BoundaryConditionArrays2D:
    n_faces = mesh.face_left.shape[0]
    n_species = species.n_species

    bc_id = np.full((n_faces,), -1, dtype=np.int32)
    inflow_rho = np.ones((n_faces,), dtype=float)
    inflow_u = np.zeros((n_faces,), dtype=float)
    inflow_v = np.zeros((n_faces,), dtype=float)
    inflow_T = np.full((n_faces,), 300.0, dtype=float)
    inflow_Tv = np.full((n_faces,), 300.0, dtype=float)
    inflow_Y = np.zeros((n_faces, n_species), dtype=float)
    inflow_Y[:, 0] = 1.0

    wall_Tw = np.zeros((n_faces,), dtype=float)
    wall_Tvw = np.zeros((n_faces,), dtype=float)
    wall_has_Tw = np.zeros((n_faces,), dtype=bool)
    wall_has_Tvw = np.zeros((n_faces,), dtype=bool)
    wall_Y = np.zeros((n_faces, n_species), dtype=float)
    wall_has_Y = np.zeros((n_faces,), dtype=bool)
    wall_u = np.zeros((n_faces,), dtype=float)
    wall_v = np.zeros((n_faces,), dtype=float)
    wall_sigma_t = np.ones((n_faces,), dtype=float)
    wall_sigma_v = np.ones((n_faces,), dtype=float)

    face_centroids = np.asarray(mesh.face_centroids)
    cell_centroids = np.asarray(mesh.cell_centroids)
    face_normals = np.asarray(mesh.face_normals)
    face_left = np.asarray(mesh.face_left)
    cell_to_face = face_centroids - cell_centroids[face_left]
    wall_dist = np.abs(np.sum(cell_to_face * face_normals, axis=1))
    wall_dist = np.clip(wall_dist, 1e-12, None)

    tags = np.asarray(mesh.boundary_tags)
    tag_to_bc = boundary_config.tag_to_bc
    for tag, bc in tag_to_bc.items():
        mask = tags == tag
        if not np.any(mask):
            continue
        bc_type = bc.get("type")
        if bc_type == "outflow":
            bc_id[mask] = BC_OUTFLOW
        elif bc_type == "inflow":
            bc_id[mask] = BC_INFLOW
            inflow_rho[mask] = float(bc["rho"])
            inflow_u[mask] = float(bc["u"])
            inflow_v[mask] = float(bc["v"])
            inflow_T[mask] = float(bc["T"])
            inflow_Tv[mask] = float(bc.get("Tv", bc["T"]))
            Y = np.asarray(bc["Y"], dtype=float)
            if Y.ndim != 1 or Y.shape[0] != n_species:
                raise ValueError("Inflow Y must have shape (n_species,)")
            inflow_Y[mask, :] = Y[None, :]
        elif bc_type == "axisymmetric":
            bc_id[mask] = BC_AXISYMMETRIC
        elif bc_type == "wall":
            bc_id[mask] = BC_WALL
            if "Tw" in bc:
                wall_Tw[mask] = float(bc["Tw"])
                wall_has_Tw[mask] = True
            if "Tvw" in bc:
                wall_Tvw[mask] = float(bc["Tvw"])
                wall_has_Tvw[mask] = True
            if "Y_wall" in bc:
                Yw = np.asarray(bc["Y_wall"], dtype=float)
                if Yw.ndim != 1 or Yw.shape[0] != n_species:
                    raise ValueError("Y_wall must have shape (n_species,)")
                wall_Y[mask, :] = Yw[None, :]
                wall_has_Y[mask] = True
        elif bc_type == "wall_euler":
            bc_id[mask] = BC_WALL_EULER
        elif bc_type == "wall_slip":
            bc_id[mask] = BC_WALL_SLIP
            if "Tw" in bc:
                wall_Tw[mask] = float(bc["Tw"])
                wall_has_Tw[mask] = True
            if "Tvw" in bc:
                wall_Tvw[mask] = float(bc["Tvw"])
                wall_has_Tvw[mask] = True
            if "Y_wall" in bc:
                Yw = np.asarray(bc["Y_wall"], dtype=float)
                if Yw.ndim != 1 or Yw.shape[0] != n_species:
                    raise ValueError("Y_wall must have shape (n_species,)")
                wall_Y[mask, :] = Yw[None, :]
                wall_has_Y[mask] = True
            if "u_wall" in bc:
                wall_u[mask] = float(bc["u_wall"])
            if "v_wall" in bc:
                wall_v[mask] = float(bc["v_wall"])
            if "sigma_t" in bc:
                wall_sigma_t[mask] = float(bc["sigma_t"])
            if "sigma_v" in bc:
                wall_sigma_v[mask] = float(bc["sigma_v"])
        else:
            raise ValueError(f"Unknown boundary condition type: {bc_type}")

    boundary_mask = mesh.face_right < 0
    missing = boundary_mask & (bc_id < 0)
    if np.any(missing):
        missing_tags = np.unique(tags[missing]).tolist()
        raise ValueError(f"Missing boundary config for tags: {missing_tags}")

    return BoundaryConditionArrays2D(
        bc_id=jnp.asarray(bc_id),
        inflow_rho=jnp.asarray(inflow_rho),
        inflow_u=jnp.asarray(inflow_u),
        inflow_v=jnp.asarray(inflow_v),
        inflow_T=jnp.asarray(inflow_T),
        inflow_Tv=jnp.asarray(inflow_Tv),
        inflow_Y=jnp.asarray(inflow_Y),
        wall_Tw=jnp.asarray(wall_Tw),
        wall_Tvw=jnp.asarray(wall_Tvw),
        wall_has_Tw=jnp.asarray(wall_has_Tw),
        wall_has_Tvw=jnp.asarray(wall_has_Tvw),
        wall_Y=jnp.asarray(wall_Y),
        wall_has_Y=jnp.asarray(wall_has_Y),
        wall_u=jnp.asarray(wall_u),
        wall_v=jnp.asarray(wall_v),
        wall_sigma_t=jnp.asarray(wall_sigma_t),
        wall_sigma_v=jnp.asarray(wall_sigma_v),
        wall_dist=jnp.asarray(wall_dist),
    )


def build_boundary_arrays(
    mesh: Mesh2D,
    boundary_config: BoundaryConditionConfig2D,
    species: chemistry_types.SpeciesTable,
) -> BoundaryConditionArrays2D:
    return _build_boundary_arrays(mesh, boundary_config, species)
