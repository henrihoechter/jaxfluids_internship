"""Helpers for building boundary-condition arrays."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from . import chemistry_types
from compressible.mesh import Mesh
from compressible.equation_manager_types import BoundaryConditionArrays
from compressible.boundary_conditions_types import (
    BC_OUTFLOW,
    BC_INFLOW,
    BC_WALL,
    BC_WALL_SLIP,
    BC_WALL_EULER,
    BC_REFLECTIVE,
)


def _empty_bc_arrays(n_faces: int, n_species: int) -> dict[str, np.ndarray]:
    """Create boundary-condition arrays with default values."""
    return dict(
        bc_id=np.full(n_faces, -1, dtype=np.int32),
        inflow_rho=np.ones(n_faces),
        inflow_u=np.zeros(n_faces),
        inflow_v=np.zeros(n_faces),
        inflow_T=np.full(n_faces, 300.0),
        inflow_Tv=np.full(n_faces, 300.0),
        inflow_Y=np.zeros((n_faces, n_species)),
        wall_Tw=np.zeros(n_faces),
        wall_Tvw=np.zeros(n_faces),
        wall_has_Tw=np.zeros(n_faces, dtype=bool),
        wall_has_Tvw=np.zeros(n_faces, dtype=bool),
        wall_Y=np.zeros((n_faces, n_species)),
        wall_has_Y=np.zeros(n_faces, dtype=bool),
        wall_u=np.zeros(n_faces),
        wall_v=np.zeros(n_faces),
        wall_sigma_t=np.ones(n_faces),
        wall_sigma_v=np.ones(n_faces),
        wall_dist=np.ones(n_faces),
    )


def _finalize(d: dict[str, np.ndarray]) -> BoundaryConditionArrays:
    """Convert numpy boundary arrays to JAX arrays."""
    return BoundaryConditionArrays(
        bc_id=jnp.asarray(d["bc_id"]),
        inflow_rho=jnp.asarray(d["inflow_rho"]),
        inflow_u=jnp.asarray(d["inflow_u"]),
        inflow_v=jnp.asarray(d["inflow_v"]),
        inflow_T=jnp.asarray(d["inflow_T"]),
        inflow_Tv=jnp.asarray(d["inflow_Tv"]),
        inflow_Y=jnp.asarray(d["inflow_Y"]),
        wall_Tw=jnp.asarray(d["wall_Tw"]),
        wall_Tvw=jnp.asarray(d["wall_Tvw"]),
        wall_has_Tw=jnp.asarray(d["wall_has_Tw"]),
        wall_has_Tvw=jnp.asarray(d["wall_has_Tvw"]),
        wall_Y=jnp.asarray(d["wall_Y"]),
        wall_has_Y=jnp.asarray(d["wall_has_Y"]),
        wall_u=jnp.asarray(d["wall_u"]),
        wall_v=jnp.asarray(d["wall_v"]),
        wall_sigma_t=jnp.asarray(d["wall_sigma_t"]),
        wall_sigma_v=jnp.asarray(d["wall_sigma_v"]),
        wall_dist=jnp.asarray(d["wall_dist"]),
    )


def build_boundary_arrays_1d(
    mesh: Mesh,
    bc_left: str,
    bc_right: str,
    n_species: int,
    inflow_left: dict | None = None,
    inflow_right: dict | None = None,
) -> BoundaryConditionArrays:
    """Build BoundaryConditionArrays for a 1D mesh.

    Maps 1D BC names to the unified BC system:
        "outflow" -> BC_OUTFLOW
        "reflective" -> BC_REFLECTIVE
        "inflow" -> BC_INFLOW

    Periodic 1D meshes do not have boundary faces (Mesh.from_1d_grid with
    periodic=True); call build_boundary_arrays_1d_periodic() instead.

    Args:
        mesh: 1D mesh built with Mesh.from_1d_grid().
        bc_left: BC type for the left boundary ("outflow", "reflective", "inflow").
        bc_right: BC type for the right boundary.
        n_species: Number of species.
        inflow_left: Inflow state dict for bc_left="inflow" (keys: rho, u, T, Tv, Y).
        inflow_right: Inflow state dict for bc_right="inflow".

    Returns:
        BoundaryConditionArrays with n_faces entries.
    """
    n_faces = mesh.face_left.shape[0]
    d = _empty_bc_arrays(n_faces, n_species)
    d["inflow_Y"][:, 0] = 1.0  # default: pure first species

    tags = np.asarray(mesh.boundary_tags)
    left_tag = mesh.boundary_tags[0]  # face 0 is left BC
    right_tag = mesh.boundary_tags[-1]  # last face is right BC

    # Interior faces get BC_OUTFLOW (they are never evaluated as BCs)
    interior_mask = tags == -1
    d["bc_id"][interior_mask] = BC_OUTFLOW

    for bc_str, tag, inflow in [
        (bc_left, left_tag, inflow_left),
        (bc_right, right_tag, inflow_right),
    ]:
        mask = tags == tag
        if not np.any(mask):
            continue
        if bc_str == "outflow":
            d["bc_id"][mask] = BC_OUTFLOW
        elif bc_str == "reflective":
            d["bc_id"][mask] = BC_REFLECTIVE
        elif bc_str == "inflow":
            if inflow is None:
                raise ValueError("bc_str='inflow' requires an inflow state dict")
            d["bc_id"][mask] = BC_INFLOW
            d["inflow_rho"][mask] = float(inflow["rho"])
            d["inflow_u"][mask] = float(inflow["u"])
            # inflow_v stays 0 for 1D
            d["inflow_T"][mask] = float(inflow["T"])
            d["inflow_Tv"][mask] = float(inflow.get("Tv", inflow["T"]))
            Y = np.asarray(inflow["Y"], dtype=float)
            if Y.shape[0] != n_species:
                raise ValueError("inflow['Y'] must have length n_species")
            d["inflow_Y"][mask] = Y[None, :]
        else:
            raise ValueError(f"Unknown 1D BC type: {bc_str!r}")

    return _finalize(d)


def build_boundary_arrays_1d_periodic(
    mesh: Mesh,
    n_species: int,
) -> BoundaryConditionArrays:
    """BoundaryConditionArrays for a periodic 1D mesh.

    For periodic meshes, all faces are interior (face_right >= 0 always), so
    no BC evaluation is performed.  Returns an array of BC_OUTFLOW IDs that
    are never selected in practice.
    """
    n_faces = mesh.face_left.shape[0]
    d = _empty_bc_arrays(n_faces, n_species)
    d["bc_id"][:] = BC_OUTFLOW
    d["inflow_Y"][:, 0] = 1.0
    return _finalize(d)


def build_boundary_arrays_2d(
    mesh: Mesh,
    tag_to_bc: dict[int, dict],
    species: chemistry_types.SpeciesTable,
) -> BoundaryConditionArrays:
    """Build BoundaryConditionArrays for a 2D unstructured mesh.

    Args:
        mesh: 2D mesh built with Mesh.from_cells() or Mesh.from_gmsh().
        tag_to_bc: Mapping from boundary tag to the BC config dict.
            Each dict must have "type" key with one of:
            "outflow", "inflow", "wall", "wall_slip", "wall_euler".
            Use "wall_euler" for inviscid slip, symmetry, and axis boundaries.
        species: Species table (for n_species and default Y).

    Returns:
        BoundaryConditionArrays with one entry per face.
    """
    n_faces = mesh.face_left.shape[0]
    n_species = species.n_species
    d = _empty_bc_arrays(n_faces, n_species)
    d["inflow_Y"][:, 0] = 1.0

    # Compute wall distances from face centroids to cell centroids
    face_centroids = np.asarray(mesh.face_centroids)
    cell_centroids = np.asarray(mesh.cell_centroids)
    face_normals = np.asarray(mesh.face_normals)
    face_left = np.asarray(mesh.face_left)
    cell_to_face = face_centroids - cell_centroids[face_left]
    wall_dist = np.abs(np.sum(cell_to_face * face_normals, axis=1))
    wall_dist = np.clip(wall_dist, 1e-12, None)
    d["wall_dist"] = wall_dist

    tags = np.asarray(mesh.boundary_tags)

    for tag, bc in tag_to_bc.items():
        mask = tags == tag
        if not np.any(mask):
            continue
        bc_type = bc.get("type")
        if bc_type == "outflow":
            d["bc_id"][mask] = BC_OUTFLOW
        elif bc_type == "inflow":
            d["bc_id"][mask] = BC_INFLOW
            d["inflow_rho"][mask] = float(bc["rho"])
            d["inflow_u"][mask] = float(bc["u"])
            d["inflow_v"][mask] = float(bc.get("v", 0.0))
            d["inflow_T"][mask] = float(bc["T"])
            d["inflow_Tv"][mask] = float(bc.get("Tv", bc["T"]))
            Y = np.asarray(bc["Y"], dtype=float)
            if Y.ndim != 1 or Y.shape[0] != n_species:
                raise ValueError("Inflow Y must have shape (n_species,)")
            d["inflow_Y"][mask] = Y[None, :]
        elif bc_type == "wall":
            d["bc_id"][mask] = BC_WALL
            if "Tw" in bc:
                d["wall_Tw"][mask] = float(bc["Tw"])
                d["wall_has_Tw"][mask] = True
            if "Tvw" in bc:
                d["wall_Tvw"][mask] = float(bc["Tvw"])
                d["wall_has_Tvw"][mask] = True
            if "Y_wall" in bc:
                Yw = np.asarray(bc["Y_wall"], dtype=float)
                d["wall_Y"][mask] = Yw[None, :]
                d["wall_has_Y"][mask] = True
        elif bc_type == "wall_euler":
            d["bc_id"][mask] = BC_WALL_EULER
        elif bc_type == "wall_slip":
            d["bc_id"][mask] = BC_WALL_SLIP
            if "Tw" in bc:
                d["wall_Tw"][mask] = float(bc["Tw"])
                d["wall_has_Tw"][mask] = True
            if "Tvw" in bc:
                d["wall_Tvw"][mask] = float(bc["Tvw"])
                d["wall_has_Tvw"][mask] = True
            if "Y_wall" in bc:
                Yw = np.asarray(bc["Y_wall"], dtype=float)
                d["wall_Y"][mask] = Yw[None, :]
                d["wall_has_Y"][mask] = True
            if "u_wall" in bc:
                d["wall_u"][mask] = float(bc["u_wall"])
            if "v_wall" in bc:
                d["wall_v"][mask] = float(bc["v_wall"])
            if "sigma_t" in bc:
                d["wall_sigma_t"][mask] = float(bc["sigma_t"])
            if "sigma_v" in bc:
                d["wall_sigma_v"][mask] = float(bc["sigma_v"])
        else:
            raise ValueError(f"Unknown boundary condition type: {bc_type!r}")

    # Check for unassigned boundary faces
    face_right = np.asarray(mesh.face_right)
    boundary_mask = face_right < 0
    missing = boundary_mask & (d["bc_id"] < 0)
    if np.any(missing):
        missing_tags = np.unique(tags[missing]).tolist()
        raise ValueError(f"Missing boundary config for tags: {missing_tags}")

    return _finalize(d)
