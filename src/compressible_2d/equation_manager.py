"""Equation manager for 2D axisymmetric multi-species solver."""

from __future__ import annotations

import dataclasses
import math
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from compressible_core import chemistry_types
from .mesh_gmsh import Mesh2D
from .equation_manager_types import (
    BoundaryConditionArrays2D,
    BoundaryConditionConfig2D,
    EquationManager2D,
)
from . import boundary_conditions
from . import equation_manager_utils
from . import solver
from . import viscous_flux
from . import source_terms

BC_OUTFLOW = 0
BC_INFLOW = 1
BC_AXISYMMETRIC = 2
BC_WALL = 3
BC_WALL_SLIP = 4


def _face_weights(mesh: Mesh2D, axisymmetric: bool) -> tuple[jnp.ndarray, jnp.ndarray]:
    face_areas = jnp.asarray(mesh.face_areas)
    cell_areas = jnp.asarray(mesh.cell_areas)
    if axisymmetric:
        face_r = jnp.asarray(mesh.face_r)
        cell_r = jnp.asarray(mesh.cell_r)
        face_w = face_areas * (2.0 * math.pi * face_r)
        cell_w = cell_areas * (2.0 * math.pi * jnp.clip(cell_r, 1e-12, None))
    else:
        face_w = face_areas
        cell_w = cell_areas
    return face_w, cell_w


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


def build_equation_manager(
    mesh: Mesh2D,
    *,
    species: chemistry_types.SpeciesTable,
    collision_integrals: chemistry_types.CollisionIntegralTable | None,
    reactions: chemistry_types.ReactionTable | None,
    numerics_config,
    boundary_config: BoundaryConditionConfig2D,
    transport_model,
    casseau_transport=None,
) -> EquationManager2D:
    boundary_arrays = _build_boundary_arrays(mesh, boundary_config, species)
    return EquationManager2D(
        species=species,
        collision_integrals=collision_integrals,
        reactions=reactions,
        numerics_config=numerics_config,
        boundary_arrays=boundary_arrays,
        transport_model=transport_model,
        casseau_transport=casseau_transport,
    )


def _ghost_inflow(
    boundary_arrays: BoundaryConditionArrays2D,
    equation_manager: EquationManager2D,
) -> Float[Array, "n_faces n_variables"]:
    return equation_manager_utils.compute_U_from_primitives(
        Y_s=boundary_arrays.inflow_Y,
        rho=boundary_arrays.inflow_rho,
        u=boundary_arrays.inflow_u,
        v=boundary_arrays.inflow_v,
        T_tr=boundary_arrays.inflow_T,
        T_V=boundary_arrays.inflow_Tv,
        equation_manager=equation_manager,
    )


def _ghost_axisymmetric(
    U_L: Float[Array, "n_faces n_variables"],
    n_hat: Float[Array, "n_faces 2"],
    equation_manager: EquationManager2D,
) -> Float[Array, "n_faces n_variables"]:
    Y_L, rho_L, u_L, v_L, T_L, Tv_L, _ = (
        equation_manager_utils.extract_primitives_from_U(U_L, equation_manager)
    )

    n_x = n_hat[:, 0]
    n_y = n_hat[:, 1]
    t_x = -n_y
    t_y = n_x

    u_n = u_L * n_x + v_L * n_y
    u_t = u_L * t_x + v_L * t_y

    u_n_g = -u_n
    u_t_g = u_t
    u_g = u_n_g * n_x + u_t_g * t_x
    v_g = u_n_g * n_y + u_t_g * t_y

    return equation_manager_utils.compute_U_from_primitives(
        Y_s=Y_L,
        rho=rho_L,
        u=u_g,
        v=v_g,
        T_tr=T_L,
        T_V=Tv_L,
        equation_manager=equation_manager,
    )


def _ghost_wall(
    U_L: Float[Array, "n_faces n_variables"],
    boundary_arrays: BoundaryConditionArrays2D,
    bc_id: Array,
    equation_manager: EquationManager2D,
) -> Float[Array, "n_faces n_variables"]:
    Y_L, rho_L, u_L, v_L, T_L, Tv_L, _ = (
        equation_manager_utils.extract_primitives_from_U(U_L, equation_manager)
    )

    wall_mask = bc_id == BC_WALL
    wall_count = jnp.sum(wall_mask)
    wall_count_safe = jnp.maximum(wall_count, 1.0)
    Tw_wall_mean = jnp.sum(T_L * wall_mask) / wall_count_safe
    Tw_default = jnp.where(wall_count > 0, Tw_wall_mean, jnp.mean(T_L))

    Tw = jnp.where(boundary_arrays.wall_has_Tw, boundary_arrays.wall_Tw, Tw_default)
    Tvw = jnp.where(boundary_arrays.wall_has_Tvw, boundary_arrays.wall_Tvw, Tw)
    Y = jnp.where(boundary_arrays.wall_has_Y[:, None], boundary_arrays.wall_Y, Y_L)

    u_g = -u_L
    v_g = -v_L

    return equation_manager_utils.compute_U_from_primitives(
        Y_s=Y,
        rho=rho_L,
        u=u_g,
        v=v_g,
        T_tr=Tw,
        T_V=Tvw,
        equation_manager=equation_manager,
    )


def _ghost_wall_slip(
    U_L: Float[Array, "n_faces n_variables"],
    boundary_arrays: BoundaryConditionArrays2D,
    bc_id: Array,
    n_hat: Float[Array, "n_faces 2"],
    equation_manager: EquationManager2D,
) -> Float[Array, "n_faces n_variables"]:
    Y_L, rho_L, u_L, v_L, T_L, Tv_L, _ = (
        equation_manager_utils.extract_primitives_from_U(U_L, equation_manager)
    )

    wall_mask = bc_id == BC_WALL_SLIP
    wall_count = jnp.sum(wall_mask)
    wall_count_safe = jnp.maximum(wall_count, 1.0)
    Tw_wall_mean = jnp.sum(T_L * wall_mask) / wall_count_safe
    Tw_default = jnp.where(wall_count > 0, Tw_wall_mean, jnp.mean(T_L))

    Tw = jnp.where(boundary_arrays.wall_has_Tw, boundary_arrays.wall_Tw, Tw_default)
    Tvw = jnp.where(boundary_arrays.wall_has_Tvw, boundary_arrays.wall_Tvw, Tw)
    Y = jnp.where(boundary_arrays.wall_has_Y[:, None], boundary_arrays.wall_Y, Y_L)

    return boundary_conditions.compute_slip_wall_ghost(
        U_L=U_L,
        n_hat=n_hat,
        equation_manager=equation_manager,
        Tw=Tw,
        Tvw=Tvw,
        Y_wall=Y,
        wall_u=boundary_arrays.wall_u,
        wall_v=boundary_arrays.wall_v,
        wall_dist=boundary_arrays.wall_dist,
        sigma_t=boundary_arrays.wall_sigma_t,
        sigma_v=boundary_arrays.wall_sigma_v,
    )


@jax.named_call
def compute_face_states(
    U: Float[Array, "n_cells n_variables"],
    mesh: Mesh2D,
    equation_manager: EquationManager2D,
) -> tuple[Float[Array, "n_faces n_variables"], Float[Array, "n_faces n_variables"]]:
    face_left = jnp.asarray(mesh.face_left)
    face_right = jnp.asarray(mesh.face_right)
    U_L = U[face_left]
    U_R = jnp.where(face_right[:, None] >= 0, U[face_right], U_L)

    bc = equation_manager.boundary_arrays
    if bc is None:
        raise ValueError(
            "boundary_arrays is required for JIT-safe execution. "
            "Build it with equation_manager.build_boundary_arrays(...)."
        )

    bc_id = bc.bc_id
    n_hat = jnp.asarray(mesh.face_normals)

    U_R_inflow = _ghost_inflow(bc, equation_manager)
    U_R_axis = _ghost_axisymmetric(U_L, n_hat, equation_manager)
    U_R_wall = _ghost_wall(U_L, bc, bc_id, equation_manager)
    U_R_wall_slip = _ghost_wall_slip(U_L, bc, bc_id, n_hat, equation_manager)

    mask_inflow = bc_id == BC_INFLOW
    mask_axis = bc_id == BC_AXISYMMETRIC
    mask_wall = bc_id == BC_WALL
    mask_wall_slip = bc_id == BC_WALL_SLIP

    U_R = jnp.where(mask_inflow[:, None], U_R_inflow, U_R)
    U_R = jnp.where(mask_axis[:, None], U_R_axis, U_R)
    U_R = jnp.where(mask_wall[:, None], U_R_wall, U_R)
    U_R = jnp.where(mask_wall_slip[:, None], U_R_wall_slip, U_R)
    return U_L, U_R


@jax.named_call
def compute_divergence(
    F: Float[Array, "n_faces n_variables"],
    mesh: Mesh2D,
    equation_manager: EquationManager2D,
) -> Float[Array, "n_cells n_variables"]:
    face_left = jnp.asarray(mesh.face_left)
    face_right = jnp.asarray(mesh.face_right)
    face_w, cell_w = _face_weights(mesh, equation_manager.numerics_config.axisymmetric)

    flux = F * face_w[:, None]
    n_cells = mesh.cell_areas.shape[0]
    n_vars = F.shape[1]
    dU = jnp.zeros((n_cells, n_vars))

    dU = dU.at[face_left].add(-flux / cell_w[face_left][:, None])
    right_mask = face_right >= 0
    safe_right = jnp.where(right_mask, face_right, 0)
    right_cell_w = cell_w[safe_right]
    right_contrib = flux / right_cell_w[:, None]
    right_contrib = jnp.where(right_mask[:, None], right_contrib, 0.0)
    dU = dU.at[safe_right].add(right_contrib)

    return dU


def compute_cfl_dt(
    U: Float[Array, "n_cells n_variables"],
    mesh: Mesh2D,
    equation_manager: EquationManager2D,
) -> float:
    # Compute max wave speed per face
    U_L, U_R = compute_face_states(U, mesh, equation_manager)
    n_hat = jnp.asarray(mesh.face_normals)
    # Use convective eigenvalues
    Y_L, rho_L, u_L, v_L, T_L, Tv_L, p_L = viscous_flux.extract_primitives(
        U_L, equation_manager
    )
    Y_R, rho_R, u_R, v_R, T_R, Tv_R, p_R = viscous_flux.extract_primitives(
        U_R, equation_manager
    )

    n_x = n_hat[:, 0]
    n_y = n_hat[:, 1]
    u_n_L = u_L * n_x + v_L * n_y
    u_n_R = u_R * n_x + v_R * n_y

    a_L = solver.compute_speed_of_sound(rho_L, p_L, Y_L, T_L, Tv_L, equation_manager)
    a_R = solver.compute_speed_of_sound(rho_R, p_R, Y_R, T_R, Tv_R, equation_manager)

    lam = jnp.maximum(jnp.abs(u_n_L) + a_L, jnp.abs(u_n_R) + a_R)

    face_w, cell_w = _face_weights(mesh, equation_manager.numerics_config.axisymmetric)

    face_left = jnp.asarray(mesh.face_left)
    face_right = jnp.asarray(mesh.face_right)
    n_cells = mesh.cell_areas.shape[0]

    speed_sum = jnp.zeros((n_cells,))
    speed_sum = speed_sum.at[face_left].add(lam * face_w)
    right_mask = face_right >= 0
    safe_right = jnp.where(right_mask, face_right, 0)
    add_vals = lam * face_w
    add_vals = jnp.where(right_mask, add_vals, 0.0)
    speed_sum = speed_sum.at[safe_right].add(add_vals)

    dt_local = cell_w / jnp.clip(speed_sum, 1e-30, None)
    dt = float(equation_manager.numerics_config.cfl * jnp.min(dt_local))
    return dt


@jax.named_call
def _compute_dU_dt(
    U: Float[Array, "n_cells n_variables"],
    mesh: Mesh2D,
    equation_manager: EquationManager2D,
) -> Float[Array, "n_cells n_variables"]:
    U_L, U_R = compute_face_states(U, mesh, equation_manager)
    normals = jnp.asarray(mesh.face_normals)

    face_primitives_L = equation_manager_utils.extract_primitives(U_L, equation_manager)
    face_primitives_R = equation_manager_utils.extract_primitives(U_R, equation_manager)

    # Convective flux
    F_c = solver.compute_flux_faces(
        U_L,
        U_R,
        normals,
        equation_manager,
        primitives_L=face_primitives_L,
        primitives_R=face_primitives_R,
    )
    dU_dt_conv = compute_divergence(F_c, mesh, equation_manager)

    # Diffusive flux
    cell_primitives = equation_manager_utils.extract_primitives(U, equation_manager)
    F_v = viscous_flux.compute_viscous_flux_faces(
        U,
        U_L,
        U_R,
        mesh,
        equation_manager,
        cell_primitives=cell_primitives,
        face_primitives_L=face_primitives_L,
        face_primitives_R=face_primitives_R,
    )
    dU_dt_diff = compute_divergence(F_v, mesh, equation_manager)

    return dU_dt_conv + dU_dt_diff


@jax.named_call
def advance_one_step(
    U: Float[Array, "n_cells n_variables"],
    mesh: Mesh2D,
    equation_manager: EquationManager2D,
    dt: float | None = None,
) -> Float[Array, "n_cells n_variables"]:
    if dt is None:
        if equation_manager.numerics_config.dt_mode == "cfl":
            dt = compute_cfl_dt(U, mesh, equation_manager)
        else:
            dt = equation_manager.numerics_config.dt

    # Source terms (half step)
    primitives = equation_manager_utils.extract_primitives(U, equation_manager)
    S = source_terms.compute_source_terms(U, equation_manager, primitives=primitives)
    U = U + 0.5 * dt * S

    dU_dt = _compute_dU_dt(U, mesh, equation_manager)

    if equation_manager.numerics_config.integrator_scheme == "forward-euler":
        U = U + dt * dU_dt
    else:
        U_half = U + 0.5 * dt * dU_dt
        dU_dt_half = _compute_dU_dt(U_half, mesh, equation_manager)
        U = U + dt * dU_dt_half

    # Source terms (half step)
    primitives = equation_manager_utils.extract_primitives(U, equation_manager)
    S = source_terms.compute_source_terms(U, equation_manager, primitives=primitives)
    U = U + 0.5 * dt * S

    return U


def run(
    U_init: Float[Array, "n_cells n_variables"],
    mesh: Mesh2D,
    equation_manager: EquationManager2D,
    t_final: float,
    save_interval: int = 100,
    history_device: str = "device",
) -> Tuple[
    Float[Array, "n_snapshots n_cells n_variables"], Float[Array, "n_snapshots"]
]:
    """Run simulation with optional host-side history storage.

    Args:
        U_init: Initial condition [n_cells, n_variables]
        mesh: Mesh data
        equation_manager: Contains all configuration
        t_final: Final simulation time
        save_interval: Save solution every N steps
        history_device: "device" to store history on accelerator, "cpu"/"host" to
            store on host memory (avoids GPU OOM at the cost of transfers).
    """
    U = U_init
    t = 0.0

    history_device = history_device.lower()
    if history_device in ("cpu", "host"):
        store_on_cpu = True
    elif history_device in ("device", "gpu"):
        store_on_cpu = False
    else:
        raise ValueError("history_device must be 'device' or 'cpu'")

    if equation_manager.numerics_config.dt_mode == "cfl":
        dt = compute_cfl_dt(U, mesh, equation_manager)
    else:
        dt = equation_manager.numerics_config.dt

    n_steps = int(t_final / dt)
    n_snapshots = int(n_steps // save_interval) + 1

    n_cells, n_vars = U_init.shape
    if store_on_cpu:
        U_history = np.zeros(
            (n_snapshots, n_cells, n_vars), dtype=np.dtype(U_init.dtype)
        )
        t_history = np.zeros((n_snapshots,), dtype=np.result_type(float(t), float(dt)))
        U_history[0] = np.asarray(jax.device_get(U_init))
        t_history[0] = 0.0
    else:
        U_history = jnp.zeros((n_snapshots, n_cells, n_vars), dtype=U_init.dtype)
        t_history = jnp.zeros((n_snapshots,), dtype=jnp.result_type(dt, 0.0))
        U_history = U_history.at[0].set(U_init)
        t_history = t_history.at[0].set(0.0)

    snapshot_idx = 1
    for step in range(1, n_steps + 1):
        if equation_manager.numerics_config.dt_mode == "cfl":
            dt = compute_cfl_dt(U, mesh, equation_manager)
        U = advance_one_step(U, mesh, equation_manager, dt)
        t += dt
        if step % save_interval == 0 and snapshot_idx < n_snapshots:
            if store_on_cpu:
                U_history[snapshot_idx] = np.asarray(jax.device_get(U))
                t_history[snapshot_idx] = float(t)
            else:
                U_history = U_history.at[snapshot_idx].set(U)
                t_history = t_history.at[snapshot_idx].set(t)
            snapshot_idx += 1

    return U_history, t_history


def run_scan(
    U_init: Float[Array, "n_cells n_variables"],
    mesh: Mesh2D,
    equation_manager: EquationManager2D,
    t_final: float,
    save_interval: int = 100,
) -> Tuple[
    Float[Array, "n_snapshots n_cells n_variables"], Float[Array, "n_snapshots"]
]:
    """Run simulation using jax.lax.scan.

    Note: boundary conditions must be converted to array form for JIT safety.
    """
    if equation_manager.boundary_arrays is None:
        raise ValueError(
            "boundary_arrays is required for JIT-safe execution. "
            "Build it with equation_manager.build_boundary_arrays(...)."
        )

    if equation_manager.numerics_config.dt_mode == "cfl":
        dt0 = compute_cfl_dt(U_init, mesh, equation_manager)
    else:
        dt0 = equation_manager.numerics_config.dt

    n_steps = int(t_final / dt0)
    n_snapshots = int(n_steps // save_interval) + 1

    n_cells, n_vars = U_init.shape
    U_history0 = jnp.zeros((n_snapshots, n_cells, n_vars), dtype=U_init.dtype)
    t_history0 = jnp.zeros((n_snapshots,), dtype=jnp.result_type(dt0, 0.0))
    U_history0 = U_history0.at[0].set(U_init)
    t_history0 = t_history0.at[0].set(0.0)

    carry0 = (
        U_init,
        jnp.array(0.0, dtype=t_history0.dtype),
        jnp.array(1, dtype=jnp.int32),
        U_history0,
        t_history0,
    )

    def body(carry, step_idx):
        U, t, snap_i, U_hist, t_hist = carry
        if equation_manager.numerics_config.dt_mode == "cfl":
            dt = compute_cfl_dt(U, mesh, equation_manager)
        else:
            dt = equation_manager.numerics_config.dt

        U = advance_one_step(U, mesh, equation_manager, dt)
        t = t + dt

        save = (step_idx % save_interval) == 0

        def do_save(args):
            U_, t_, snap_i_, U_hist_, t_hist_ = args
            U_hist_ = U_hist_.at[snap_i_].set(U_)
            t_hist_ = t_hist_.at[snap_i_].set(t_)
            return (U_, t_, snap_i_ + jnp.array(1, jnp.int32), U_hist_, t_hist_)

        def no_save(args):
            return args

        carry = jax.lax.cond(save, do_save, no_save, (U, t, snap_i, U_hist, t_hist))
        return carry, None

    carry_final, _ = jax.lax.scan(
        body,
        carry0,
        xs=jnp.arange(1, n_steps + 1, dtype=jnp.int32),
    )
    _, _, _, U_history, t_history = carry_final
    return U_history, t_history
