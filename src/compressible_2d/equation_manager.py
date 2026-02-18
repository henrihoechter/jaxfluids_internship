"""Equation manager for 2D axisymmetric multi-species solver."""

from __future__ import annotations

import math
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from .mesh_gmsh import Mesh2D
from .equation_manager_types import EquationManager2D
from . import boundary_conditions
from . import equation_manager_utils
from . import solver
from . import viscous_flux
from . import source_terms


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




from .equation_manager_utils import build_equation_manager




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
    U_L, U_R = boundary_conditions.compute_face_states(U, mesh, equation_manager)
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
    U_L, U_R = boundary_conditions.compute_face_states(U, mesh, equation_manager)
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
            "Build it with compressible_2d.build_boundary_arrays(...)."
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
