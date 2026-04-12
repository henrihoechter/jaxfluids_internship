"""Time-stepping helpers for the unified compressible solver."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from .mesh import Mesh
from .equation_manager_types import EquationManager
from . import boundary_conditions
from . import solver
from . import source_terms
from . import state as state_module
from . import viscous_flux


def _face_weights(
    mesh: Mesh,
) -> tuple[Float[Array, "n_faces"], Float[Array, "n_cells"]]:
    """Return the face and cell metric weights."""
    face_areas = jnp.asarray(mesh.face_areas)
    cell_areas = jnp.asarray(mesh.cell_areas)
    if mesh.axisymmetric:
        face_r = jnp.asarray(mesh.face_r)
        cell_r = jnp.asarray(mesh.cell_r)
        face_w = face_areas * (2.0 * math.pi * face_r)
        cell_w = cell_areas * (2.0 * math.pi * jnp.clip(cell_r, 1e-12, None))
    else:
        face_w = face_areas
        cell_w = cell_areas
    return face_w, cell_w

@jax.named_call
def compute_divergence(
    F: Float[Array, "n_faces n_variables"],
    mesh: Mesh,
    equation_manager: EquationManager,
) -> Float[Array, "n_cells n_variables"]:
    """Scatter the face fluxes into a cell-centered divergence."""
    face_left = jnp.asarray(mesh.face_left)
    face_right = jnp.asarray(mesh.face_right)
    face_w, cell_w = _face_weights(mesh)

    flux = F * face_w[:, None]
    n_cells = mesh.cell_areas.shape[0]
    n_vars = F.shape[1]
    dU = jnp.zeros((n_cells, n_vars))

    dU = dU.at[face_left].add(-flux / cell_w[face_left][:, None])

    right_mask = face_right >= 0
    safe_right = jnp.where(right_mask, face_right, 0)
    right_contrib = flux / cell_w[safe_right][:, None]
    right_contrib = jnp.where(right_mask[:, None], right_contrib, 0.0)
    dU = dU.at[safe_right].add(right_contrib)

    return dU


@jax.named_call
def compute_cfl_dt(
    U: Float[Array, "n_cells n_variables"],
    mesh: Mesh,
    equation_manager: EquationManager,
) -> float:
    """Compute the CFL-limited time step."""
    U_L, U_R = boundary_conditions.compute_face_states(U, mesh, equation_manager)
    n_hat = jnp.asarray(mesh.face_normals)

    prim_L = state_module.extract_primitives_from_U(U_L, equation_manager)
    prim_R = state_module.extract_primitives_from_U(U_R, equation_manager)

    n_x = n_hat[:, 0]
    n_y = n_hat[:, 1]
    u_n_L = prim_L.u * n_x + prim_L.v * n_y
    u_n_R = prim_R.u * n_x + prim_R.v * n_y

    a_L = solver.compute_speed_of_sound(
        prim_L.rho, prim_L.p, prim_L.Y_s, prim_L.T, prim_L.Tv, equation_manager
    )
    a_R = solver.compute_speed_of_sound(
        prim_R.rho, prim_R.p, prim_R.Y_s, prim_R.T, prim_R.Tv, equation_manager
    )
    lam = jnp.maximum(jnp.abs(u_n_L) + a_L, jnp.abs(u_n_R) + a_R)

    face_w, cell_w = _face_weights(mesh)
    face_left = jnp.asarray(mesh.face_left)
    face_right = jnp.asarray(mesh.face_right)
    n_cells = mesh.cell_areas.shape[0]

    speed_sum = jnp.zeros((n_cells,))
    speed_sum = speed_sum.at[face_left].add(lam * face_w)
    right_mask = face_right >= 0
    safe_right = jnp.where(right_mask, face_right, 0)
    speed_sum = speed_sum.at[safe_right].add(jnp.where(right_mask, lam * face_w, 0.0))

    dt_local = cell_w / jnp.clip(speed_sum, 1e-30, None)
    return float(equation_manager.numerics_config.cfl * jnp.min(dt_local))


@jax.named_call
def _compute_dU_dt(
    U: Float[Array, "n_cells n_variables"],
    mesh: Mesh,
    equation_manager: EquationManager,
) -> Float[Array, "n_cells n_variables"]:
    """Compute the convective and diffusive contribution to dU/dt."""
    spatial_scheme = equation_manager.numerics_config.spatial_scheme
    if spatial_scheme == "muscl":
        U_L, U_R = solver.compute_face_states_muscl(U, mesh, equation_manager)
    else:
        U_L, U_R = solver.compute_face_states(U, mesh, equation_manager)

    normals = jnp.asarray(mesh.face_normals)

    prim_L = state_module.extract_primitives_from_U(U_L, equation_manager)
    prim_R = state_module.extract_primitives_from_U(U_R, equation_manager)

    F_c = solver.compute_flux_faces(
        U_L,
        U_R,
        normals,
        equation_manager,
        primitives_L=prim_L,
        primitives_R=prim_R,
    )
    dU_dt = compute_divergence(F_c, mesh, equation_manager)

    cell_prim = state_module.extract_primitives_from_U(U, equation_manager)
    F_v = viscous_flux.compute_viscous_flux_faces(
        U,
        U_L,
        U_R,
        mesh,
        equation_manager,
        cell_primitives=cell_prim,
        face_primitives_L=prim_L,
        face_primitives_R=prim_R,
    )
    dU_dt = dU_dt + compute_divergence(F_v, mesh, equation_manager)

    return dU_dt

@jax.named_call
def _clip_conserved_state(
    U: Float[Array, "n_cells n_variables"],
    equation_manager: EquationManager,
) -> Float[Array, "n_cells n_variables"]:
    """Clip the conserved state before it is reused."""
    n_species = equation_manager.species.n_species
    clip = equation_manager.numerics_config.clipping

    U = U.at[:, :n_species].set(
        jnp.clip(U[:, :n_species], clip.rho_s_min, clip.rho_s_max)
    )
    U = U.at[:, n_species].set(
        jnp.clip(U[:, n_species], clip.rho_u_min, clip.rho_u_max)
    )
    U = U.at[:, n_species + 1].set(
        jnp.clip(U[:, n_species + 1], clip.rho_v_min, clip.rho_v_max)
    )
    U = U.at[:, n_species + 2].set(
        jnp.clip(U[:, n_species + 2], clip.rho_E_min, clip.rho_E_max)
    )
    U = U.at[:, n_species + 3].set(
        jnp.clip(U[:, n_species + 3], clip.rho_Ev_min, clip.rho_Ev_max)
    )
    return U


@jax.named_call
def advance_one_step(
    U: Float[Array, "n_cells n_variables"],
    mesh: Mesh,
    equation_manager: EquationManager,
    dt: float | None = None,
) -> Float[Array, "n_cells n_variables"]:
    """Advance the solution by one time step."""
    if dt is None:
        if equation_manager.numerics_config.dt_mode == "cfl":
            dt = compute_cfl_dt(U, mesh, equation_manager)
        else:
            dt = equation_manager.numerics_config.dt

    prim = state_module.extract_primitives_from_U(U, equation_manager)
    S = source_terms.compute_source_terms(U, equation_manager, primitives=prim)
    U = U + 0.5 * dt * S
    U = _clip_conserved_state(U, equation_manager)

    dU_dt = _compute_dU_dt(U, mesh, equation_manager)
    integrator = equation_manager.numerics_config.integrator_scheme
    if integrator == "forward-euler":
        U = U + dt * dU_dt
    else:
        U_half = U + 0.5 * dt * dU_dt
        U_half = _clip_conserved_state(U_half, equation_manager)
        dU_dt_half = _compute_dU_dt(U_half, mesh, equation_manager)
        U = U + dt * dU_dt_half
    U = _clip_conserved_state(U, equation_manager)

    prim = state_module.extract_primitives_from_U(U, equation_manager)
    S = source_terms.compute_source_terms(U, equation_manager, primitives=prim)
    U = U + 0.5 * dt * S
    return _clip_conserved_state(U, equation_manager)

def _check_muscl(mesh: Mesh, equation_manager: EquationManager) -> None:
    """Check that MUSCL was requested on a mesh with a valid stencil."""
    if equation_manager.numerics_config.spatial_scheme == "muscl":
        if np.all(np.asarray(mesh.muscl_ll) < 0):
            raise ValueError(
                "spatial_scheme='muscl' requires a structured mesh built with "
                "Mesh.from_1d_grid(). The provided mesh has no MUSCL stencil."
            )


def run(
    U_init: Float[Array, "n_cells n_variables"],
    mesh: Mesh,
    equation_manager: EquationManager,
    t_final: float,
    save_interval: int = 100,
    history_device: str = "device",
    dt_array: Float[Array, "n_steps"] | None = None,
) -> tuple[
    Float[Array, "n_snapshots n_cells n_variables"],
    Float[Array, "n_snapshots"],
]:
    """Run the solver loop until the requested final time.

    Args:
        U_init: Initial condition.
        mesh: Unified Mesh (1D or 2D).
        equation_manager: Physics and numerics configuration.
        t_final: Final simulation time [s].
        save_interval: Save solution every N time steps.
        history_device: "device" (accelerator) or "cpu"/"host" (host memory).
        dt_array: Optional explicit per-step dt sequence (overrides t_final).

    Returns:
        Tuple of (U_history, t_history): solution snapshots and corresponding times.
    """
    _check_muscl(mesh, equation_manager)

    history_device = history_device.lower()
    if history_device in ("cpu", "host"):
        store_on_cpu = True
    elif history_device in ("device", "gpu"):
        store_on_cpu = False
    else:
        raise ValueError("history_device must be 'device' or 'cpu'")

    U = U_init
    t = 0.0
    _step = jax.jit(advance_one_step)

    if dt_array is not None:
        dt_sequence = np.asarray(dt_array)
        n_steps = int(dt_sequence.shape[0])
        dt = (
            float(dt_sequence[0])
            if n_steps > 0
            else float(equation_manager.numerics_config.dt)
        )
    elif equation_manager.numerics_config.dt_mode == "cfl":
        dt = compute_cfl_dt(U, mesh, equation_manager)
        dt_sequence = None
        n_steps = int(t_final / dt)
    else:
        dt = equation_manager.numerics_config.dt
        dt_sequence = None
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
        if dt_sequence is not None:
            dt = float(dt_sequence[step - 1])
        elif equation_manager.numerics_config.dt_mode == "cfl":
            dt = compute_cfl_dt(U, mesh, equation_manager)
        U = _step(U, mesh, equation_manager, dt)
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
    mesh: Mesh,
    equation_manager: EquationManager,
    t_final: float,
    save_interval: int = 100,
    dt_array: Float[Array, "n_steps"] | None = None,
) -> tuple[
    Float[Array, "n_snapshots n_cells n_variables"],
    Float[Array, "n_snapshots"],
]:
    """Run the solver loop with `jax.lax.scan`.

    Raises:
        ValueError: If dt_mode is not 'fixed'.
    """
    _check_muscl(mesh, equation_manager)

    if equation_manager.numerics_config.dt_mode != "fixed":
        raise ValueError(
            "run_scan only supports dt_mode='fixed'. Use run() for CFL-adaptive stepping."
        )

    dt0 = equation_manager.numerics_config.dt
    if dt_array is None:
        n_steps = int(t_final / dt0)
        dt_sequence = jnp.full((n_steps,), dt0, dtype=jnp.result_type(dt0, 0.0))
    else:
        dt_sequence = jnp.asarray(dt_array)
        n_steps = int(dt_sequence.shape[0])

    n_snapshots = int(n_steps // save_interval) + 1
    n_cells, n_vars = U_init.shape

    U_history0 = jnp.zeros((n_snapshots, n_cells, n_vars), dtype=U_init.dtype)
    t_history0 = jnp.zeros((n_snapshots,), dtype=jnp.result_type(dt_sequence, 0.0))
    U_history0 = U_history0.at[0].set(U_init)
    t_history0 = t_history0.at[0].set(0.0)

    carry0 = (
        U_init,
        jnp.array(0.0, dtype=t_history0.dtype),
        jnp.array(1, dtype=jnp.int32),
        U_history0,
        t_history0,
    )

    def _dump_if_nan(U_concrete, t_concrete, step_idx_concrete):
        if jnp.any(jnp.isnan(U_concrete)):
            path = f"nan_dump_step{int(step_idx_concrete)}.npz"
            np.savez(path, U=np.array(U_concrete), t=np.array(t_concrete))
            raise RuntimeError(
                f"NaN detected at step {int(step_idx_concrete)}, "
                f"t={float(t_concrete):.6e}. State dumped to {path}"
            )

    def body(carry, xs):
        U, t, snap_i, U_hist, t_hist = carry
        dt, step_idx = xs
        U = advance_one_step(U, mesh, equation_manager, dt)
        t = t + dt

        jax.debug.callback(_dump_if_nan, U, t, step_idx, ordered=False)

        save = (step_idx % save_interval) == 0

        def do_save(args):
            U_, t_, snap_i_, U_hist_, t_hist_ = args
            U_hist_ = U_hist_.at[snap_i_].set(U_)
            t_hist_ = t_hist_.at[snap_i_].set(t_)
            return (U_, t_, snap_i_ + jnp.array(1, jnp.int32), U_hist_, t_hist_)

        carry = jax.lax.cond(
            save, do_save, lambda args: args, (U, t, snap_i, U_hist, t_hist)
        )
        return carry, None

    run_scan_jit = jax.jit(
        lambda: jax.lax.scan(
            body,
            carry0,
            xs=(dt_sequence, jnp.arange(1, n_steps + 1, dtype=jnp.int32)),
        )
    )
    carry_final, _ = run_scan_jit()
    _, _, _, U_history, t_history = carry_final
    return U_history, t_history
