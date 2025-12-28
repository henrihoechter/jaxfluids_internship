"""Equation manager for 1D multi-species two-temperature Euler solver.

Implements the main solver loop with operator splitting for source terms.
"""
import warnings
from jaxtyping import Array, Float
import jax.numpy as jnp

from compressible_1d import boundary_conditions as bc_module
from compressible_1d import solver
from compressible_1d import source_terms
from compressible_1d import equation_manager_types
from compressible_1d import viscous_flux as viscous_flux_module
from compressible_1d import transport
from compressible_1d import equation_manager_utils
from compressible_1d import thermodynamic_relations
from compressible_1d.diagnose import runtime_check_array_sizes
def run(
    U_init: Float[Array, "n_cells n_variables"],
    equation_manager: equation_manager_types.EquationManager,
    t_final: float,
    save_interval: int = 100,
) -> tuple[
    Float[Array, "n_snapshots n_cells n_variables"], Float[Array, "n_snapshots"]
]:
    """Run simulation from t=0 to t=t_final.

    Args:
        U_init: Initial condition [n_cells, n_variables]
        equation_manager: Contains all configuration
        t_final: Final simulation time
        save_interval: Save solution every N steps

    Returns:
        U_history: Solution snapshots [n_snapshots, n_cells, n_variables]
        t_history: Time values [n_snapshots]
    """
    dt = equation_manager.numerics_config.dt

    # Check diffusive CFL at start (warns if violated)
    check_diffusive_cfl(U_init, equation_manager)

    # Time loop
    U = U_init
    t = 0.0
    n_steps = int(t_final / dt)
    n_snapshots = n_steps // save_interval + 1

    n_cells, n_variables = U_init.shape
    U_history = jnp.zeros((n_snapshots, n_cells, n_variables))
    t_history = jnp.zeros(n_snapshots)

    U_history = U_history.at[0, :, :].set(U_init)
    t_history = t_history.at[0].set(0.0)

    snapshot_idx = 1
    for step in range(1, n_steps + 1):
        U = advance_one_step(U, equation_manager)
        t += dt

        if step % save_interval == 0 and snapshot_idx < n_snapshots:
            U_history = U_history.at[snapshot_idx, :, :].set(U)
            t_history = t_history.at[snapshot_idx].set(t)
            snapshot_idx += 1

    return U_history, t_history


@runtime_check_array_sizes
def advance_one_step(
    U: Float[Array, "n_cells n_variables"],
    equation_manager: equation_manager_types.EquationManager,
) -> Float[Array, "n_cells n_variables"]:
    """Advance solution by one time step using Strang splitting.

    Strang splitting for source terms:
    1. Half-step source terms (dt/2)
    2. Full-step convection (dt)
    3. Full-stpe diffusion (dt)
    3. Half-step source terms (dt/2)

    Args:
        U: Conserved state [n_cells, n_variables]
        equation_manager: Contains all configuration

    Returns:
        U_next: Updated state [n_cells, n_variables]
    """
    dt = equation_manager.numerics_config.dt

    # Step 1: Half-step source terms (vibrational relaxation)
    S = source_terms.compute_source_terms(U, equation_manager)
    U = U + 0.5 * dt * S

    # Step 2: Full-step convection
    U_with_halo = apply_boundary_conditions(U, equation_manager)
    dU_dt_convective = compute_dU_dt_convective(U_with_halo, equation_manager)
    dU_dt_diffusive = compute_dU_dt_diffusive(U_with_halo, equation_manager)
    dU_dt = dU_dt_convective + dU_dt_diffusive
    U = integrate_in_time(U, dU_dt, equation_manager)

    # Step 3: Half-step source terms
    S = source_terms.compute_source_terms(U, equation_manager)
    U = U + 0.5 * dt * S

    return U


@runtime_check_array_sizes
def apply_boundary_conditions(
    U: Float[Array, "n_cells n_variables"],
    equation_manager: equation_manager_types.EquationManager,
) -> Float[Array, "n_cells_with_halo n_variables"]:
    """Apply boundary conditions to conserved variables.

    Args:
        U: Conserved state [n_cells, n_variables]
        equation_manager: Contains boundary condition type and n_halo_cells

    Returns:
        U_with_halo: State with ghost cells [n_cells + 2*n_halo, n_variables]
    """
    bc_type = equation_manager.boundary_condition
    n_ghosts = equation_manager.numerics_config.n_halo_cells
    return bc_module.apply_boundary_conditions(U, bc_type, n_ghosts)


@runtime_check_array_sizes
def compute_dU_dt_convective(
    U: Float[Array, "n_cells_with_halo n_variables"],
    equation_manager: equation_manager_types.EquationManager,
) -> Float[Array, "n_cells_interior n_variables"]:
    """Compute convective derivative dU/dt.

    Args:
        U: State with ghost cells [n_cells + 2*n_halo, n_variables]
        equation_manager: Contains dx and configuration

    Returns:
        dU_dt: Convective derivative [n_cells, n_variables]
    """
    U_L, U_R = compute_left_right_states(U, equation_manager)
    F = compute_convective_flux(U_L, U_R, equation_manager)

    n_halo = equation_manager.numerics_config.n_halo_cells
    n_cells = U.shape[0] - 2 * n_halo

    # Correct flux indexing for interior cells
    F_L = F[n_halo - 1 : n_halo + n_cells - 1, :]
    F_R = F[n_halo : n_halo + n_cells, :]

    return -1 / equation_manager.numerics_config.dx * (F_R - F_L)


@runtime_check_array_sizes
def compute_left_right_states(
    U: Float[Array, "n_cells_with_halo n_variables"],
    equation_manager: equation_manager_types.EquationManager,
) -> tuple[
    Float[Array, "n_interfaces n_variables"],
    Float[Array, "n_interfaces n_variables"],
]:
    """Compute left and right states at cell interfaces.

    Args:
        U: State with ghost cells [n_cells + 2*n_halo, n_variables]
        equation_manager: Contains spatial scheme configuration

    Returns:
        U_L, U_R: Left and right states at interfaces
    """
    if equation_manager.numerics_config.spatial_scheme == "first_order":
        U_L = U[:-1, :]
        U_R = U[1:, :]
    elif equation_manager.numerics_config.spatial_scheme == "muscl":
        delta_minus = U[1:-1, :] - U[:-2, :]  # U[i] - U[i-1]
        delta_plus = U[2:, :] - U[1:-1, :]  # U[i+1] - U[i]

        # Minmod slope limiter
        delta_center = jnp.where(
            delta_minus * delta_plus > 0,
            jnp.sign(delta_minus)
            * jnp.minimum(jnp.abs(delta_minus), jnp.abs(delta_plus)),
            0.0,
        )
        delta_U = jnp.zeros_like(U)
        delta_U = delta_U.at[1:-1, :].set(delta_center)

        U_L = U[:-1, :] + 0.5 * delta_U[:-1, :]
        U_R = U[1:, :] - 0.5 * delta_U[1:, :]
    else:
        raise ValueError(
            f"Unknown spatial scheme: {equation_manager.numerics_config.spatial_scheme}"
        )

    return U_L, U_R


@runtime_check_array_sizes
def compute_convective_flux(
    U_L: Float[Array, "n_interfaces n_variables"],
    U_R: Float[Array, "n_interfaces n_variables"],
    equation_manager: equation_manager_types.EquationManager,
) -> Float[Array, "n_interfaces n_variables"]:
    """Compute numerical flux at cell interfaces.

    Args:
        U_L: Left states at interfaces
        U_R: Right states at interfaces
        equation_manager: Contains flux scheme configuration

    Returns:
        F: Numerical flux at interfaces
    """
    if equation_manager.numerics_config.flux_scheme == "lax_friedrichs":
        raise NotImplementedError("Lax-Friedrichs flux not yet implemented")
    elif equation_manager.numerics_config.flux_scheme == "hllc":
        return solver.compute_flux(U_L, U_R, equation_manager)
    else:
        raise ValueError(
            f"Unknown flux scheme: {equation_manager.numerics_config.flux_scheme}"
        )


@runtime_check_array_sizes
def compute_dU_dt_diffusive(
    U: Float[Array, "n_cells+n_halo_cells n_variables"],
    equation_manager: equation_manager_types.EquationManager,
) -> Float[Array, "n_cells n_variables"]:
    pass


@runtime_check_array_sizes
def integrate_in_time(
    U: Float[Array, "n_cells n_variables"],
    dU_dt: Float[Array, "n_cells n_variables"],
    equation_manager: equation_manager_types.EquationManager,
) -> Float[Array, "n_cells n_variables"]:
    """Integrate in time using specified scheme.

    Args:
        U: Current state [n_cells, n_variables]
        dU_dt: Time derivative [n_cells, n_variables]
        equation_manager: Contains integrator scheme and dt

    Returns:
        U_next: Updated state [n_cells, n_variables]
    """
    dt = equation_manager.numerics_config.dt

    if equation_manager.numerics_config.integrator_scheme == "forward-euler":
        return U + dt * dU_dt

    elif equation_manager.numerics_config.integrator_scheme == "rk2":
        # Midpoint method (RK2)
        U_half = U + 0.5 * dt * dU_dt
        U_half_with_halo = apply_boundary_conditions(U_half, equation_manager)
        dU_dt_half_convective = compute_dU_dt_convective(
            U_half_with_halo, equation_manager
        )
        dU_dt_half_diffusive = compute_dU_dt_diffusive(
            U_half_with_halo, equation_manager
        )
        dU_dt_half = dU_dt_half_convective + dU_dt_half_diffusive

        return U + dt * dU_dt_half

    else:
        raise ValueError(
            f"Unknown integrator: {equation_manager.numerics_config.integrator_scheme}"
        )

