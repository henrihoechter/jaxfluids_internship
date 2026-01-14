"""Equation manager for 1D multi-species two-temperature Euler solver.

Implements the main solver loop with operator splitting for source terms.
"""

import warnings
from jaxtyping import Array, Float
import jax
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


def check_diffusive_cfl(
    U: Float[Array, "n_cells n_variables"],
    equation_manager: equation_manager_types.EquationManager,
) -> float | None:
    """Check diffusive CFL condition and warn if violated.

    The diffusive CFL condition requires:
        dt <= dx^2 / (2 * max(nu, alpha, D))

    where:
        nu = mu/rho is kinematic viscosity
        alpha = eta/(rho*cv) is thermal diffusivity
        D = max species diffusion coefficient

    Args:
        U: Conserved state [n_cells, n_variables]
        equation_manager: Contains configuration and collision integrals

    Returns:
        dt_diff: Maximum stable diffusive timestep [s], or None if inviscid
    """
    if equation_manager.collision_integrals is None:
        return None

    species_table = equation_manager.species
    collision_integrals = equation_manager.collision_integrals
    dx = equation_manager.numerics_config.dx
    dt = equation_manager.numerics_config.dt
    n_species = species_table.n_species

    # Extract primitives
    Y_s, rho, T, T_v, p = equation_manager_utils.extract_primitives_from_U(
        U, equation_manager
    )

    # Compute mass fractions
    rho_s = U[:, :n_species]
    c_s = rho_s / rho[:, None]

    # Build pair index matrix
    pair_indices = transport.build_pair_index_matrix(
        species_table.names, collision_integrals
    )

    # Interpolate collision integrals
    pi_omega_11 = transport.interpolate_collision_integral(
        T,
        collision_integrals.omega_11_2000K,
        collision_integrals.omega_11_4000K,
    )
    pi_omega_22 = transport.interpolate_collision_integral(
        T,
        collision_integrals.omega_22_2000K,
        collision_integrals.omega_22_4000K,
    )

    # Compute modified collision integrals
    M_s = species_table.molar_masses
    delta_1 = transport.compute_modified_collision_integral_1(
        T, M_s, M_s, pi_omega_11, pair_indices
    )
    delta_2 = transport.compute_modified_collision_integral_2(
        T, M_s, M_s, pi_omega_22, pair_indices
    )

    # Compute molar concentrations
    gamma_s = c_s / M_s

    # Compute transport properties
    mu = transport.compute_mixture_viscosity(T, gamma_s, M_s, delta_2)
    eta_t = transport.compute_translational_thermal_conductivity(
        T, gamma_s, M_s, delta_2
    )

    is_molecule = ~species_table.is_monoatomic.astype(bool)
    eta_r = transport.compute_rotational_thermal_conductivity(
        T, gamma_s, is_molecule, delta_1
    )

    # Total thermal conductivity
    eta = eta_t + eta_r

    # Compute diffusion coefficients
    D_sr = transport.compute_binary_diffusion_coefficient(T, p, delta_1)
    D_s = transport.compute_effective_diffusion_coefficient(gamma_s, M_s, D_sr)

    # Compute cv for thermal diffusivity
    cv_tr = thermodynamic_relations.compute_cv_tr(
        T, species_table
    )  # [n_species, n_cells]
    cv_mix = jnp.sum(c_s * cv_tr.T, axis=1)  # [n_cells]

    # Compute maximum stable diffusive timestep
    dt_diff = viscous_flux_module.compute_diffusive_cfl(rho, mu, eta, D_s, cv_mix, dx)

    # Convert to Python float for comparison
    dt_diff_value = float(dt_diff)

    if dt > dt_diff_value:
        warnings.warn(
            f"Diffusive CFL violated: dt={dt:.2e} > dt_diff={dt_diff_value:.2e}. "
            f"Consider reducing dt by factor {dt/dt_diff_value:.1f}.",
            RuntimeWarning,
            stacklevel=2,
        )

    return dt_diff_value


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
    # check_diffusive_cfl(U_init, equation_manager)

    # Time loop
    U = U_init
    t = 0.0
    n_steps = int(t_final / dt)
    n_snapshots = int(n_steps // save_interval) + 1

    n_cells, n_variables = U_init.shape
    U_history = jnp.zeros((n_snapshots, n_cells, n_variables))
    t_history = jnp.zeros(n_snapshots)

    U_history = U_history.at[0, :, :].set(U_init)
    t_history = t_history.at[0].set(0.0)

    advance_one_step_jitted = jax.jit(advance_one_step)

    snapshot_idx = 1
    for step in range(1, n_steps + 1):
        U = advance_one_step_jitted(U, equation_manager)
        t += dt

        if step % save_interval == 0 and snapshot_idx < n_snapshots:
            U_history = U_history.at[snapshot_idx, :, :].set(U)
            t_history = t_history.at[snapshot_idx].set(t)
            snapshot_idx += 1

    return U_history, t_history


@jax.jit(static_argnums=(2, 3))
def run_scan(
    U_init: Float[Array, "n_cells n_variables"],
    equation_manager: equation_manager_types.EquationManager,
    t_final: float,
    save_interval: int = 100,
) -> tuple[
    Float[Array, "n_snapshots n_cells n_variables"], Float[Array, "n_snapshots"]
]:
    """Run simulation from t=0 to t=t_final using jax.lax.scan."""
    dt = equation_manager.numerics_config.dt

    # Number of steps/snapshots (static given dt/t_final/save_interval)
    n_steps = int(t_final / dt)
    n_snapshots = int(n_steps // save_interval) + 1

    n_cells, n_variables = U_init.shape
    U_history0 = jnp.zeros((n_snapshots, n_cells, n_variables), dtype=U_init.dtype)
    t_history0 = jnp.zeros((n_snapshots,), dtype=jnp.result_type(dt, 0.0))

    # Write initial snapshot
    U_history0 = U_history0.at[0].set(U_init)
    t_history0 = t_history0.at[0].set(0.0)

    # Carry: (U, t, snapshot_idx, U_history, t_history)
    carry0 = (
        U_init,
        jnp.array(0.0, dtype=t_history0.dtype),
        jnp.array(1, dtype=jnp.int32),
        U_history0,
        t_history0,
    )

    def body(carry, step_idx):
        U, t, snap_i, U_hist, t_hist = carry

        # Advance
        U = advance_one_step(U, equation_manager)
        t = t + dt

        # Save every save_interval steps
        save = (step_idx % save_interval) == 0  # step_idx is 1..n_steps

        def do_save(args):
            U_, t_, snap_i_, U_hist_, t_hist_ = args
            U_hist_ = U_hist_.at[snap_i_].set(U_)
            t_hist_ = t_hist_.at[snap_i_].set(t_)
            return (U_, t_, snap_i_ + jnp.array(1, jnp.int32), U_hist_, t_hist_)

        def no_save(args):
            return args

        carry = jax.lax.cond(save, do_save, no_save, (U, t, snap_i, U_hist, t_hist))
        return carry, None

    # Scan over steps 1..n_steps (so modulo matches your original for-loop)
    carry_final, _ = jax.lax.scan(
        body, carry0, xs=jnp.arange(1, n_steps + 1, dtype=jnp.int32)
    )
    _, _, _, U_history, t_history = carry_final
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
    U: Float[Array, "n_cells_with_halo n_variables"],
    equation_manager: equation_manager_types.EquationManager,
) -> Float[Array, "n_cells_interior n_variables"]:
    """Compute diffusive derivative dU/dt using viscous flux.

    For inviscid case (collision_integrals=None), returns zeros.
    For viscous case, computes flux divergence.

    The viscous contribution follows from:
        ∂U/∂t + ∂F_c/∂x = ∂F_v/∂x + S

    where F_v is the viscous flux. The diffusive dU/dt is ∂F_v/∂x.

    Args:
        U: State with ghost cells [n_cells + 2*n_halo, n_variables]
        equation_manager: Contains configuration and collision integrals

    Returns:
        dU_dt: Diffusive derivative [n_cells, n_variables]
    """
    n_halo = equation_manager.numerics_config.n_halo_cells
    n_cells = U.shape[0] - 2 * n_halo
    n_vars = U.shape[1]
    dx = equation_manager.numerics_config.dx

    # Check if viscous terms are enabled
    if equation_manager.collision_integrals is None:
        return jnp.zeros((n_cells, n_vars))

    # Compute viscous flux at all interfaces (including ghost cells)
    F_v = viscous_flux_module.compute_viscous_flux(U, equation_manager)

    # F_v has shape [n_cells_with_halo - 1, n_variables]
    # We need fluxes at interior cell interfaces
    # Interior cells are from n_halo to n_halo + n_cells
    # Left interface of cell i is at index i-1 in F_v (0-indexed from first interface)
    # Right interface of cell i is at index i

    # For interior cells [n_halo, n_halo+n_cells), we need:
    # - Left flux: F_v[n_halo-1 : n_halo+n_cells-1]
    # - Right flux: F_v[n_halo : n_halo+n_cells]
    F_L = F_v[n_halo - 1 : n_halo + n_cells - 1, :]
    F_R = F_v[n_halo : n_halo + n_cells, :]

    # Flux divergence: dU/dt_diffusive = (F_R - F_L) / dx
    # Note: This has the correct sign because F_v already accounts for
    # the direction of diffusive transport
    dU_dt = (F_R - F_L) / dx

    return dU_dt


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
