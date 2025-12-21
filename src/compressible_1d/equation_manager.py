from jaxtyping import Array, Float
import jax.numpy as jnp

from compressible_1d import chemistry, numerics
from compressible_1d import equation_manager_types
from compressible_1d.diagnose import runtime_check_array_sizes
from compressible_1d.two_temperature import (
    solver_adapter as two_temperature_solver_adapter,
)


@runtime_check_array_sizes
def advance_one_step(
    U: Float[Array, "n_cells n_variables"],
    equation_manager: equation_manager_types.EquationManager,
) -> Float[Array, "n_cells n_variables"]:
    U_with_halo = apply_boundary_conditions(U, equation_manager)

    U_star = chemistry.apply_chemistry_source_terms(U_with_halo, equation_manager)

    dU_dt_convective = compute_dU_dt_convective(U_star, equation_manager)

    dU_dt_diffusive = compute_dU_dt_diffusive(U_star, equation_manager)

    dU_dt = dU_dt_convective + dU_dt_diffusive

    U_next = integrate_in_time(U_star, dU_dt, equation_manager)

    return U_next


@runtime_check_array_sizes
def apply_boundary_conditions(
    U: Float[Array, "n_cells n_variables"],
    equation_manager: equation_manager_types.EquationManager,
) -> Float[Array, "n_cells+n_halo_cells n_variables"]:
    pass


@runtime_check_array_sizes
def compute_dU_dt_convective(
    U: Float[Array, "n_cells+n_halo_cells n_variables"],
    equation_manager: equation_manager_types.EquationManager,
) -> Float[Array, "n_cells n_variables"]:
    U_L, U_R = compute_left_right_states(U, equation_manager)

    F = compute_convective_flux(U_L, U_R, equation_manager)

    n_halo = equation_manager.numerics_config.n_halo_cells
    F_L, F_R = F.at[n_halo - 1 : -n_halo + 1]

    return -1 / equation_manager.numerics_config.dx * (F_R - F_L)


@runtime_check_array_sizes
def compute_left_right_states(
    U: Float[Array, "n_cells+n_halo_cells n_variables"],
    equation_manager: equation_manager_types.EquationManager,
) -> tuple[
    Float[Array, "n_cells+n_halo_cells-1 n_variables"],
    Float[Array, "n_cells+n_halo_cells-1 n_variables"],
]:
    if equation_manager.numerics_config.spatial_scheme == "first_order":
        U_L = U.at[:-1]
        U_R = U.at[1:]
    elif equation_manager.numerics_config.spatial_scheme == "muscl":
        delta_minus = U[1:-1] - U.at[:-2]  # U[i] - U[i-1]
        delta_plus = U.at[2:] - U[1:-1]  # U[i+1] - U[i]

        delta_center = numerics.slope_limiter_minmod(
            delta_minus=delta_minus, delta_plus=delta_plus
        )
        delta_U = jnp.zeros_like(U)
        delta_U = delta_U.at[1:-1].set(delta_center)

        U_L = U.at[:-1] + 0.5 * delta_U.at[:-1]
        U_R = U.at[1:] - 0.5 * delta_U.at[1:]

    return U_L, U_R


@runtime_check_array_sizes
def compute_convective_flux(
    U_L: Float[Array, "n_cells+n_halo_cells n_variables"],
    U_R: Float[Array, "n_cells+n_halo_cells n_variables"],
    equation_manager: equation_manager_types.EquationManager,
) -> Float[Array, "n_cells+n_halo_cells n_variables"]:
    if equation_manager.numerics_config.flux_scheme == "lax_friedrichs":
        raise NotImplementedError
    elif equation_manager.numerics_config.flux_scheme == "hllc":
        # TODO: replace flux calculation
        return two_temperature_solver_adapter.compute_flux_two_temperature(
            U_L, U_R, equation_manager
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
    dt = equation_manager.numerics_config.dt

    if equation_manager.numerics_config.integrator_scheme == "forward-euler":
        return U + dt * dU_dt

    elif equation_manager.numerics_config.integrator_scheme == "rk2":
        U_half = U + 0.5 * dt * dU_dt
        dU_dt_half_convective = compute_dU_dt_convective(U_half, equation_manager)
        dU_dt_half_diffusive = compute_dU_dt_diffusive(U_half, equation_manager)
        dU_dt_half = dU_dt_half_convective + dU_dt_half_diffusive

        return U + 0.5 * dt * (dU_dt + dU_dt_half)
