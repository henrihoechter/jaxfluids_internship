from jaxtyping import Float, Array
import jax.numpy as jnp
import jax
from dataclasses import dataclass

from compressible_1d import boundary_conditions, solver, diagnose, numerics


@dataclass
class Input:
    U_init: Float[Array, "3 n_cells"]
    """Initial flow field. The solver expects conserved, normalized variables."""
    gamma: float
    """Specific heat ratio."""
    boundary_condition: str
    """Type of boundary condition at domain edges."""
    solver_type: str
    """Type of solver to use. Currently available: 'lf', 'hllc', 'exact'."""
    delta_x: float
    """Spatial step size."""
    delta_t: float
    """Time step size."""
    n_steps: int
    """Number of time steps to simulate."""
    n_ghost_cells: int
    """Number of ghost cells on each side of the domain. Currently only 1 is supported."""
    is_debug: bool
    """If True, print some diagnostics during simulation."""
    is_abort: bool
    """If True, aborts the simulation if critical requirements are not met.
    
    For details, check `diagnose.check_all()`. 
    """


def step(
    U_field: Float[Array, "3 n_cells"],
    n_ghost_cells: int,
    delta_x: Float,
    delta_t: Float,
    gamma: Float,
    boundary_condition_type: str,
    solver_type: str,
) -> Float[Array, "3 n_cells"]:
    """Perform a single time step update of the flow field."""
    U_field_with_ghosts: Float[Array, "3 n_total_cells"] = (
        boundary_conditions.apply_boundary_condition(
            U_field,
            boundary_condition_type=boundary_condition_type,
            n_ghosts=n_ghost_cells,
        )
    )
    U_L = U_field_with_ghosts[:, :-1]
    U_R = U_field_with_ghosts[:, 1:]

    if solver_type == "lf":
        flux = solver.lax_friedrichs(U_L, U_R, gamma=gamma, diffusivity_scale=1.0)
    elif solver_type == "hllc":
        flux = solver.harten_lax_van_leer_contact(U_L, U_R, gamma=gamma)
    elif solver_type == "exact":
        flux = solver.exact_riemann(U_L, U_R, gamma=gamma)
    else:
        raise ValueError("Select solver.")

    N = U_field.shape[1]
    flux_L = flux[:, n_ghost_cells - 1 : n_ghost_cells + N - 1]
    flux_R = flux[:, n_ghost_cells : n_ghost_cells + N]

    dU_dt = -1 / delta_x * (flux_R - flux_L)

    return U_field + delta_t * dU_dt


def run(input: Input):
    """Run the CFD simulation based on the provided input parameters."""
    U_field = input.U_init

    step = jax.jit(
        numerics.step,
        static_argnames=("n_ghost_cells", "boundary_condition_type", "solver_type"),
    )

    U_solutions = jnp.empty(
        (input.U_init.shape[0], input.U_init.shape[1], input.n_steps)
    )

    U_solutions = U_solutions.at[:, :, 0].set(input.U_init)
    for i in range(input.n_steps):
        if input.is_debug:
            print(f"step: {i}")

        new_U_field = step(
            U_field,
            input.n_ghost_cells,
            input.delta_x,
            input.delta_t,
            input.gamma,
            boundary_condition_type=input.boundary_condition,
            solver_type=input.solver_type,
        )
        # diagnose solution
        diagnose.check_all(
            new_U_field, U_field, debug=input.is_debug, abort=input.is_abort
        )

        if i % 100 == 0:
            diagnose.live_diagnostics(new_U_field, i)

        U_solutions = U_solutions.at[:, :, i].set(new_U_field)
        U_field = new_U_field

    return U_solutions


def calculate_dt(
    U: Float[Array, "3 N"], gamma: float, delta_x: float, cmax: float = 1.0
) -> float:
    """Calculate time step width according to CFL condition."""

    def a(U: Float[Array, "3 N"], gamma: float):
        return jnp.sqrt(gamma * U[2, :] / U[0, :])

    return cmax * delta_x / jnp.max(jnp.abs(U[1, :]) + a(U, gamma))


def step_two_temperature(
    U_field: Float[Array, "..."],
    n_ghost_cells: int,
    delta_x: float,
    delta_t: float,
    species_list: list,
    config,
    boundary_condition_type: str,
) -> Float[Array, "..."]:
    """Perform a single time step update for two-temperature model.

    Uses Strang operator splitting:
    1. Half-step chemistry (dt/2)
    2. Full-step hydrodynamics (dt)
    3. Half-step chemistry (dt/2)

    Args:
        U_field: Conserved state vector [n_conserved, n_cells]
        n_ghost_cells: Number of ghost cells per side
        delta_x: Spatial step size [m]
        delta_t: Time step size [s]
        species_list: List of species data
        config: Two-temperature model configuration
        boundary_condition_type: Boundary condition type

    Returns:
        U_new: Updated state vector [n_conserved, n_cells]
    """
    from compressible_1d.two_temperature.source_terms import apply_chemistry_source
    from compressible_1d.two_temperature.solver_adapter import (
        compute_flux_two_temperature,
    )

    # Step 1: Half-step chemistry
    U_field = apply_chemistry_source(U_field, delta_t / 2.0, species_list, config)

    # Step 2: Full-step hydrodynamics
    U_field_with_ghosts = boundary_conditions.apply_boundary_condition(
        U_field,
        boundary_condition_type=boundary_condition_type,
        n_ghosts=n_ghost_cells,
    )
    U_L = U_field_with_ghosts[:, :-1]
    U_R = U_field_with_ghosts[:, 1:]

    flux = compute_flux_two_temperature(U_L, U_R, species_list, config)

    N = U_field.shape[1]
    flux_L = flux[:, n_ghost_cells - 1 : n_ghost_cells + N - 1]
    flux_R = flux[:, n_ghost_cells : n_ghost_cells + N]

    dU_dt = -1 / delta_x * (flux_R - flux_L)

    U_field = U_field + delta_t * dU_dt

    # Step 3: Half-step chemistry
    U_field = apply_chemistry_source(U_field, delta_t / 2.0, species_list, config)

    return U_field


def calculate_dt_two_temperature(
    U: Float[Array, "..."],
    species_list: list,
    config,
    delta_x: float,
    cmax: float = 0.4,
) -> float:
    """Calculate time step for two-temperature model with CFL condition.

    Accounts for both hydrodynamic CFL and chemistry stiffness.

    Args:
        U: Conserved state vector [n_conserved, N]
        species_list: List of species data
        config: Two-temperature model configuration
        delta_x: Spatial step size [m]
        cmax: CFL number (default: 0.4)

    Returns:
        dt: Time step [s]
    """
    from compressible_1d.two_temperature.source_terms import extract_primitives_from_U
    from compressible_1d.two_temperature.kinetics import compute_chemical_timescale
    from compressible_1d.two_temperature import thermodynamics

    # Extract primitives
    Y, rho, T, Tv, p = extract_primitives_from_U(U, species_list, config)

    # Velocity
    n_species = config.n_species
    u = U[n_species] / rho

    # Speed of sound
    a = thermodynamics.compute_speed_of_sound(rho, p, Y, T, Tv, species_list)

    # Hydrodynamic CFL time step
    dt_hydro = cmax * delta_x / jnp.max(jnp.abs(u) + a)

    # Chemical timescale
    tau_chem = compute_chemical_timescale(Y, rho, T, Tv, species_list, config)
    dt_chem = jnp.min(tau_chem) / 10.0  # Use 1/10 of chemical time

    # Vibrational relaxation timescale (CRITICAL for stability!)
    from compressible_1d.two_temperature.relaxation import (
        compute_mixture_relaxation_time,
    )

    tau_v = compute_mixture_relaxation_time(Y, T, p, tau_chem, species_list, config)
    dt_relax = jnp.min(tau_v) / 10.0  # Use 1/10 of relaxation time

    # Take minimum of all constraints
    dt = jnp.minimum(jnp.minimum(dt_hydro, dt_chem), dt_relax)

    return float(dt)


def run_two_temperature(
    U_init: Float[Array, "..."],
    delta_x: float,
    delta_t: float,
    n_steps: int,
    species_list: list,
    config,
    boundary_condition: str,
    n_ghost_cells: int = 1,
    is_debug: bool = False,
):
    """Run two-temperature CFD simulation.

    Args:
        U_init: Initial state [n_conserved, n_cells]
        delta_x: Spatial step size [m]
        delta_t: Time step size [s]
        n_steps: Number of time steps
        species_list: List of species data
        config: Two-temperature model configuration
        boundary_condition: Boundary condition type
        n_ghost_cells: Number of ghost cells per side
        is_debug: Whether to print debug info

    Returns:
        U_solutions: Solution at all time steps [n_conserved, n_cells, n_steps]
    """
    U_field = U_init

    # Convert species_list to tuple for JAX static_argnames (must be hashable)
    species_tuple = tuple(species_list)

    step = jax.jit(
        step_two_temperature,
        static_argnames=(
            "n_ghost_cells",
            "boundary_condition_type",
            "species_list",
            "config",
        ),
    )

    n_conserved, n_cells = U_init.shape
    U_solutions = jnp.empty((n_conserved, n_cells, n_steps))

    U_solutions = U_solutions.at[:, :, 0].set(U_init)

    for i in range(n_steps):
        if is_debug and i % 100 == 0:
            print(f"Step: {i}/{n_steps}")

        new_U_field = step(
            U_field,
            n_ghost_cells,
            delta_x,
            delta_t,
            species_tuple,
            config,
            boundary_condition,
        )

        U_solutions = U_solutions.at[:, :, i].set(new_U_field)
        U_field = new_U_field

    return U_solutions
