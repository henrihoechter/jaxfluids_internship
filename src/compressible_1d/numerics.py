from jaxtyping import Float, Array
import jax.numpy as jnp
import jax
from dataclasses import dataclass

from compressible_1d import boundary_conditions, solver, diagnose, numerics


@dataclass
class Input:
    U_init: Float[Array, "3 n_cells"]
    gamma: float
    boundary_condition: str
    solver_type: str
    delta_x: float
    delta_t: float
    n_steps: int
    n_ghost_cells: int
    is_debug: bool
    is_abort: bool


def step(
    U_field: Float[Array, "3 n_cells"],
    n_ghost_cells: int,
    delta_x: Float,
    delta_t: Float,
    gamma: Float,
    boundary_condition_type: str,
    solver_type: str,
) -> Float[Array, "3 n_cells"]:
    U_field_with_ghosts: Float[Array, "3 n_cells+2*n_ghost_cells"] = (
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
    else:
        raise ValueError("Select solver.")

    N = U_field.shape[1]
    flux_L = flux[:, n_ghost_cells - 1 : n_ghost_cells + N - 1]
    flux_R = flux[:, n_ghost_cells : n_ghost_cells + N]

    dU_dt = -1 / delta_x * (flux_R - flux_L)

    return U_field + delta_t * dU_dt


def run(input: Input):
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
    def a(U: Float[Array, "3 N"], gamma: float):
        return jnp.sqrt(gamma * U[2, :] / U[0, :])

    return cmax * delta_x / jnp.max(jnp.abs(U[1, :]) + a(U, gamma))
