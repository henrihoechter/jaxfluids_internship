from jaxtyping import Float, Array

from compressible_1d import boundary_conditions, solver, diagnose


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
def calculate_dt(
    U: Float[Array, "3 N"], gamma: float, delta_x: float, cmax: float = 1.0
) -> float:
    def a(U: Float[Array, "3 N"], gamma: float):
        return jnp.sqrt(gamma * U[2, :] / U[0, :])

    return cmax * delta_x / jnp.max(jnp.abs(U[1, :]) + a(U, gamma))
