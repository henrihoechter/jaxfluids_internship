from jaxtyping import Float, Array

from compressible_1d import boundary_conditions, solver, diagnose


def step(
    U_field: Float[Array, "3 n_cells"],
    n_ghost_cells: int,
    delta_x: Float,
    delta_t: Float,
    gamma: Float,
) -> Float[Array, "3 n_cells"]:
    U_field_with_ghosts: Float[Array, "3 n_cells+2*n_ghost_cells"] = (
        boundary_conditions.apply_boundary_condition(
            U_field, boundary_condition_type="reflective", n_ghosts=n_ghost_cells
        )
    )
    # U_next_field: Float[Array, "3 n_cells+2*n_ghost_cells"] = jnp.zeros_like(
    #     U_field_with_ghosts
    # )
    U_L = U_field_with_ghosts[:, :-1]
    U_R = U_field_with_ghosts[:, 1:]
    flux = solver.lax_friedrichs(U_L, U_R, gamma=gamma, diffusivity_scale=0.0)
    N = U_field.shape[1]
    flux_L = flux[:, n_ghost_cells - 1 : n_ghost_cells + N - 1]
    flux_R = flux[:, n_ghost_cells : n_ghost_cells + N]

    dU_dt = -1 / delta_x * (flux_R - flux_L)

    U_next_field = U_field_with_ghosts.at[:, n_ghost_cells : N + n_ghost_cells].add(
        delta_t * dU_dt
    )

    # diagnose solution
    diagnose.check_all(U_next_field[:, n_ghost_cells : N + n_ghost_cells], U_field)

    return U_next_field[:, n_ghost_cells : N + n_ghost_cells]
