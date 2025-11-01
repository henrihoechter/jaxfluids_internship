import jax.numpy as jnp

from compressible_1d import numerics


def test_step_steady_state():
    U_initial = jnp.stack([jnp.ones(10), jnp.zeros(10), jnp.ones(10)], axis=0)
    U_expected = U_initial
    U_next = numerics.step(
        U_initial,
        n_ghost_cells=1,
        delta_x=0.1,
        delta_t=0.01,
        gamma=1.4,
        boundary_condition_type="periodic",
        solver_type="lf",
    )

    if not jnp.allclose(U_next, U_expected):
        raise ValueError("Step function does not preserve steady state solution.")


def test_calculate_dt():
    U = jnp.array([1.225, 10.0, 1e5])[:, jnp.newaxis]
    gamma = 1.4
    delta_x = 1.0

    expected_delta_t = 2.873053814e-3

    delta_t = numerics.calculate_dt(U, gamma, delta_x, cmax=1.0)
    print(delta_t)

    if not jnp.allclose(delta_t, expected_delta_t):
        raise ValueError("delta_t was not computed correctly.")
