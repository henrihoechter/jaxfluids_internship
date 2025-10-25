import jax.numpy as jnp

from compressible_1d import numerics


def test_step_steady_state():
    U_initial = jnp.stack([jnp.ones(10), jnp.zeros(10), jnp.ones(10)], axis=0)
    U_expected = U_initial
    U_next = numerics.step(
        U_initial, n_ghost_cells=1, delta_x=0.1, delta_t=0.01, gamma=1.4
    )

    if not jnp.allclose(U_next, U_expected):
        raise ValueError("Step function does not preserve steady state solution.")
