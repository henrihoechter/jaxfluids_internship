import jax.numpy as jnp

from compressible_1d import physics


def test_to_conserved():
    U1 = jnp.array([1.225, 0.0, 1e5])
    U2 = jnp.array([0.4, 10.0, 1.5e5])
    gamma = 1.4

    U_expected = jnp.stack(
        [jnp.array([1.0, 0.0, 2.5]), jnp.array([0.32653, 0.0096589, 3.750143])], axis=1
    )

    U_conserved = physics.to_conserved(
        U_field=jnp.stack([U1, U2], axis=1), rho_ref=1.225, p_ref=1e5, gamma=gamma
    )

    if not jnp.allclose(U_conserved, U_expected):
        raise ValueError("Conversion from primitive to conserved state vector failed.")


def test_to_primitive():
    U1 = jnp.array([1.0, 0.0, 2.5])
    U2 = jnp.array([0.32653, 0.0096589, 3.750143])
    gamma = 1.4

    U_expected = jnp.stack(
        [jnp.array([1.0, 0.0, 1.0]), jnp.array([0.32653, 0.02958, 1.5])], axis=1
    )

    U_primitive = physics.to_primitives(
        U_field=jnp.stack([U1, U2], axis=1), gamma=gamma
    )

    if not jnp.allclose(U_primitive, U_expected, atol=1e-5):
        raise ValueError("Conversion from conserved to primitive state vector failed.")
