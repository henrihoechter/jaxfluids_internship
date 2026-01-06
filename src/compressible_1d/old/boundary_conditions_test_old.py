import jax.numpy as jnp

from compressible_1d import boundary_conditions


def test_periodic_bc():
    U_test = jnp.linspace(jnp.zeros(3), jnp.ones(3), num=10, axis=-1)
    U_expected = jnp.concatenate(
        [jnp.ones(3)[:, jnp.newaxis], U_test, jnp.zeros(3)[:, jnp.newaxis]], axis=-1
    )

    U_with_ghosts = boundary_conditions.apply_boundary_condition(
        U_test, boundary_condition_type="periodic", n_ghosts=1
    )

    if U_with_ghosts.shape[1] != U_test.shape[1] + 2:
        raise ValueError("Field with ghost cells does not have expected length.")

    if not jnp.array_equal(U_with_ghosts, U_expected):
        raise ValueError("Returned array has values different to expected result.")


def test_transmissive_bc():
    U_test = jnp.linspace(jnp.zeros(3), jnp.ones(3), num=10, axis=-1)
    U_expected = jnp.concatenate(
        [jnp.zeros(3)[:, jnp.newaxis], U_test, jnp.ones(3)[:, jnp.newaxis]], axis=-1
    )

    U_with_ghosts = boundary_conditions.apply_boundary_condition(
        U_test, boundary_condition_type="transmissive", n_ghosts=1
    )

    if U_with_ghosts.shape[1] != U_test.shape[1] + 2:
        raise ValueError("Field with ghost cells does not have expected length.")

    if not jnp.array_equal(U_with_ghosts, U_expected):
        raise ValueError("Returned array has values different to expected result.")


def test_reflective_bc():
    U_test = jnp.linspace(jnp.zeros(3), jnp.ones(3), num=10, axis=-1)
    U_expected = jnp.concatenate(
        [
            jnp.zeros(3)[:, jnp.newaxis],
            U_test,
            jnp.stack([1.0, -1.0, 1.0])[:, jnp.newaxis],
        ],
        axis=-1,
    )

    U_with_ghosts = boundary_conditions.apply_boundary_condition(
        U_test, boundary_condition_type="reflective", n_ghosts=1
    )

    if U_with_ghosts.shape[1] != U_test.shape[1] + 2:
        raise ValueError("Field with ghost cells does not have expected length.")

    if not jnp.array_equal(U_with_ghosts, U_expected):
        raise ValueError("Returned array has values different to expected result.")
