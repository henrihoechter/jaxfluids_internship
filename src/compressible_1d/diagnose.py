import jax.numpy as jnp

RTOL = 1e-10


def check_conservation(U, U_ref) -> None:
    if not jnp.allclose(jnp.sum(U[0, :]), jnp.sum(U_ref[0, :]), rtol=RTOL):
        raise ValueError("Total mass is not conserved.")

    if not jnp.allclose(jnp.sum(U[1, :]), jnp.sum(U_ref[1, :]), rtol=RTOL):
        raise ValueError("Total momentum is not conserved.")

    if not jnp.allclose(jnp.sum(U[2, :]), jnp.sum(U_ref[2, :]), rtol=RTOL):
        raise ValueError("Total energy is not conserved.")

    return None


def check_nonnegativity(U) -> None:
    if jnp.any(U[0, :] < 0.0):
        raise ValueError("Mass density negative.")

    if jnp.any(U[2, :] < 0.0):
        raise ValueError("Energy density negative.")


def check_nan_inf(U) -> None:
    if jnp.any(jnp.isnan(U)):
        raise ValueError("NaN values present in solution.")

    if jnp.any(jnp.isinf(U)):
        raise ValueError("Inf values present in solution.")


def check_all(U, U_ref) -> None:
    check_nan_inf(U)
    check_nonnegativity(U)
    check_conservation(U, U_ref)
