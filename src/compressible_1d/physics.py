from jaxtyping import Float, Array
import jax.numpy as jnp


def a(U_primitive: Float[Array, "3 N"], gamma: float) -> Float[Array, "1 N"]:
    # assert not jnp.any(U_primitive[0,:] == 0.0)

    return jnp.sqrt(gamma * U_primitive[2, :] / U_primitive[0, :])


def to_conserved(
    U_field: Float[Array, "3 N"], rho_ref: float, p_ref: float, gamma: float
) -> Float[Array, "3 N"]:
    """Converts a primitive state vector to a normalized, conserved state vector.

    primitive:
    U[0]: mass density rho
    U[1]: velocity u
    U[2]: pressure p

    conserved:
    U[0]: normalized mass density
    U[1]: normalized momentum density
    U[2]: normalized energy density

    normalization by rho_ref, p_ref and ref. speed of sound

    E = e + 0.5 * u**2
    e = p / ((gamma -1) * rho)
    """
    rho = U_field[0, :]
    u = U_field[1, :]
    p = U_field[2, :]

    a_ref = jnp.sqrt(gamma * p_ref / rho_ref)  # ref. speed of sound [m/s]

    assert gamma - 1 > 0
    rho_norm = rho / rho_ref
    rhoU_norm = rho_norm * (u / a_ref)
    rhoE_norm = (p / p_ref) / (gamma - 1) + 0.5 * rho_norm * (u / a_ref) ** 2

    return jnp.stack([rho_norm, rhoU_norm, rhoE_norm], axis=0)


def to_primitives(U_field: Float[Array, "3 N"], gamma: float) -> Float[Array, "3 N"]:
    """Converts a normalized, conserved state vector to a primitive state vector.

    primitive:
    U[0]: mass density rho
    U[1]: velocity u
    U[2]: pressure p

    conserved:
    U[0]: normalized mass density
    U[1]: normalized momentum density
    U[2]: normalized energy density

    This function does not scale the variables back to physical units, but leaves them
    normalized.
    """

    rho_norm = U_field[0, :]
    rhoU_norm = U_field[1, :]
    rhoE_norm = U_field[2, :]

    u_norm = rhoU_norm / rho_norm
    E_norm = rhoE_norm / rho_norm

    # E = e + 1/2 u**2
    # p = (gamma - 1) rho e
    # p = (gamma - 1) rho (E - 1/2 u**2)
    p_norm = (gamma - 1) * rho_norm * (E_norm - 1 / 2 * u_norm**2)

    return jnp.stack([rho_norm, u_norm, p_norm], axis=0)
