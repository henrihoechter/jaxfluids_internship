import pydantic

from jax import numpy as jnp
from jaxtyping import Float, Array


def flux_function(
    U: Float[Array, "3 ..."], gamma: pydantic.NonNegativeFloat
) -> Float[Array, "3 ..."]:
    """
    U = [rho, rho*u, rho*E]T (normalized)
    F = [rho*u, rho*u^2 + p, (rho*E + p)*u]
    """
    rho = U[0]
    rho_u = U[1]
    rho_E = U[2]

    gamma = jnp.asarray(gamma, dtype=U.dtype)

    u = rho_u / rho
    p = (rho_E - 0.5 * rho * u**2) * (gamma - 1)

    F0 = rho_u
    F1 = rho_u * u + p
    F2 = (rho_E + p) * u

    return jnp.stack([F0, F1, F2], axis=0)


def lax_friedrichs(
    U_l: Float[Array, "3 ..."],
    U_r: Float[Array, "3 ..."],
    gamma: pydantic.NonNegativeFloat,
    diffusivity_scale: pydantic.NonNegativeFloat,
) -> Float[Array, "3 N"]:
    """Computes flux on cell boundary"""

    # a_max = max( |u_l| + c_l , |u_r| + c_r)
    a_max = jnp.maximum(
        jnp.abs(U_l[1, :] / U_l[0, :])
        + jnp.sqrt(
            gamma
            * (gamma - 1)
            * (U_l[2, :] / U_l[0, :] - 0.5 * (U_l[1, :] / U_l[0, :]) ** 2)
        ),
        jnp.abs(U_r[1, :] / U_r[0, :])
        + jnp.sqrt(
            gamma
            * (gamma - 1)
            * (U_r[2, :] / U_r[0, :] - 0.5 * (U_r[1, :] / U_r[0, :]) ** 2)
        ),
    )
    return 0.5 * (
        flux_function(U_r, gamma) + flux_function(U_l, gamma)
    ) - 0.5 * diffusivity_scale * a_max * (U_r - U_l)


# def hll(
#     U_l: physics_types.U_conserved, U_r: physics_types.U_conserved
# ) -> physics_types.U_conserved:
#     a_l = 0.0
#     a_r = 0.0
#     u_l = 0.0
#     s_l = jnp.min()

#     raise NotImplementedError
