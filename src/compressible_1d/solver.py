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
    a_max = 0.0
    if diffusivity_scale > 0.0:
        raise NotImplementedError(
            "Diffusivity not yet implemented in Lax-Friedrichs Riemann solver."
        )
    return 0.5(
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
