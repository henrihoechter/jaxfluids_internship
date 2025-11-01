import pydantic

from jax import numpy as jnp
from jaxtyping import Float, Array

from compressible_1d import physics


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


def harten_lax_van_leer_contact(
    U_l: Float[Array, "3 ..."],
    U_r: Float[Array, "3 ..."],
    gamma: pydantic.NonNegativeFloat,
) -> Float[Array, "3 N"]:
    U_l_primitives = physics.to_primitives(U_field=U_l, gamma=gamma)
    U_r_primitives = physics.to_primitives(U_field=U_r, gamma=gamma)

    a_l = physics.a(U_l_primitives, gamma=gamma)
    a_r = physics.a(U_r_primitives, gamma=gamma)

    s_l = jnp.min(
        jnp.stack([U_l_primitives[1, :] - a_l, U_r_primitives[1, :] - a_r], axis=0),
        axis=0,
    )
    s_r = jnp.max(
        jnp.stack([U_l_primitives[1, :] + a_l, U_r_primitives[1, :] + a_r], axis=0),
        axis=0,
    )

    s_star = (
        U_r_primitives[2, :]
        - U_l_primitives[2, :]
        + U_l_primitives[0, :] * U_l_primitives[1, :] * (s_l - U_l_primitives[1, :])
        - U_r_primitives[0, :] * U_r_primitives[1, :] * (s_r - U_r_primitives[1, :])
    ) / (
        U_l_primitives[0, :] * (s_l - U_l_primitives[1, :])
        - U_r_primitives[0, :] * (s_r - U_r_primitives[1, :])
    )
    p_star = U_l_primitives[2, :] + U_l_primitives[0, :] * (
        s_l - U_l_primitives[1, :]
    ) * (s_star - U_l_primitives[1, :])

    rho_l = U_l_primitives[0, :] * (s_l - U_l_primitives[1, :]) / (s_l - s_star)
    rhoE_l = (
        (s_l - U_l_primitives[1, :]) * U_l[2, :]
        - U_l_primitives[2, :] * U_l_primitives[1, :]
        + p_star * s_star
    ) / (s_l - s_star)
    U_star_l = jnp.stack([rho_l, rho_l * s_l, rhoE_l])

    rho_r = U_r_primitives[0, :] * (s_r - U_r_primitives[1, :]) / (s_r - s_star)
    rhoE_r = (
        (s_r - U_r_primitives[1, :]) * U_r[2, :]
        - U_r_primitives[2, :] * U_r_primitives[1, :]
        + p_star * s_star
    ) / (s_r - s_star)
    U_star_r = jnp.stack([rho_r, rho_r * s_r, rhoE_r])

    F = jnp.empty(U_l.shape)
    F = jnp.where(s_l >= 0.0, flux_function(U_l, gamma=gamma), F)
    F = jnp.where(
        jnp.all(jnp.stack([s_l < 0.0, s_star >= 0.0], axis=0), axis=0),
        flux_function(U_l, gamma=gamma) + s_l * (U_star_l - U_l),
        F,
    )
    F = jnp.where(
        jnp.all(jnp.stack([s_star < 0.0, s_r >= 0.0], axis=0), axis=0),
        flux_function(U_r, gamma=gamma) + s_r * (U_star_r - U_r),
        F,
    )
    F = jnp.where(s_r < 0.0, flux_function(U_r, gamma=gamma), F)

    return F


