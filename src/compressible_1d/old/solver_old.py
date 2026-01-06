import pydantic
from jaxtyping import Float, Array
from jax import numpy as jnp, lax

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

    a_max = jnp.maximum(a_l, a_r)
    s_l = jnp.minimum(U_l_primitives[1, :], U_r_primitives[1, :]) - a_max
    s_r = jnp.maximum(U_l_primitives[1, :], U_r_primitives[1, :]) + a_max

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
    U_star_l = jnp.stack([rho_l, rho_l * s_star, rhoE_l])

    rho_r = U_r_primitives[0, :] * (s_r - U_r_primitives[1, :]) / (s_r - s_star)
    rhoE_r = (
        (s_r - U_r_primitives[1, :]) * U_r[2, :]
        - U_r_primitives[2, :] * U_r_primitives[1, :]
        + p_star * s_star
    ) / (s_r - s_star)
    U_star_r = jnp.stack([rho_r, rho_r * s_star, rhoE_r])

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


# exact riemann ()


# ------------------------------------------------------------
# primitives <-> conserved, sound speed
# ------------------------------------------------------------
def _primitives_from_U(U: Float[Array, "3 ..."], gamma: float):
    rho = jnp.clip(U[0], 1e-16)
    u = U[1] / rho
    p = (U[2] - 0.5 * rho * u**2) * (gamma - 1.0)
    p = jnp.clip(p, 1e-16)
    return rho, u, p


def _U_from_rho_u_p(rho, u, p, gamma: float):
    rho = jnp.clip(rho, 1e-16)
    p = jnp.clip(p, 1e-16)
    rho_u = rho * u
    rho_E = p / (gamma - 1.0) + 0.5 * rho * u**2
    return jnp.stack([rho, rho_u, rho_E], axis=0)


def _a_from_rho_p(rho, p, gamma: float):
    return jnp.sqrt(gamma * p / jnp.clip(rho, 1e-16))


def _phi_side(p, rho_i, u_i, p_i, a_i, gamma: float):
    """Computes function and derivative that relates pressure to velocity change
    across a wave."""
    p = jnp.clip(p, 1e-16)
    pi = jnp.clip(p_i, 1e-16)
    rhoi = jnp.clip(rho_i, 1e-16)
    ai = jnp.clip(a_i, 1e-16)

    # Rarefaction branch
    expo = (gamma - 1.0) / (2.0 * gamma)
    rar_fac = (p / pi) ** expo
    f_rar = (2.0 * ai / (gamma - 1.0)) * (rar_fac - 1.0)
    df_rar = (1.0 / (rhoi * ai)) * (p / pi) ** (-(gamma + 1.0) / (2.0 * gamma))

    # Shock branch
    A = 2.0 / ((gamma + 1.0) * rhoi)
    B = (gamma - 1.0) / (gamma + 1.0) * pi
    sqrt_term = jnp.sqrt(A / (p + B))
    f_sh = (p - pi) * sqrt_term
    df_sh = sqrt_term * (1.0 - 0.5 * (p - pi) / (p + B))

    is_shock = p > pi
    f = jnp.where(is_shock, f_sh, f_rar)
    df = jnp.where(is_shock, df_sh, df_rar)
    return f, df


def _pstar_initial(rhoL, uL, pL, aL, rhoR, uR, pR, aR, gamma: float):
    p_pv = 0.5 * (pL + pR) - 0.125 * (uR - uL) * (rhoL + rhoR) * (aL + aR)
    p_pv = jnp.maximum(1e-16, p_pv)

    pmin = jnp.minimum(pL, pR)
    pmax = jnp.maximum(pL, pR)

    # Two-rarefaction estimate
    alpha = (gamma - 1.0) / (2.0 * gamma)
    denom_tr = aL * jnp.power(jnp.maximum(pL, 1e-16), -alpha) + aR * jnp.power(
        jnp.maximum(pR, 1e-16), -alpha
    )
    base_tr = (aL + aR - 0.5 * (gamma - 1.0) * (uR - uL)) / jnp.clip(denom_tr, 1e-16)
    p_tr = jnp.maximum(1e-16, jnp.power(jnp.maximum(base_tr, 1e-16), 1.0 / alpha))

    # Two-shock estimate
    AL = 2.0 / ((gamma + 1.0) * jnp.clip(rhoL, 1e-16))
    BL = (gamma - 1.0) / (gamma + 1.0) * pL
    AR = 2.0 / ((gamma + 1.0) * jnp.clip(rhoR, 1e-16))
    BR = (gamma - 1.0) / (gamma + 1.0) * pR
    gL = jnp.sqrt(AL / (pL + BL))
    gR = jnp.sqrt(AR / (pR + BR))
    p_ts = (gL * pL + gR * pR - (uR - uL)) / jnp.clip(gL + gR, 1e-16)
    p_ts = jnp.maximum(1e-16, p_ts)

    p0 = jnp.where(p_pv < pmin, p_tr, p_pv)
    p0 = jnp.where(p_pv > pmax, p_ts, p0)
    return jnp.maximum(1e-16, p0)


def _solve_pstar(rhoL, uL, pL, aL, rhoR, uR, pR, aR, gamma: float):
    """Solves for p* using Newton-Raphson with fixed number of iterations.

    Goal: find p* at the location where the left and the right wave predict the same
    velocity. pressure and velocity must be continous across the discontinuity.
    """
    p0 = _pstar_initial(rhoL, uL, pL, aL, rhoR, uR, pR, aR, gamma)
    max_iter = 25
    tol = 1e-8

    def body(i, carry):
        p = carry  # current guess for p*
        fL, dfL = _phi_side(p, rhoL, uL, pL, aL, gamma)  # wave contribution left
        fR, dfR = _phi_side(p, rhoR, uR, pR, aR, gamma)  # wave contribution right
        phi = fL + fR + (uR - uL)  # residual
        dphi = dfL + dfR  # residual derivative
        dp = phi / jnp.clip(dphi, 1e-16)  # step
        pnew = jnp.maximum(1e-16, p - dp)  # update

        # freeze converged entries to avoid oscillations across iterations
        done = (jnp.abs(phi) < tol) | (jnp.abs(dp) / jnp.maximum(pnew, 1e-16) < 1e-8)
        p = jnp.where(done, p, pnew)
        return p

    pstar = lax.fori_loop(0, max_iter, body, p0)
    return jnp.maximum(1e-16, pstar)


def _ustar(pstar, rhoL, uL, pL, aL, rhoR, uR, pR, aR, gamma: float):
    fL, _ = _phi_side(pstar, rhoL, uL, pL, aL, gamma)
    fR, _ = _phi_side(pstar, rhoR, uR, pR, aR, gamma)
    return 0.5 * (uL + uR + fR - fL)


def _left_at_interface(pstar, ustar, rhoL, uL, pL, aL, gamma: float):
    is_shock = pstar > pL

    # Left shock speed S_L
    SL = uL - aL * jnp.sqrt(
        0.5 * (gamma + 1.0) / gamma * (pstar / pL) + 0.5 * (gamma - 1.0) / gamma
    )

    # Left rarefaction head/tail
    SHL = uL - aL
    a_starL = aL * (pstar / pL) ** ((gamma - 1.0) / (2.0 * gamma))
    STL = ustar - a_starL

    # Left star density (post-wave)
    beta = (gamma - 1.0) / (gamma + 1.0)
    rho_sh = rhoL * ((pstar / pL + beta) / (beta * pstar / pL + 1.0))
    rho_ra = rhoL * (pstar / pL) ** (1.0 / gamma)
    rho_star = jnp.where(is_shock, rho_sh, rho_ra)
    UL_star = _U_from_rho_u_p(rho_star, ustar, pstar, gamma)

    # Left state
    UL_L = _U_from_rho_u_p(rhoL, uL, pL, gamma)

    # Fan state at ξ=0 (safe eval with clips)
    # Toro rarefaction self-similar relations evaluated at xi=0
    # These expressions are only USED when inside the fan; safe to clip bases.
    u_fan = (2.0 / (gamma + 1.0)) * (aL + 0.5 * (gamma - 1.0) * uL)
    a_fan = (2.0 / (gamma + 1.0)) * (aL + 0.5 * (gamma - 1.0) * (uL - 0.0))
    a_fan = jnp.clip(a_fan, 1e-16)
    rho_fan = rhoL * (a_fan / jnp.clip(aL, 1e-16)) ** (2.0 / (gamma - 1.0))
    p_fan = pL * (a_fan / jnp.clip(aL, 1e-16)) ** (2.0 * gamma / (gamma - 1.0))
    UL_fan = _U_from_rho_u_p(rho_fan, u_fan, p_fan, gamma)

    # Assemble by regions with jnp.where
    # Shock case: if 0 <= SL => region 1 (left), else region 2* (star)
    UL_if_shock = jnp.where(0.0 <= SL, UL_L, UL_star)

    # Rarefaction case:
    #  - if 0 <= SHL -> left state
    #  - elif 0 >= STL -> star state
    #  - else -> fan state
    UL_if_rare = jnp.where(0.0 <= SHL, UL_L, jnp.where(0.0 >= STL, UL_star, UL_fan))

    UL_if = jnp.where(is_shock, UL_if_shock, UL_if_rare)
    return UL_if


def _right_at_interface(pstar, ustar, rhoR, uR, pR, aR, gamma: float):
    is_shock = pstar > pR

    # Right shock speed S_R
    SR = uR + aR * jnp.sqrt(
        0.5 * (gamma + 1.0) / gamma * (pstar / pR) + 0.5 * (gamma - 1.0) / gamma
    )

    # Right rarefaction head/tail
    SHR = uR + aR
    a_starR = aR * (pstar / pR) ** ((gamma - 1.0) / (2.0 * gamma))
    STR = ustar + a_starR

    # Right star density
    beta = (gamma - 1.0) / (gamma + 1.0)
    rho_sh = rhoR * ((pstar / pR + beta) / (beta * pstar / pR + 1.0))
    rho_ra = rhoR * (pstar / pR) ** (1.0 / gamma)
    rho_star = jnp.where(is_shock, rho_sh, rho_ra)
    UR_star = _U_from_rho_u_p(rho_star, ustar, pstar, gamma)

    # Right state
    UR_R = _U_from_rho_u_p(rhoR, uR, pR, gamma)

    # Fan state at ξ=0 (safe eval with clips)
    u_fan = (2.0 / (gamma + 1.0)) * (-aR + 0.5 * (gamma - 1.0) * uR)
    a_fan = (2.0 / (gamma + 1.0)) * (-aR + 0.5 * (gamma - 1.0) * (0.0 - uR))
    a_fan = jnp.clip(a_fan, 1e-16)
    rho_fan = rhoR * (a_fan / jnp.clip(aR, 1e-16)) ** (2.0 / (gamma - 1.0))
    p_fan = pR * (a_fan / jnp.clip(aR, 1e-16)) ** (2.0 * gamma / (gamma - 1.0))
    UR_fan = _U_from_rho_u_p(rho_fan, u_fan, p_fan, gamma)

    # Assemble by regions
    UR_if_shock = jnp.where(0.0 >= SR, UR_R, UR_star)
    UR_if_rare = jnp.where(0.0 >= SHR, UR_R, jnp.where(0.0 <= STR, UR_star, UR_fan))
    UR_if = jnp.where(is_shock, UR_if_shock, UR_if_rare)
    return UR_if


def exact_riemann(
    U_l: Float[Array, "3 ..."],
    U_r: Float[Array, "3 ..."],
    gamma: float,
) -> Float[Array, "3 ..."]:
    """
    Exact Riemann solver for 1D Euler (Toro). Returns flux at each interface.
    U_l, U_r: [3, N]
    """
    rhoL, uL, pL = _primitives_from_U(U_l, gamma)
    rhoR, uR, pR = _primitives_from_U(U_r, gamma)
    aL = _a_from_rho_p(rhoL, pL, gamma)
    aR = _a_from_rho_p(rhoR, pR, gamma)

    pstar = _solve_pstar(rhoL, uL, pL, aL, rhoR, uR, pR, aR, gamma)
    ustar = _ustar(pstar, rhoL, uL, pL, aL, rhoR, uR, pR, aR, gamma)

    UL_if = _left_at_interface(pstar, ustar, rhoL, uL, pL, aL, gamma)
    UR_if = _right_at_interface(pstar, ustar, rhoR, uR, pR, aR, gamma)

    # Choose which side provides the interface state at ξ=0 by sign of u*
    U_if = jnp.where(ustar[None, :] >= 0.0, UL_if, UR_if)
    return flux_function(U_if, gamma)


def hllc_two_temperature(
    U_l: Float[Array, "n_conserved ..."],
    U_r: Float[Array, "n_conserved ..."],
    species_list: list,
    config,
) -> Float[Array, "n_conserved ..."]:
    """HLLC Riemann solver for two-temperature multi-species system.

    Wrapper function that calls the solver implementation from solver_adapter.

    Args:
        U_l: Left state [n_conserved, ...]
        U_r: Right state [n_conserved, ...]
        species_list: List of species data
        config: Two-temperature model configuration

    Returns:
        F: Numerical flux [n_conserved, ...]
    """
    from compressible_1d.two_temperature.solver_adapter import (
        compute_flux_two_temperature,
    )

    return compute_flux_two_temperature(U_l, U_r, species_list, config)
