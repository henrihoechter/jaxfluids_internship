import pydantic

from jax import numpy as jnp
from jax import lax
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


EPS = 1e-14
P_MIN = 1e-12
P_MAX = 1e12

def _cons_from_prim(rho, u, p, gamma):
    mom = rho * u
    E = p / (gamma - 1.0) + 0.5 * rho * u * u
    return jnp.stack([rho, mom, E], axis=0)

# --- Branch functions -------------------------------------------------------

def _f_rare(p, rho, pK, aK, gamma):
    # f(p) and df/dp for rarefaction branch (vectorized)
    pr = jnp.maximum(p, P_MIN) / pK
    expo = (gamma - 1.0) / (2.0 * gamma)
    f = (2.0 * aK / (gamma - 1.0)) * (pr**expo - 1.0)
    df = (aK / (gamma * pK)) * pr**(expo - 1.0)
    return f, df

def _f_shock(p, rho, pK, gamma):
    # f(p) and df/dp for shock branch (vectorized)
    A = 2.0 / ((gamma + 1.0) * rho)
    B = (gamma - 1.0) / (gamma + 1.0) * pK
    denom = jnp.maximum(p + B, EPS)
    root = jnp.sqrt(A / denom)
    f = (p - pK) * root
    df = root * (1.0 - 0.5 * (p - pK) / denom)
    return f, df

def _phi_and_dphi(p, rhoL, uL, pL, aL, rhoR, uR, pR, aR, gamma):
    # choose branch per interface
    left_is_rare  = p <= pL
    right_is_rare = p <= pR

    fL_r, dL_r = _f_rare(p, rhoL, pL, aL, gamma)
    fL_s, dL_s = _f_shock(p, rhoL, pL, gamma)
    fL  = jnp.where(left_is_rare,  fL_r,  fL_s)
    dL  = jnp.where(left_is_rare,  dL_r,  dL_s)

    fR_r, dR_r = _f_rare(p, rhoR, pR, aR, gamma)
    fR_s, dR_s = _f_shock(p, rhoR, pR, gamma)
    fR  = jnp.where(right_is_rare, fR_r,  fR_s)
    dR  = jnp.where(right_is_rare, dR_r,  dR_s)

    phi  = fL + fR + (uR - uL)
    dphi = dL + dR
    return phi, dphi

def _pvrs_guess(rhoL, uL, pL, aL, rhoR, uR, pR, aR, gamma):
    cbar = 0.5 * (aL + aR)
    pavg = 0.5 * (pL + pR) - 0.125 * (uR - uL) * (rhoL + rhoR) * cbar
    return jnp.clip(pavg, P_MIN, P_MAX)

def _two_rarefaction_guess(rhoL, uL, pL, aL, rhoR, uR, pR, aR, gamma):
    expo = (2.0 * gamma) / (gamma - 1.0)
    alphaL = aL * pL ** (-(gamma - 1.0) / (2.0 * gamma))
    alphaR = aR * pR ** (-(gamma - 1.0) / (2.0 * gamma))
    term = (aL + aR) - 0.5 * (gamma - 1.0) * (uR - uL)
    p = (term / (alphaL + alphaR)) ** expo
    return jnp.clip(p, P_MIN, P_MAX)

def _two_shock_guess(rhoL, uL, pL, rhoR, uR, pR, gamma):
    GL = jnp.sqrt( (2.0 / ((gamma + 1.0) * rhoL)) / jnp.maximum((pL * (gamma - 1.0) / (gamma + 1.0)), EPS) )
    GR = jnp.sqrt( (2.0 / ((gamma + 1.0) * rhoR)) / jnp.maximum((pR * (gamma - 1.0) / (gamma + 1.0)), EPS) )
    p = (GL * pL + GR * pR - (uR - uL)) / (GL + GR)
    return jnp.clip(p, P_MIN, P_MAX)

def _vacuum_flag(uL, aL, uR, aR, gamma):
    return (uR - uL) >= (2.0 / (gamma - 1.0)) * (aL + aR)

# --- Newton iteration (vectorized) -----------------------------------------

def _solve_p_star(rhoL, uL, pL, aL, rhoR, uR, pR, aR, gamma):
    # initial guess
    p0 = _pvrs_guess(rhoL, uL, pL, aL, rhoR, uR, pR, aR, gamma)

    def newton_body(p, _):
        phi, dphi = _phi_and_dphi(p, rhoL, uL, pL, aL, rhoR, uR, pR, aR, gamma)
        step = phi / jnp.where(jnp.abs(dphi) > EPS, dphi, jnp.sign(dphi) * EPS)
        p_new = jnp.clip(p - step, P_MIN, P_MAX)
        return p_new, None

    # fixed number of iterations for JIT-friendliness
    p, _ = lax.scan(newton_body, p0, xs=None, length=20)

    # (optional) single safeguard iteration with two-shock/two-rarefaction mix
    two_shock = _two_shock_guess(rhoL, uL, pL, rhoR, uR, pR, gamma)
    two_rare  = _two_rarefaction_guess(rhoL, uL, pL, aL, rhoR, uR, pR, aR, gamma)
    # if PVRS fell way below both pL and pR, favor two-shock; else two-rarefaction
    p_seed = jnp.where(p0 < jnp.minimum(pL, pR), two_shock, two_rare)

    def refine_once(p):
        phi, dphi = _phi_and_dphi(p, rhoL, uL, pL, aL, rhoR, uR, pR, aR, gamma)
        return jnp.clip(p - phi / jnp.where(jnp.abs(dphi) > EPS, dphi, jnp.sign(dphi) * EPS), P_MIN, P_MAX)

    p_ref = refine_once(p_seed)
    # pick the better of (p, p_ref) by |phi|
    phi_p, _ = _phi_and_dphi(p,      rhoL, uL, pL, aL, rhoR, uR, pR, aR, gamma)
    phi_q, _ = _phi_and_dphi(p_ref,  rhoL, uL, pL, aL, rhoR, uR, pR, aR, gamma)
    p_star = jnp.where(jnp.abs(phi_q) < jnp.abs(phi_p), p_ref, p)
    return p_star

# --- Wave speeds and star densities ----------------------------------------

def _star_state(p_star, rhoL, uL, pL, rhoR, uR, pR, gamma):
    # fL/fR at p* for u*
    def f_side(p, rho, u, pK):
        # mixture of rare/shock at p vs pK
        # Need aK only for rarefaction
        return p

    # left branch
    aL = jnp.sqrt(gamma * pL / rhoL)
    aR = jnp.sqrt(gamma * pR / rhoR)

    fL_r, _ = _f_rare (p_star, rhoL, pL, aL, gamma)
    fL_s, _ = _f_shock(p_star, rhoL, pL, gamma)
    fR_r, _ = _f_rare (p_star, rhoR, pR, aR, gamma)
    fR_s, _ = _f_shock(p_star, rhoR, pR, gamma)

    left_is_rare  = p_star <= pL
    right_is_rare = p_star <= pR

    fL = jnp.where(left_is_rare,  fL_r, fL_s)
    fR = jnp.where(right_is_rare, fR_r, fR_s)

    u_star = 0.5 * (uL + uR + fR - fL)

    # star densities
    # rarefaction: rho* = rho (p*/p)^{1/gamma}
    # shock:       rho* = rho * (p*/p + (γ-1)/(γ+1)) / ((γ-1)/(γ+1) * p*/p + 1)
    ratioL = p_star / pL
    ratioR = p_star / pR
    rrL = rhoL * ratioL ** (1.0 / gamma)
    rrR = rhoR * ratioR ** (1.0 / gamma)
    z = (gamma - 1.0) / (gamma + 1.0)
    rsL = rhoL * (ratioL + z) / (z * ratioL + 1.0)
    rsR = rhoR * (ratioR + z) / (z * ratioR + 1.0)

    rhoL_star = jnp.where(left_is_rare,  rrL, rsL)
    rhoR_star = jnp.where(right_is_rare, rrR, rsR)

    return u_star, rhoL_star, rhoR_star

def _wave_speeds(p_star, u_star, rhoL, uL, pL, rhoR, uR, pR, gamma):
    aL = jnp.sqrt(gamma * pL / rhoL)
    aR = jnp.sqrt(gamma * pR / rhoR)

    left_is_rare  = p_star <= pL
    right_is_rare = p_star <= pR

    # left
    # rarefaction: head uL - aL, tail u* - aL*
    aL_star = jnp.sqrt(gamma * p_star / (rhoL * ((p_star/pL) ** (1.0/gamma))))  # uses rhoL*
    # but we don’t have rhoL* here—compute directly:
    rhoL_star_rare = rhoL * (p_star / pL) ** (1.0 / gamma)
    aL_star = jnp.sqrt(gamma * p_star / rhoL_star_rare)
    SL_head = uL - aL
    SL_tail = u_star - aL_star

    # shock: sL = uL - sqrt(AL (p* + BL))
    AL = 2.0 / ((gamma + 1.0) * rhoL)
    BL = (gamma - 1.0) / (gamma + 1.0) * pL
    SL_shock = uL - jnp.sqrt(AL * jnp.maximum(p_star + BL, EPS))

    S_L = jnp.where(left_is_rare, SL_head, SL_shock)
    S_L_tail = jnp.where(left_is_rare, SL_tail, jnp.nan)  # nan indicates no tail

    # right
    rhoR_star_rare = rhoR * (p_star / pR) ** (1.0 / gamma)
    aR_star = jnp.sqrt(gamma * p_star / rhoR_star_rare)
    SR_head = uR + aR
    SR_tail = u_star + aR_star

    AR = 2.0 / ((gamma + 1.0) * rhoR)
    BR = (gamma - 1.0) / (gamma + 1.0) * pR
    SR_shock = uR + jnp.sqrt(AR * jnp.maximum(p_star + BR, EPS))

    S_R = jnp.where(right_is_rare, SR_head, SR_shock)
    S_R_tail = jnp.where(right_is_rare, SR_tail, jnp.nan)

    return S_L, S_L_tail, u_star, S_R_tail, S_R

# --- Exact Riemann flux at ξ = 0 -------------------------------------------

def exact_riemann(U_l, U_r, gamma):
    """
    Vectorized exact Riemann solver for Euler ideal gas.
    Inputs:
      U_l, U_r : (3, N)
    Returns:
      F        : (3, N) numerical flux at interfaces
    """
    # primitives
    primL = physics.to_primitives(U_field=U_l, gamma=gamma)  # (rhoL, uL, pL)
    primR = physics.to_primitives(U_field=U_r, gamma=gamma)  # (rhoR, uR, pR)
    rhoL, uL, pL = primL[0, :], primL[1, :], primL[2, :]
    rhoR, uR, pR = primR[0, :], primR[1, :], primR[2, :]
    aL = jnp.sqrt(gamma * pL / rhoL)
    aR = jnp.sqrt(gamma * pR / rhoR)

    # optional vacuum guard: fall back to HLLC if vacuum criterion is violated
    vac = _vacuum_flag(uL, aL, uR, aR, gamma)

    # p*, u*, rho* left/right
    p_star = _solve_p_star(rhoL, uL, pL, aL, rhoR, uR, pR, aR, gamma)
    u_star, rhoL_star, rhoR_star = _star_state(p_star, rhoL, uL, pL, rhoR, uR, pR, gamma)

    # wave speeds
    S_L, S_L_tail, S_M, S_R_tail, S_R = _wave_speeds(p_star, u_star, rhoL, uL, pL, rhoR, uR, pR, gamma)

    # Sampling at xi = 0
    left_is_rare  = p_star <= pL
    right_is_rare = p_star <= pR

    # left-side state when S_M >= 0 controls the interface
    # left rarefaction cases:
    #   if 0 <= S_L_head -> left state
    #   elif 0 >= S_L_tail -> left star
    #   else -> inside fan at xi=0
    rhoL_fan = lambda aH, aT: None  # only name placeholder

    # inside left fan at xi=0:
    # u(0) = uL + 2/(γ+1) * (0 - (uL - aL)) = uL + 2 aL/(γ+1)
    uL_in = uL + 2.0 * aL / (gamma + 1.0)
    aL_in = aL - 0.5 * (gamma - 1.0) * (uL_in - uL)
    # but the invariant formula is simpler via standard derivation:
    # Use canonical formulas:
    aL_in = jnp.maximum((2.0/(gamma+1.0))*(aL + 0.5*(gamma-1.0)*uL), EPS)  # safe
    # Use the textbook fan relations directly:
    # We prefer explicit fan formulas below to avoid ambiguity:
    def left_fan_state():
        u = uL + 2.0/(gamma+1.0) * (0.0 - (uL - aL))  # xi=0
        a = aL - 0.5*(gamma-1.0)*(u - uL)
        p = pL * (a / aL) ** (2.0 * gamma / (gamma - 1.0))
        rho = rhoL * (a / aL) ** (2.0 / (gamma - 1.0))
        return rho, u, p

    def right_fan_state():
        u = uR + 2.0/(gamma+1.0) * ((uR + aR) - 0.0)  # xi=0
        a = aR - 0.5*(gamma-1.0)*(uR - u)
        p = pR * (a / aR) ** (2.0 * gamma / (gamma - 1.0))
        rho = rhoR * (a / aR) ** (2.0 / (gamma - 1.0))
        return rho, u, p

    rhoLf, uLf, pLf = left_fan_state()
    rhoRf, uRf, pRf = right_fan_state()

    # Compose candidate states:
    # Left candidates (used if S_M >= 0)
    # - left far state
    rhoL_far, uL_far, pL_far = rhoL, uL, pL
    # - left star
    rhoL_st, uL_st, pL_st = rhoL_star, S_M, p_star
    # - left fan-at-zero
    rhoL_in, uL_in, pL_in = rhoLf, uLf, pLf

    # Right candidates (used if S_M < 0)
    rhoR_far, uR_far, pR_far = rhoR, uR, pR
    rhoR_st, uR_st, pR_st = rhoR_star, S_M, p_star
    rhoR_in, uR_in, pR_in = rhoRf, uRf, pRf

    # Decide which side controls interface by sign of S_M
    left_controls = S_M >= 0.0

    # LEFT decision tree at xi=0:
    # shock: if 0 <= S_L -> left far, else left star
    # rare:  if 0 <= S_L_head -> left far
    #        elif 0 >= S_L_tail -> left star
    #        else -> fan state
    SL_head = uL - aL  # used only when rare
    use_left_far  = jnp.where(left_is_rare,  (0.0 <= SL_head), (0.0 <= S_L))
    use_left_star = jnp.where(left_is_rare,  (0.0 >= S_L_tail), (0.0 >  S_L))
    use_left_fan  = jnp.logical_and(left_is_rare, jnp.logical_not(jnp.logical_or(use_left_far, use_left_star)))

    rhoL_pick = jnp.where(use_left_far,  rhoL_far,
                   jnp.where(use_left_star, rhoL_st, rhoL_in))
    uL_pick   = jnp.where(use_left_far,  uL_far,
                   jnp.where(use_left_star, uL_st,   uL_in))
    pL_pick   = jnp.where(use_left_far,  pL_far,
                   jnp.where(use_left_star, pL_st,   pL_in))

    # RIGHT decision tree at xi=0:
    # shock: if 0 >= S_R -> right far, else right star
    # rare:  if 0 >= S_R_head -> right far
    #        elif 0 <= S_R_tail -> right star
    #        else -> fan state
    SR_head = uR + aR
    use_right_far  = jnp.where(right_is_rare, (0.0 >= SR_head), (0.0 >= S_R))
    use_right_star = jnp.where(right_is_rare, (0.0 <= S_R_tail), (0.0 <  S_R))
    use_right_fan  = jnp.logical_and(right_is_rare, jnp.logical_not(jnp.logical_or(use_right_far, use_right_star)))

    rhoR_pick = jnp.where(use_right_far,  rhoR_far,
                   jnp.where(use_right_star, rhoR_st, rhoR_in))
    uR_pick   = jnp.where(use_right_far,  uR_far,
                   jnp.where(use_right_star, uR_st,   uR_in))
    pR_pick   = jnp.where(use_right_far,  pR_far,
                   jnp.where(use_right_star, pR_st,   pR_in))

    # select side by S_M sign
    rho_if = jnp.where(left_controls, rhoL_pick, rhoR_pick)
    u_if   = jnp.where(left_controls, uL_pick,   uR_pick)
    p_if   = jnp.where(left_controls, pL_pick,   pR_pick)

    # build conservatives and flux
    U_if = _cons_from_prim(rho_if, u_if, p_if, gamma)
    F_exact = flux_function(U_if, gamma=gamma)

    # Optional fallback to HLLC in vacuum-danger zones (keeps robustness)
    if 'harten_lax_van_leer_contact' in globals():
        F_hllc = harten_lax_van_leer_contact(U_l, U_r, gamma=gamma)
        F_exact = jnp.where(vac[None, :], F_hllc, F_exact)

    return F_exact