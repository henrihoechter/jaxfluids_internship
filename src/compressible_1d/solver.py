"""Solver module for computing numerical fluxes.

Implements:
- HLLC Riemann solver for multi-species two-temperature Euler equations.
- Exact Riemann solver (Godunov flux) using a polytropic ideal gas assumption
  with the local frozen specific heat ratio. Valid when the frozen gamma
  approximation holds (low temperature or effectively single-species flow).
  Species mass fractions and vibrational energy are advected as passive
  scalars following the contact wave.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Float, Array

from compressible_1d import (
    equation_manager_types,
    equation_manager_utils,
)
from compressible_core import thermodynamic_relations


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def compute_speed_of_sound(
    rho: Float[Array, "n_cells"],
    p: Float[Array, "n_cells"],
    Y_s: Float[Array, "n_cells n_species"],
    T: Float[Array, "n_cells"],
    Tv: Float[Array, "n_cells"],
    equation_manager: equation_manager_types.EquationManager,
) -> Float[Array, "n_cells"]:
    """Compute speed of sound for two-temperature multi-species gas.

    a = sqrt(gamma_frozen * p / rho)

    where gamma_frozen = cp_tr / cv_tr is the mixture-averaged frozen specific
    heat ratio (vibrational modes frozen on short timescales).

    Args:
        rho: Density [n_cells]
        p: Pressure [n_cells]
        Y_s: Mole fractions [n_cells, n_species]
        T: Translational temperature [n_cells]
        Tv: Vibrational temperature [n_cells]
        equation_manager: Contains species data

    Returns:
        a: Speed of sound [n_cells]
    """
    cp = thermodynamic_relations.compute_cp(T, equation_manager.species)
    cv_tr = thermodynamic_relations.compute_cv_tr(T, equation_manager.species)

    M_s = equation_manager.species.molar_masses
    Y_M = Y_s * M_s[None, :]
    c_s = Y_M / jnp.sum(Y_M, axis=1, keepdims=True)

    cp_mix = jnp.sum(c_s * cp.T, axis=1)
    cv_tr_mix = jnp.sum(c_s * cv_tr.T, axis=1)

    gamma_frozen = cp_mix / (cv_tr_mix + 1e-14)
    return jnp.sqrt(gamma_frozen * p / (rho + 1e-14))


def compute_physical_flux(
    U: Float[Array, "n_cells n_variables"],
    p: Float[Array, "n_cells"],
    equation_manager: equation_manager_types.EquationManager,
) -> Float[Array, "n_cells n_variables"]:
    """Compute physical flux for conserved variables.

    F = [rho_i*u, rho*u^2 + p, (rho*E + p)*u, rho*E_v*u]

    Args:
        U: Conserved state [n_cells, n_variables]
        p: Pressure [n_cells]
        equation_manager: Contains species data

    Returns:
        F: Physical flux [n_cells, n_variables]
    """
    n_species = equation_manager.species.n_species
    n_cells, n_variables = U.shape

    rho_species = U[:, :n_species]
    rho_u = U[:, n_species]
    rho_E = U[:, n_variables - 2]
    rho_Ev = U[:, n_variables - 1]

    rho = jnp.sum(rho_species, axis=1)
    u = rho_u / rho

    F = jnp.zeros((n_cells, n_variables))
    F = F.at[:, :n_species].set(rho_species * u[:, None])
    F = F.at[:, n_species].set(rho_u * u + p)
    F = F.at[:, n_variables - 2].set((rho_E + p) * u)
    F = F.at[:, n_variables - 1].set(rho_Ev * u)
    return F


# ---------------------------------------------------------------------------
# HLLC Riemann solver
# ---------------------------------------------------------------------------


def compute_hllc_flux(
    U_L: Float[Array, "n_interfaces n_variables"],
    U_R: Float[Array, "n_interfaces n_variables"],
    equation_manager: equation_manager_types.EquationManager,
    primitives_L: equation_manager_utils.Primitives1D | None = None,
    primitives_R: equation_manager_utils.Primitives1D | None = None,
) -> Float[Array, "n_interfaces n_variables"]:
    """Compute numerical flux at cell interfaces using the HLLC Riemann solver.

    Args:
        U_L: Left states at interfaces [n_interfaces, n_variables]
        U_R: Right states at interfaces [n_interfaces, n_variables]
        equation_manager: Contains species table and config
        primitives_L: Precomputed primitives for U_L (optional)
        primitives_R: Precomputed primitives for U_R (optional)

    Returns:
        F: Numerical flux [n_interfaces, n_variables]

    Notes:
        State vector: [rho_1, ..., rho_ns, rho*u, rho*E, rho*E_v]
    """
    if primitives_L is None:
        primitives_L = equation_manager_utils.extract_primitives(U_L, equation_manager)
    if primitives_R is None:
        primitives_R = equation_manager_utils.extract_primitives(U_R, equation_manager)

    Y_L, rho_L, T_L, Tv_L, p_L = primitives_L
    Y_R, rho_R, T_R, Tv_R, p_R = primitives_R

    n_species = equation_manager.species.n_species
    u_L = U_L[:, n_species] / rho_L
    u_R = U_R[:, n_species] / rho_R

    a_L = compute_speed_of_sound(rho_L, p_L, Y_L, T_L, Tv_L, equation_manager)
    a_R = compute_speed_of_sound(rho_R, p_R, Y_R, T_R, Tv_R, equation_manager)

    # Wave speed estimates (HLL)
    a_max = jnp.maximum(a_L, a_R)
    S_L = jnp.minimum(u_L, u_R) - a_max
    S_R = jnp.maximum(u_L, u_R) + a_max

    # Contact wave speed (HLLC)
    S_star = (p_R - p_L + rho_L * u_L * (S_L - u_L) - rho_R * u_R * (S_R - u_R)) / (
        rho_L * (S_L - u_L) - rho_R * (S_R - u_R) + 1e-14
    )

    F_L = compute_physical_flux(U_L, p_L, equation_manager)
    F_R = compute_physical_flux(U_R, p_R, equation_manager)

    n_interfaces, n_variables = U_L.shape
    F = jnp.zeros((n_interfaces, n_variables))

    mask1 = (S_L >= 0.0)[:, None]
    F = jnp.where(mask1, F_L, F)

    U_star_L = _hllc_star_state(U_L, S_L, S_star, p_L, rho_L, u_L, equation_manager)
    U_star_R = _hllc_star_state(U_R, S_R, S_star, p_R, rho_R, u_R, equation_manager)

    F_star_L = F_L + S_L[:, None] * (U_star_L - U_L)
    mask2 = ((S_L < 0.0) & (S_star >= 0.0))[:, None]
    F = jnp.where(mask2, F_star_L, F)

    F_star_R = F_R + S_R[:, None] * (U_star_R - U_R)
    mask3 = ((S_star < 0.0) & (S_R >= 0.0))[:, None]
    F = jnp.where(mask3, F_star_R, F)

    mask4 = (S_R < 0.0)[:, None]
    F = jnp.where(mask4, F_R, F)

    return F


def _hllc_star_state(
    U: Float[Array, "n_cells n_variables"],
    S: Float[Array, "n_cells"],
    S_star: Float[Array, "n_cells"],
    p: Float[Array, "n_cells"],
    rho: Float[Array, "n_cells"],
    u: Float[Array, "n_cells"],
    equation_manager: equation_manager_types.EquationManager,
) -> Float[Array, "n_cells n_variables"]:
    """Compute HLLC star state.

    Args:
        U: Conserved state [n_cells, n_variables]
        S: Outer wave speed (S_L or S_R) [n_cells]
        S_star: Contact wave speed [n_cells]
        p: Pressure [n_cells]
        rho: Density [n_cells]
        u: Velocity [n_cells]
        equation_manager: Contains species data

    Returns:
        U_star: Star state [n_cells, n_variables]
    """
    n_species = equation_manager.species.n_species

    rho_star = rho * (S - u) / (S - S_star + 1e-14)

    n_variables = U.shape[1]
    rho_species = U[:, :n_species]
    rho_E = U[:, n_variables - 2]
    rho_Ev = U[:, n_variables - 1]

    U_star = jnp.zeros_like(U)

    factor = ((S - u) / (S - S_star + 1e-14))[:, None]
    U_star = U_star.at[:, :n_species].set(rho_species * factor)
    U_star = U_star.at[:, n_species].set(rho_star * S_star)

    p_star_term = p / (S - u + 1e-14)
    rho_E_star = factor[:, 0] * (
        rho_E + (S_star - u) * (rho_star * S_star + p_star_term)
    )
    U_star = U_star.at[:, n_variables - 2].set(rho_E_star)

    rho_Ev_star = (rho_star / rho) * rho_Ev
    U_star = U_star.at[:, n_variables - 1].set(rho_Ev_star)

    return U_star


# ---------------------------------------------------------------------------
# Exact Riemann solver (Godunov flux, polytropic ideal gas)
# ---------------------------------------------------------------------------


def compute_exact_riemann_flux(
    U_L: Float[Array, "n_interfaces n_variables"],
    U_R: Float[Array, "n_interfaces n_variables"],
    equation_manager: equation_manager_types.EquationManager,
    primitives_L: equation_manager_utils.Primitives1D | None = None,
    primitives_R: equation_manager_utils.Primitives1D | None = None,
) -> Float[Array, "n_interfaces n_variables"]:
    """Compute Godunov flux at interfaces using the exact Riemann solver.

    Solves the exact Riemann problem at each interface assuming a polytropic
    ideal gas with the local frozen specific heat ratio gamma = a^2 * rho / p.
    Species mass fractions and specific vibrational energy are advected as
    passive scalars that follow the contact wave.

    The approach is only accurate when gamma is approximately uniform across
    the domain (e.g. low-temperature single-species flow). For the Sod problem
    with pure N2 at ~300 K, gamma_frozen ≈ 1.4 everywhere and the result is
    essentially exact.

    Args:
        U_L: Left states at interfaces [n_interfaces, n_variables]
        U_R: Right states at interfaces [n_interfaces, n_variables]
        equation_manager: Contains species table and config
        primitives_L: Precomputed primitives for U_L (optional)
        primitives_R: Precomputed primitives for U_R (optional)

    Returns:
        F: Godunov flux [n_interfaces, n_variables]
    """
    if primitives_L is None:
        primitives_L = equation_manager_utils.extract_primitives(U_L, equation_manager)
    if primitives_R is None:
        primitives_R = equation_manager_utils.extract_primitives(U_R, equation_manager)

    Y_L, rho_L, T_L, Tv_L, p_L = primitives_L
    Y_R, rho_R, T_R, Tv_R, p_R = primitives_R

    n_species = equation_manager.species.n_species
    u_L = U_L[:, n_species] / rho_L
    u_R = U_R[:, n_species] / rho_R

    a_L = compute_speed_of_sound(rho_L, p_L, Y_L, T_L, Tv_L, equation_manager)
    a_R = compute_speed_of_sound(rho_R, p_R, Y_R, T_R, Tv_R, equation_manager)

    # Frozen gamma from a^2 = gamma * p / rho; average across the interface.
    gamma_L = a_L**2 * rho_L / (p_L + 1e-14)
    gamma_R = a_R**2 * rho_R / (p_R + 1e-14)
    gamma = 0.5 * (gamma_L + gamma_R)

    # Solve star state and sample at xi = x/t = 0 for every interface
    p_star, u_star = _exact_solve_star_state(
        rho_L, u_L, p_L, a_L, rho_R, u_R, p_R, a_R, gamma
    )
    rho_s, u_s, p_s = _exact_sample_at_interface(
        rho_L, u_L, p_L, a_L, rho_R, u_R, p_R, a_R, p_star, u_star, gamma
    )

    # Advected scalars: upwind direction follows the contact wave (u_star)
    is_left_contact = u_star >= 0.0

    # Mass fractions
    c_s_L = U_L[:, :n_species] / (rho_L[:, None] + 1e-14)
    c_s_R = U_R[:, :n_species] / (rho_R[:, None] + 1e-14)
    c_s = jnp.where(is_left_contact[:, None], c_s_L, c_s_R)

    # Specific vibrational energy (per unit mass): ev = rho*Ev / rho
    n_variables = U_L.shape[1]
    ev_L = U_L[:, n_variables - 1] / (rho_L + 1e-14)
    ev_R = U_R[:, n_variables - 1] / (rho_R + 1e-14)
    ev_s = jnp.where(is_left_contact, ev_L, ev_R)

    # Reconstruct volumetric vibrational energy and total energy at interface.
    # The solver stores energy with a chemical reference offset (Gnoffo 1989):
    #   e_tr = e_0 + cv_tr*(T - T_ref)  where T_ref = 298.16 K
    # From the equation of state p = rho*R*T => T = p/(rho*R), so:
    #   rho*e_tr = p/(gamma-1) + rho*delta_e
    # where delta_e = sum_s c_s*(e_s0 - cv_tr_s*T_ref) is the reference offset.
    T_ref = 298.16
    T_dummy = jnp.ones(1)
    cv_tr_s = thermodynamic_relations.compute_cv_tr(T_dummy, equation_manager.species)[
        :, 0
    ]  # [n_species]
    e_s0 = thermodynamic_relations.compute_reference_internal_energy(
        equation_manager.species.h_s0,
        equation_manager.species.molar_masses,
        T_ref=T_ref,
    )  # [n_species]
    delta_e_s = e_s0 - cv_tr_s * T_ref  # [n_species]
    # c_s are the upwind mass fractions at the interface
    delta_e = jnp.sum(c_s * delta_e_s[None, :], axis=1)  # [n_interfaces]

    rho_Ev_s = rho_s * ev_s
    rho_E_s = p_s / (gamma - 1.0) + rho_s * delta_e + rho_Ev_s + 0.5 * rho_s * u_s**2

    # Assemble flux
    n_interfaces, n_variables = U_L.shape
    F = jnp.zeros((n_interfaces, n_variables))
    F = F.at[:, :n_species].set(rho_s[:, None] * c_s * u_s[:, None])
    F = F.at[:, n_species].set(rho_s * u_s**2 + p_s)
    F = F.at[:, n_variables - 2].set((rho_E_s + p_s) * u_s)
    F = F.at[:, n_variables - 1].set(rho_Ev_s * u_s)

    return F


def _exact_pressure_function(
    p_star: Float[Array, "n_interfaces"],
    p_k: Float[Array, "n_interfaces"],
    rho_k: Float[Array, "n_interfaces"],
    a_k: Float[Array, "n_interfaces"],
    gamma: Float[Array, "n_interfaces"],
) -> tuple[Float[Array, "n_interfaces"], Float[Array, "n_interfaces"]]:
    """Evaluate f_k(p*) and df_k/dp* for each interface.

    f_k is the pressure function appearing in the total pressure equation for
    the exact Riemann solver (Toro, Chapter 4). It takes the shock branch when
    p* > p_k and the isentropic rarefaction branch otherwise.

    Args:
        p_star: Current star pressure iterate [n_interfaces]
        p_k: Side pressure (p_L or p_R) [n_interfaces]
        rho_k: Side density [n_interfaces]
        a_k: Side speed of sound [n_interfaces]
        gamma: Frozen specific heat ratio [n_interfaces]

    Returns:
        f_k: Pressure function value [n_interfaces]
        df_k: Derivative with respect to p_star [n_interfaces]
    """
    # Shock branch constants
    A_k = 2.0 / ((gamma + 1.0) * rho_k + 1e-14)
    B_k = (gamma - 1.0) / (gamma + 1.0) * p_k
    sq = jnp.sqrt(A_k / (p_star + B_k + 1e-14))
    f_shock = (p_star - p_k) * sq
    df_shock = sq * (1.0 - (p_star - p_k) / (2.0 * (p_star + B_k + 1e-14)))

    # Rarefaction branch
    g1 = (gamma - 1.0) / (2.0 * gamma)
    ratio = p_star / (p_k + 1e-14)
    f_rare = 2.0 * a_k / (gamma - 1.0 + 1e-14) * (ratio**g1 - 1.0)
    df_rare = 1.0 / (rho_k * a_k + 1e-14) * ratio ** (-(gamma + 1.0) / (2.0 * gamma))

    is_shock = p_star > p_k
    return jnp.where(is_shock, f_shock, f_rare), jnp.where(is_shock, df_shock, df_rare)


def _exact_solve_star_state(
    rho_L: Float[Array, "n"],
    u_L: Float[Array, "n"],
    p_L: Float[Array, "n"],
    a_L: Float[Array, "n"],
    rho_R: Float[Array, "n"],
    u_R: Float[Array, "n"],
    p_R: Float[Array, "n"],
    a_R: Float[Array, "n"],
    gamma: Float[Array, "n"],
) -> tuple[Float[Array, "n"], Float[Array, "n"]]:
    """Solve for the star-region pressure p* and velocity u* at all interfaces.

    Uses Newton-Raphson iteration on the total pressure equation:
        f_L(p*) + f_R(p*) + (u_R - u_L) = 0

    Initialized with the primitive-variable Riemann solver (PVRS) estimate.
    All n interfaces are advanced simultaneously (no per-interface Python loop).

    Args:
        rho_L, u_L, p_L, a_L: Left state at each interface
        rho_R, u_R, p_R, a_R: Right state at each interface
        gamma: Frozen specific heat ratio at each interface

    Returns:
        p_star: Star-region pressure [n]
        u_star: Star-region velocity [n]
    """
    du = u_R - u_L

    # PVRS initial guess (Toro eq. 9.20)
    p_0 = jnp.maximum(
        0.5 * (p_L + p_R) - 0.125 * du * (rho_L + rho_R) * (a_L + a_R),
        1e-10,
    )

    def _newton_step(p_k, _):
        f_L, df_L = _exact_pressure_function(p_k, p_L, rho_L, a_L, gamma)
        f_R, df_R = _exact_pressure_function(p_k, p_R, rho_R, a_R, gamma)
        p_new = jnp.maximum(p_k - (f_L + f_R + du) / (df_L + df_R + 1e-14), 1e-10)
        return p_new, None

    p_star, _ = jax.lax.scan(_newton_step, p_0, xs=None, length=20)

    f_L, _ = _exact_pressure_function(p_star, p_L, rho_L, a_L, gamma)
    f_R, _ = _exact_pressure_function(p_star, p_R, rho_R, a_R, gamma)
    u_star = 0.5 * (u_L + u_R) + 0.5 * (f_R - f_L)

    return p_star, u_star


def _exact_sample_at_interface(
    rho_L: Float[Array, "n"],
    u_L: Float[Array, "n"],
    p_L: Float[Array, "n"],
    a_L: Float[Array, "n"],
    rho_R: Float[Array, "n"],
    u_R: Float[Array, "n"],
    p_R: Float[Array, "n"],
    a_R: Float[Array, "n"],
    p_star: Float[Array, "n"],
    u_star: Float[Array, "n"],
    gamma: Float[Array, "n"],
) -> tuple[Float[Array, "n"], Float[Array, "n"], Float[Array, "n"]]:
    """Sample the exact Riemann solution at the interface (xi = x/t = 0).

    Following Toro Chapter 4. All branches (shock / rarefaction, fan interior)
    are evaluated and the correct one selected with jnp.where.

    Args:
        rho_L, u_L, p_L, a_L: Left state
        rho_R, u_R, p_R, a_R: Right state
        p_star: Star-region pressure
        u_star: Star-region velocity
        gamma: Frozen specific heat ratio

    Returns:
        rho_s, u_s, p_s: Sampled primitive state at xi = 0
    """
    mu2 = (gamma - 1.0) / (gamma + 1.0)  # = (gamma-1)/(gamma+1)
    gm1 = gamma - 1.0
    g1 = gm1 / (2.0 * gamma)  # = (gamma-1) / (2*gamma)
    g2 = (gamma + 1.0) / (2.0 * gamma)  # = (gamma+1) / (2*gamma)

    # --- Left wave ---
    is_l_shock = p_star > p_L

    # Star density (shock: Rankine-Hugoniot; rarefaction: isentropic)
    rho_star_L = jnp.where(
        is_l_shock,
        rho_L * (p_star / p_L + mu2) / (1.0 + mu2 * p_star / p_L),
        rho_L * (p_star / (p_L + 1e-14)) ** (1.0 / gamma),
    )

    # Shock speed and rarefaction fan boundaries
    S_L_shock = u_L - a_L * jnp.sqrt(g2 * p_star / (p_L + 1e-14) + g1)
    a_star_L = a_L * (p_star / (p_L + 1e-14)) ** g1
    S_HL = u_L - a_L  # rarefaction head speed
    S_TL = u_star - a_star_L  # rarefaction tail speed

    # Left rarefaction fan at xi = 0 (Toro eq. 4.56 with xi = 0)
    u_fan_L = 2.0 / (gamma + 1.0) * (a_L + gm1 / 2.0 * u_L)
    a_fan_L = a_L - gm1 / 2.0 * (u_fan_L - u_L)
    rho_fan_L = rho_L * (a_fan_L / (a_L + 1e-14)) ** (2.0 / gm1)
    p_fan_L = p_L * (a_fan_L / (a_L + 1e-14)) ** (2.0 * gamma / gm1)

    # Select left-side sample
    rho_left = jnp.where(
        is_l_shock,
        jnp.where(S_L_shock >= 0.0, rho_L, rho_star_L),
        jnp.where(S_HL >= 0.0, rho_L, jnp.where(S_TL >= 0.0, rho_fan_L, rho_star_L)),
    )
    u_left = jnp.where(
        is_l_shock,
        jnp.where(S_L_shock >= 0.0, u_L, u_star),
        jnp.where(S_HL >= 0.0, u_L, jnp.where(S_TL >= 0.0, u_fan_L, u_star)),
    )
    p_left = jnp.where(
        is_l_shock,
        jnp.where(S_L_shock >= 0.0, p_L, p_star),
        jnp.where(S_HL >= 0.0, p_L, jnp.where(S_TL >= 0.0, p_fan_L, p_star)),
    )

    # --- Right wave ---
    is_r_shock = p_star > p_R

    rho_star_R = jnp.where(
        is_r_shock,
        rho_R * (p_star / p_R + mu2) / (1.0 + mu2 * p_star / p_R),
        rho_R * (p_star / (p_R + 1e-14)) ** (1.0 / gamma),
    )

    S_R_shock = u_R + a_R * jnp.sqrt(g2 * p_star / (p_R + 1e-14) + g1)
    a_star_R = a_R * (p_star / (p_R + 1e-14)) ** g1
    S_HR = u_R + a_R  # rarefaction head speed
    S_TR = u_star + a_star_R  # rarefaction tail speed

    # Right rarefaction fan at xi = 0 (Toro eq. 4.63 with xi = 0)
    u_fan_R = 2.0 / (gamma + 1.0) * (-a_R + gm1 / 2.0 * u_R)
    a_fan_R = a_R + gm1 / 2.0 * (u_R - u_fan_R)
    rho_fan_R = rho_R * (a_fan_R / (a_R + 1e-14)) ** (2.0 / gm1)
    p_fan_R = p_R * (a_fan_R / (a_R + 1e-14)) ** (2.0 * gamma / gm1)

    rho_right = jnp.where(
        is_r_shock,
        jnp.where(S_R_shock <= 0.0, rho_R, rho_star_R),
        jnp.where(S_HR <= 0.0, rho_R, jnp.where(S_TR <= 0.0, rho_fan_R, rho_star_R)),
    )
    u_right = jnp.where(
        is_r_shock,
        jnp.where(S_R_shock <= 0.0, u_R, u_star),
        jnp.where(S_HR <= 0.0, u_R, jnp.where(S_TR <= 0.0, u_fan_R, u_star)),
    )
    p_right = jnp.where(
        is_r_shock,
        jnp.where(S_R_shock <= 0.0, p_R, p_star),
        jnp.where(S_HR <= 0.0, p_R, jnp.where(S_TR <= 0.0, p_fan_R, p_star)),
    )

    # --- Select side based on contact position relative to xi = 0 ---
    is_left_of_contact = u_star >= 0.0
    rho_s = jnp.where(is_left_of_contact, rho_left, rho_right)
    u_s = jnp.where(is_left_of_contact, u_left, u_right)
    p_s = jnp.where(is_left_of_contact, p_left, p_right)

    return rho_s, u_s, p_s
