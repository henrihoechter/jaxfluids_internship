"""Flux and reconstruction helpers for the compressible solver."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from . import thermodynamic_relations
from .equation_manager_types import EquationManager
from .mesh import Mesh
from . import state as state_module


def compute_speed_of_sound(
    rho: Float[Array, "n"],
    p: Float[Array, "n"],
    Y_s: Float[Array, "n n_species"],
    T: Float[Array, "n"],
    Tv: Float[Array, "n"],
    equation_manager: EquationManager,
) -> Float[Array, "n"]:
    """Compute the frozen speed of sound."""
    cp = thermodynamic_relations.compute_cp(T, equation_manager.species)
    cv_tr = thermodynamic_relations.compute_cv_tr(T, equation_manager.species)

    M_s = equation_manager.species.molar_masses
    Y_M = Y_s * M_s[None, :]
    c_s = Y_M / jnp.sum(Y_M, axis=1, keepdims=True)

    cp_mix = jnp.sum(c_s * cp.T, axis=1)
    cv_tr_mix = jnp.sum(c_s * cv_tr.T, axis=1)
    gamma_frozen = cp_mix / (cv_tr_mix + 1e-14)
    return jnp.sqrt(gamma_frozen * p / (rho + 1e-14))

def _compute_physical_flux_normal(
    U: Float[Array, "n_faces n_vars"],
    p: Float[Array, "n_faces"],
    equation_manager: EquationManager,
) -> Float[Array, "n_faces n_vars"]:
    """Compute the physical flux in the face-normal frame."""
    n_species = equation_manager.species.n_species
    n_faces, n_vars = U.shape

    rho_s = U[:, :n_species]
    rho_u_n = U[:, n_species]
    rho_u_t = U[:, n_species + 1]
    rho_E = U[:, n_vars - 2]
    rho_Ev = U[:, n_vars - 1]

    rho = jnp.sum(rho_s, axis=1)
    u_n = rho_u_n / rho

    F = jnp.zeros((n_faces, n_vars))
    F = F.at[:, :n_species].set(rho_s * u_n[:, None])
    F = F.at[:, n_species].set(rho_u_n * u_n + p)
    F = F.at[:, n_species + 1].set(rho_u_t * u_n)
    F = F.at[:, n_vars - 2].set((rho_E + p) * u_n)
    F = F.at[:, n_vars - 1].set(rho_Ev * u_n)
    return F

def _hllc_star_state_normal(
    U: Float[Array, "n n_vars"],
    S: Float[Array, "n"],
    S_star: Float[Array, "n"],
    p: Float[Array, "n"],
    rho: Float[Array, "n"],
    u_n: Float[Array, "n"],
) -> Float[Array, "n n_vars"]:
    """Compute the HLLC star state in the normal frame."""
    n_vars = U.shape[1]
    rho_star = rho * (S - u_n) / (S - S_star + 1e-14)
    factor = ((S - u_n) / (S - S_star + 1e-14))[:, None]

    n_species = n_vars - 4
    rho_u_t = U[:, n_species + 1]
    rho_E = U[:, n_vars - 2]
    rho_Ev = U[:, n_vars - 1]

    U_star = jnp.zeros_like(U)
    U_star = U_star.at[:, :n_species].set(U[:, :n_species] * factor)
    U_star = U_star.at[:, n_species].set(rho_star * S_star)
    U_star = U_star.at[:, n_species + 1].set(rho_u_t * factor[:, 0])

    p_star_term = p / (S - u_n + 1e-14)
    rho_E_star = factor[:, 0] * (
        rho_E + (S_star - u_n) * (rho_star * S_star + p_star_term)
    )
    U_star = U_star.at[:, n_vars - 2].set(rho_E_star)
    U_star = U_star.at[:, n_vars - 1].set((rho_star / rho) * rho_Ev)
    return U_star


def _hllc_flux_normal(
    U_Ln: Float[Array, "n_faces n_vars"],
    U_Rn: Float[Array, "n_faces n_vars"],
    p_L: Float[Array, "n_faces"],
    p_R: Float[Array, "n_faces"],
    equation_manager: EquationManager,
) -> Float[Array, "n_faces n_vars"]:
    """Compute the HLLC flux in the face-normal frame."""
    n_species = equation_manager.species.n_species
    rho_L = jnp.sum(U_Ln[:, :n_species], axis=1)
    rho_R = jnp.sum(U_Rn[:, :n_species], axis=1)
    u_n_L = U_Ln[:, n_species] / rho_L
    u_n_R = U_Rn[:, n_species] / rho_R

    # Primitives for speed of sound (thermodynamics invariant under rotation)
    prim_L = state_module.extract_primitives_from_U(U_Ln, equation_manager)
    prim_R = state_module.extract_primitives_from_U(U_Rn, equation_manager)

    a_L = compute_speed_of_sound(
        rho_L, p_L, prim_L.Y_s, prim_L.T, prim_L.Tv, equation_manager
    )
    a_R = compute_speed_of_sound(
        rho_R, p_R, prim_R.Y_s, prim_R.T, prim_R.Tv, equation_manager
    )

    a_max = jnp.maximum(a_L, a_R)
    S_L = jnp.minimum(u_n_L, u_n_R) - a_max
    S_R = jnp.maximum(u_n_L, u_n_R) + a_max
    S_star = (
        p_R - p_L + rho_L * u_n_L * (S_L - u_n_L) - rho_R * u_n_R * (S_R - u_n_R)
    ) / (rho_L * (S_L - u_n_L) - rho_R * (S_R - u_n_R) + 1e-14)

    F_L = _compute_physical_flux_normal(U_Ln, p_L, equation_manager)
    F_R = _compute_physical_flux_normal(U_Rn, p_R, equation_manager)

    n_faces, n_vars = U_Ln.shape
    F = jnp.zeros((n_faces, n_vars))

    mask1 = (S_L >= 0.0)[:, None]
    F = jnp.where(mask1, F_L, F)

    U_star_L = _hllc_star_state_normal(U_Ln, S_L, S_star, p_L, rho_L, u_n_L)
    U_star_R = _hllc_star_state_normal(U_Rn, S_R, S_star, p_R, rho_R, u_n_R)

    F_star_L = F_L + S_L[:, None] * (U_star_L - U_Ln)
    mask2 = ((S_L < 0.0) & (S_star >= 0.0))[:, None]
    F = jnp.where(mask2, F_star_L, F)

    F_star_R = F_R + S_R[:, None] * (U_star_R - U_Rn)
    mask3 = ((S_star < 0.0) & (S_R >= 0.0))[:, None]
    F = jnp.where(mask3, F_star_R, F)

    mask4 = (S_R < 0.0)[:, None]
    F = jnp.where(mask4, F_R, F)

    return F

def _exact_pressure_function(
    p_star: Float[Array, "n"],
    p_k: Float[Array, "n"],
    rho_k: Float[Array, "n"],
    a_k: Float[Array, "n"],
    gamma: Float[Array, "n"],
) -> tuple[Float[Array, "n"], Float[Array, "n"]]:
    """Compute the exact-Riemann pressure function and its derivative."""
    A_k = 2.0 / ((gamma + 1.0) * rho_k + 1e-14)
    B_k = (gamma - 1.0) / (gamma + 1.0) * p_k
    sq = jnp.sqrt(A_k / (p_star + B_k + 1e-14))
    f_shock = (p_star - p_k) * sq
    df_shock = sq * (1.0 - (p_star - p_k) / (2.0 * (p_star + B_k + 1e-14)))

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
    """Solve for the star pressure and velocity."""
    du = u_R - u_L
    p_0 = jnp.maximum(
        0.5 * (p_L + p_R) - 0.125 * du * (rho_L + rho_R) * (a_L + a_R),
        1e-10,
    )

    def _step(p_k, _):
        f_L, df_L = _exact_pressure_function(p_k, p_L, rho_L, a_L, gamma)
        f_R, df_R = _exact_pressure_function(p_k, p_R, rho_R, a_R, gamma)
        return jnp.maximum(p_k - (f_L + f_R + du) / (df_L + df_R + 1e-14), 1e-10), None

    p_star, _ = jax.lax.scan(_step, p_0, xs=None, length=20)
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
    """Sample the exact Riemann solution at the interface."""
    mu2 = (gamma - 1.0) / (gamma + 1.0)
    gm1 = gamma - 1.0
    g1 = gm1 / (2.0 * gamma)
    g2 = (gamma + 1.0) / (2.0 * gamma)

    # Left wave
    is_l_shock = p_star > p_L
    rho_star_L = jnp.where(
        is_l_shock,
        rho_L * (p_star / p_L + mu2) / (1.0 + mu2 * p_star / p_L),
        rho_L * (p_star / (p_L + 1e-14)) ** (1.0 / gamma),
    )
    S_L_shock = u_L - a_L * jnp.sqrt(g2 * p_star / (p_L + 1e-14) + g1)
    a_star_L = a_L * (p_star / (p_L + 1e-14)) ** g1
    S_HL = u_L - a_L
    S_TL = u_star - a_star_L
    u_fan_L = 2.0 / (gamma + 1.0) * (a_L + gm1 / 2.0 * u_L)
    a_fan_L = a_L - gm1 / 2.0 * (u_fan_L - u_L)
    rho_fan_L = rho_L * (a_fan_L / (a_L + 1e-14)) ** (2.0 / gm1)
    p_fan_L = p_L * (a_fan_L / (a_L + 1e-14)) ** (2.0 * gamma / gm1)
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

    # Right wave
    is_r_shock = p_star > p_R
    rho_star_R = jnp.where(
        is_r_shock,
        rho_R * (p_star / p_R + mu2) / (1.0 + mu2 * p_star / p_R),
        rho_R * (p_star / (p_R + 1e-14)) ** (1.0 / gamma),
    )
    S_R_shock = u_R + a_R * jnp.sqrt(g2 * p_star / (p_R + 1e-14) + g1)
    a_star_R = a_R * (p_star / (p_R + 1e-14)) ** g1
    S_HR = u_R + a_R
    S_TR = u_star + a_star_R
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

    is_left = u_star >= 0.0
    return (
        jnp.where(is_left, rho_left, rho_right),
        jnp.where(is_left, u_left, u_right),
        jnp.where(is_left, p_left, p_right),
    )


def _exact_riemann_flux_normal(
    U_Ln: Float[Array, "n_faces n_vars"],
    U_Rn: Float[Array, "n_faces n_vars"],
    p_L: Float[Array, "n_faces"],
    p_R: Float[Array, "n_faces"],
    equation_manager: EquationManager,
) -> Float[Array, "n_faces n_vars"]:
    """Godunov flux (exact Riemann) in the face-normal frame.

    Species mass fractions, specific tangential velocity, and specific
    vibrational energy are advected as passive scalars following the
    contact wave.
    """
    n_species = equation_manager.species.n_species
    n_faces, n_vars = U_Ln.shape

    rho_L = jnp.sum(U_Ln[:, :n_species], axis=1)
    rho_R = jnp.sum(U_Rn[:, :n_species], axis=1)
    u_n_L = U_Ln[:, n_species] / rho_L
    u_n_R = U_Rn[:, n_species] / rho_R

    prim_L = state_module.extract_primitives_from_U(U_Ln, equation_manager)
    prim_R = state_module.extract_primitives_from_U(U_Rn, equation_manager)

    a_L = compute_speed_of_sound(
        rho_L, p_L, prim_L.Y_s, prim_L.T, prim_L.Tv, equation_manager
    )
    a_R = compute_speed_of_sound(
        rho_R, p_R, prim_R.Y_s, prim_R.T, prim_R.Tv, equation_manager
    )

    gamma_L = a_L**2 * rho_L / (p_L + 1e-14)
    gamma_R = a_R**2 * rho_R / (p_R + 1e-14)
    gamma = 0.5 * (gamma_L + gamma_R)

    p_star, u_star = _exact_solve_star_state(
        rho_L, u_n_L, p_L, a_L, rho_R, u_n_R, p_R, a_R, gamma
    )
    rho_s, u_n_s, p_s = _exact_sample_at_interface(
        rho_L, u_n_L, p_L, a_L, rho_R, u_n_R, p_R, a_R, p_star, u_star, gamma
    )

    is_left_contact = u_star >= 0.0

    # Upwind mass fractions
    c_s_L = U_Ln[:, :n_species] / (rho_L[:, None] + 1e-14)
    c_s_R = U_Rn[:, :n_species] / (rho_R[:, None] + 1e-14)
    c_s = jnp.where(is_left_contact[:, None], c_s_L, c_s_R)

    # Upwind specific tangential velocity
    u_t_L = U_Ln[:, n_species + 1] / (rho_L + 1e-14)
    u_t_R = U_Rn[:, n_species + 1] / (rho_R + 1e-14)
    u_t_s = jnp.where(is_left_contact, u_t_L, u_t_R)

    # Upwind specific vibrational energy
    ev_L = U_Ln[:, n_vars - 1] / (rho_L + 1e-14)
    ev_R = U_Rn[:, n_vars - 1] / (rho_R + 1e-14)
    ev_s = jnp.where(is_left_contact, ev_L, ev_R)

    # Reconstruct total energy at interface (Gnoffo reference offset)
    T_ref = 298.16
    T_dummy = jnp.ones(1)
    cv_tr_s = thermodynamic_relations.compute_cv_tr(T_dummy, equation_manager.species)[
        :, 0
    ]
    e_s0 = thermodynamic_relations.compute_reference_internal_energy(
        equation_manager.species.h_s0,
        equation_manager.species.molar_masses,
        T_ref=T_ref,
    )
    delta_e_s = e_s0 - cv_tr_s * T_ref
    delta_e = jnp.sum(c_s * delta_e_s[None, :], axis=1)

    rho_Ev_s = rho_s * ev_s
    rho_E_s = (
        p_s / (gamma - 1.0)
        + rho_s * delta_e
        + rho_Ev_s
        + 0.5 * rho_s * (u_n_s**2 + u_t_s**2)
    )

    F = jnp.zeros((n_faces, n_vars))
    F = F.at[:, :n_species].set(rho_s[:, None] * c_s * u_n_s[:, None])
    F = F.at[:, n_species].set(rho_s * u_n_s**2 + p_s)
    F = F.at[:, n_species + 1].set(rho_s * u_t_s * u_n_s)
    F = F.at[:, n_vars - 2].set((rho_E_s + p_s) * u_n_s)
    F = F.at[:, n_vars - 1].set(rho_Ev_s * u_n_s)
    return F

def _lax_friedrichs_flux_normal(
    U_Ln: Float[Array, "n_faces n_vars"],
    U_Rn: Float[Array, "n_faces n_vars"],
    p_L: Float[Array, "n_faces"],
    p_R: Float[Array, "n_faces"],
    equation_manager: EquationManager,
) -> Float[Array, "n_faces n_vars"]:
    """Compute the local Lax-Friedrichs flux in the face-normal frame."""
    n_species = equation_manager.species.n_species
    rho_L = jnp.sum(U_Ln[:, :n_species], axis=1)
    rho_R = jnp.sum(U_Rn[:, :n_species], axis=1)
    u_n_L = U_Ln[:, n_species] / rho_L
    u_n_R = U_Rn[:, n_species] / rho_R

    prim_L = state_module.extract_primitives_from_U(U_Ln, equation_manager)
    prim_R = state_module.extract_primitives_from_U(U_Rn, equation_manager)

    a_L = compute_speed_of_sound(
        rho_L, p_L, prim_L.Y_s, prim_L.T, prim_L.Tv, equation_manager
    )
    a_R = compute_speed_of_sound(
        rho_R, p_R, prim_R.Y_s, prim_R.T, prim_R.Tv, equation_manager
    )

    alpha = jnp.maximum(jnp.abs(u_n_L) + a_L, jnp.abs(u_n_R) + a_R)

    F_L = _compute_physical_flux_normal(U_Ln, p_L, equation_manager)
    F_R = _compute_physical_flux_normal(U_Rn, p_R, equation_manager)
    return 0.5 * (F_L + F_R) - 0.5 * alpha[:, None] * (U_Rn - U_Ln)


@jax.named_call
def compute_flux_faces(
    U_L: Float[Array, "n_faces n_variables"],
    U_R: Float[Array, "n_faces n_variables"],
    n_hat: Float[Array, "n_faces 2"],
    equation_manager: EquationManager,
    primitives_L: state_module.Primitives | None = None,
    primitives_R: state_module.Primitives | None = None,
) -> Float[Array, "n_faces n_variables"]:
    """Compute the numerical flux across each face."""
    if primitives_L is None:
        primitives_L = state_module.extract_primitives_from_U(U_L, equation_manager)
    if primitives_R is None:
        primitives_R = state_module.extract_primitives_from_U(U_R, equation_manager)

    n_species = equation_manager.species.n_species
    n_faces, n_vars = U_L.shape

    n_x = n_hat[:, 0]
    n_y = n_hat[:, 1]
    t_x = -n_y
    t_y = n_x

    # Rotate momentum to (normal, tangential) frame
    u_n_L = primitives_L.u * n_x + primitives_L.v * n_y
    u_t_L = primitives_L.u * t_x + primitives_L.v * t_y
    u_n_R = primitives_R.u * n_x + primitives_R.v * n_y
    u_t_R = primitives_R.u * t_x + primitives_R.v * t_y

    U_Ln = jnp.zeros((n_faces, n_vars))
    U_Rn = jnp.zeros((n_faces, n_vars))

    U_Ln = U_Ln.at[:, :n_species].set(U_L[:, :n_species])
    U_Ln = U_Ln.at[:, n_species].set(primitives_L.rho * u_n_L)
    U_Ln = U_Ln.at[:, n_species + 1].set(primitives_L.rho * u_t_L)
    U_Ln = U_Ln.at[:, n_vars - 2].set(U_L[:, n_vars - 2])
    U_Ln = U_Ln.at[:, n_vars - 1].set(U_L[:, n_vars - 1])

    U_Rn = U_Rn.at[:, :n_species].set(U_R[:, :n_species])
    U_Rn = U_Rn.at[:, n_species].set(primitives_R.rho * u_n_R)
    U_Rn = U_Rn.at[:, n_species + 1].set(primitives_R.rho * u_t_R)
    U_Rn = U_Rn.at[:, n_vars - 2].set(U_R[:, n_vars - 2])
    U_Rn = U_Rn.at[:, n_vars - 1].set(U_R[:, n_vars - 1])

    p_L = primitives_L.p
    p_R = primitives_R.p

    scheme = equation_manager.numerics_config.flux_scheme
    if scheme == "hllc":
        F_n = _hllc_flux_normal(U_Ln, U_Rn, p_L, p_R, equation_manager)
    elif scheme == "exact_riemann":
        F_n = _exact_riemann_flux_normal(U_Ln, U_Rn, p_L, p_R, equation_manager)
    elif scheme == "lax_friedrichs":
        F_n = _lax_friedrichs_flux_normal(U_Ln, U_Rn, p_L, p_R, equation_manager)
    else:
        raise ValueError(f"Unknown flux scheme: {scheme!r}")

    # Rotate momentum flux back to Cartesian
    F_mom_n = F_n[:, n_species]
    F_mom_t = F_n[:, n_species + 1]

    F = jnp.zeros((n_faces, n_vars))
    F = F.at[:, :n_species].set(F_n[:, :n_species])
    F = F.at[:, n_species].set(F_mom_n * n_x + F_mom_t * t_x)
    F = F.at[:, n_species + 1].set(F_mom_n * n_y + F_mom_t * t_y)
    F = F.at[:, n_vars - 2].set(F_n[:, n_vars - 2])
    F = F.at[:, n_vars - 1].set(F_n[:, n_vars - 1])
    return F

def compute_face_states(
    U: Float[Array, "n_cells n_variables"],
    mesh: Mesh,
    equation_manager: EquationManager,
) -> tuple[Float[Array, "n_faces n_variables"], Float[Array, "n_faces n_variables"]]:
    """Build first-order face states."""
    from compressible import boundary_conditions

    return boundary_conditions.compute_face_states(U, mesh, equation_manager)


def compute_face_states_muscl(
    U: Float[Array, "n_cells n_variables"],
    mesh: Mesh,
    equation_manager: EquationManager,
) -> tuple[Float[Array, "n_faces n_variables"], Float[Array, "n_faces n_variables"]]:
    """Build MUSCL-reconstructed face states."""
    from compressible import boundary_conditions

    # First-order face states (includes ghost states at boundaries)
    U_L0, U_R0 = boundary_conditions.compute_face_states(U, mesh, equation_manager)

    face_left = mesh.face_left
    face_right = mesh.face_right

    # Safe indices: replace -1 with a valid index to avoid OOB access
    safe_r = jnp.where(face_right >= 0, face_right, face_left)
    safe_ll = jnp.where(mesh.muscl_ll >= 0, mesh.muscl_ll, face_left)
    safe_rr = jnp.where(mesh.muscl_rr >= 0, mesh.muscl_rr, safe_r)

    # Cell-value differences for slope estimation
    dU_minus = U[face_left] - U[safe_ll]  # U[i] - U[i-1]
    dU_center = U[safe_r] - U[face_left]  # U[j] - U[i]
    dU_plus = U[safe_rr] - U[safe_r]  # U[j+1] - U[j]

    valid_ll = (mesh.muscl_ll >= 0)[:, None]
    valid_rr = (mesh.muscl_rr >= 0)[:, None]

    slope_L = jnp.where(valid_ll, _slope(dU_minus, dU_center, equation_manager), 0.0)
    slope_R = jnp.where(valid_rr, _slope(dU_center, dU_plus, equation_manager), 0.0)

    U_L = U[face_left] + 0.5 * slope_L
    U_R_interior = U[safe_r] - 0.5 * slope_R

    # For boundary faces (face_right < 0), keep the ghost state from BCs
    interior_mask = (face_right >= 0)[:, None]
    U_R = jnp.where(interior_mask, U_R_interior, U_R0)

    # Clamp species densities: MUSCL can overshoot to negative values at strong
    # discontinuities, which corrupts chemistry source terms immediately.
    n_species = equation_manager.species.n_species
    rho_s_min = equation_manager.numerics_config.clipping.rho_s_min
    U_L = U_L.at[:, :n_species].set(jnp.clip(U_L[:, :n_species], rho_s_min, None))
    U_R = U_R.at[:, :n_species].set(jnp.clip(U_R[:, :n_species], rho_s_min, None))

    return U_L, U_R


def _slope(
    dU_a: Float[Array, "n_faces n_vars"],
    dU_b: Float[Array, "n_faces n_vars"],
    equation_manager: EquationManager,
) -> Float[Array, "n_faces n_vars"]:
    """Apply the configured slope limiter to (dU_a, dU_b)."""
    limiter = equation_manager.numerics_config.slope_limiter
    if limiter == "minmod":
        return _minmod2(dU_a, dU_b)
    elif limiter == "mc":
        return _minmod3(
            0.5 * (dU_a + dU_b),
            2.0 * dU_a,
            2.0 * dU_b,
        )
    else:
        raise ValueError(f"Unknown slope limiter: {limiter!r}")


def _minmod2(
    a: Float[Array, "n_faces n_vars"], b: Float[Array, "n_faces n_vars"]
) -> Float[Array, "n_faces n_vars"]:
    """Apply the two-argument minmod limiter."""
    return jnp.where(
        a * b > 0.0, jnp.sign(a) * jnp.minimum(jnp.abs(a), jnp.abs(b)), 0.0
    )


def _minmod3(
    a: Float[Array, "n_faces n_vars"],
    b: Float[Array, "n_faces n_vars"],
    c: Float[Array, "n_faces n_vars"],
) -> Float[Array, "n_faces n_vars"]:
    """Apply the three-argument minmod limiter."""
    return _minmod2(a, _minmod2(b, c))
