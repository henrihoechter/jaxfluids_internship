"""2D HLLC solver for multi-species two-temperature Euler equations."""

import jax.numpy as jnp
from jaxtyping import Array, Float

from .equation_manager_types import EquationManager2D
from . import equation_manager_utils
from compressible_core import thermodynamic_relations


def compute_speed_of_sound(
    rho: Float[Array, "n_cells"],
    p: Float[Array, "n_cells"],
    Y_s: Float[Array, "n_cells n_species"],
    T: Float[Array, "n_cells"],
    Tv: Float[Array, "n_cells"],
    equation_manager: EquationManager2D,
) -> Float[Array, "n_cells"]:
    # Compute specific heats
    cp = thermodynamic_relations.compute_cp(T, equation_manager.species)
    cv_tr = thermodynamic_relations.compute_cv_tr(T, equation_manager.species)

    # Convert mole fractions to mass fractions
    M_s = equation_manager.species.molar_masses
    Y_M = Y_s * M_s[None, :]
    c_s = Y_M / jnp.sum(Y_M, axis=1, keepdims=True)

    cp_mix = jnp.sum(c_s * cp.T, axis=1)
    cv_tr_mix = jnp.sum(c_s * cv_tr.T, axis=1)
    gamma_frozen = cp_mix / (cv_tr_mix + 1e-14)

    return jnp.sqrt(gamma_frozen * p / (rho + 1e-14))


def compute_flux_faces(
    U_L: Float[Array, "n_faces n_variables"],
    U_R: Float[Array, "n_faces n_variables"],
    n_hat: Float[Array, "n_faces 2"],
    equation_manager: EquationManager2D,
    primitives_L: equation_manager_utils.Primitives2D | None = None,
    primitives_R: equation_manager_utils.Primitives2D | None = None,
) -> Float[Array, "n_faces n_variables"]:
    """Compute numerical flux across faces (normal flux)."""
    # Extract primitives (allow precomputed values)
    if primitives_L is None:
        primitives_L = equation_manager_utils.extract_primitives(U_L, equation_manager)
    if primitives_R is None:
        primitives_R = equation_manager_utils.extract_primitives(U_R, equation_manager)

    Y_L = primitives_L.Y_s
    rho_L = primitives_L.rho
    u_L = primitives_L.u
    v_L = primitives_L.v
    T_L = primitives_L.T
    Tv_L = primitives_L.Tv
    p_L = primitives_L.p

    Y_R = primitives_R.Y_s
    rho_R = primitives_R.rho
    u_R = primitives_R.u
    v_R = primitives_R.v
    T_R = primitives_R.T
    Tv_R = primitives_R.Tv
    p_R = primitives_R.p

    n_x = n_hat[:, 0]
    n_y = n_hat[:, 1]
    t_x = -n_y
    t_y = n_x

    u_n_L = u_L * n_x + v_L * n_y
    u_t_L = u_L * t_x + v_L * t_y
    u_n_R = u_R * n_x + v_R * n_y
    u_t_R = u_R * t_x + v_R * t_y

    n_species = equation_manager.species.n_species

    # Build rotated states U' = [rho_s, rho*u_n, rho*u_t, rho*E, rho*E_v]
    U_Ln = jnp.zeros((U_L.shape[0], n_species + 4))
    U_Rn = jnp.zeros((U_R.shape[0], n_species + 4))

    rho_u_n_L = rho_L * u_n_L
    rho_u_t_L = rho_L * u_t_L
    rho_u_n_R = rho_R * u_n_R
    rho_u_t_R = rho_R * u_t_R

    U_Ln = U_Ln.at[:, :n_species].set(U_L[:, :n_species])
    U_Ln = U_Ln.at[:, n_species].set(rho_u_n_L)
    U_Ln = U_Ln.at[:, n_species + 1].set(rho_u_t_L)
    U_Ln = U_Ln.at[:, n_species + 2].set(U_L[:, n_species + 2])
    U_Ln = U_Ln.at[:, n_species + 3].set(U_L[:, n_species + 3])

    U_Rn = U_Rn.at[:, :n_species].set(U_R[:, :n_species])
    U_Rn = U_Rn.at[:, n_species].set(rho_u_n_R)
    U_Rn = U_Rn.at[:, n_species + 1].set(rho_u_t_R)
    U_Rn = U_Rn.at[:, n_species + 2].set(U_R[:, n_species + 2])
    U_Rn = U_Rn.at[:, n_species + 3].set(U_R[:, n_species + 3])

    # Compute HLLC flux in normal direction
    F_n = compute_flux_normal(U_Ln, U_Rn, p_L, p_R, equation_manager)

    # Map back to Cartesian momentum components
    F_species = F_n[:, :n_species]
    F_mom_n = F_n[:, n_species]
    F_mom_t = F_n[:, n_species + 1]
    F_energy = F_n[:, n_species + 2]
    F_ev = F_n[:, n_species + 3]

    F_rho_u = F_mom_n * n_x + F_mom_t * t_x
    F_rho_v = F_mom_n * n_y + F_mom_t * t_y

    F = jnp.zeros_like(U_L)
    F = F.at[:, :n_species].set(F_species)
    F = F.at[:, n_species].set(F_rho_u)
    F = F.at[:, n_species + 1].set(F_rho_v)
    F = F.at[:, n_species + 2].set(F_energy)
    F = F.at[:, n_species + 3].set(F_ev)

    return F


def compute_flux_normal(
    U_L: Float[Array, "n_faces n_vars"],
    U_R: Float[Array, "n_faces n_vars"],
    p_L: Float[Array, "n_faces"],
    p_R: Float[Array, "n_faces"],
    equation_manager: EquationManager2D,
) -> Float[Array, "n_faces n_vars"]:
    """HLLC flux in the normal direction for extended state."""
    n_species = equation_manager.species.n_species

    # Extract primitives for normal system
    rho_s_L = U_L[:, :n_species]
    rho_s_R = U_R[:, :n_species]
    rho_L = jnp.sum(rho_s_L, axis=1)
    rho_R = jnp.sum(rho_s_R, axis=1)

    rho_u_n_L = U_L[:, n_species]
    rho_u_n_R = U_R[:, n_species]
    u_n_L = rho_u_n_L / rho_L
    u_n_R = rho_u_n_R / rho_R

    # Need temperature for speed of sound
    # Reconstruct full U with dummy tangential momentum
    # We already have full U in caller, use equation_manager_utils via U_L_full?
    # Approximate by using p and rho
    Y_L, _, _, _, T_L, Tv_L, _ = equation_manager_utils.extract_primitives_from_U(
        _merge_normal_to_full(U_L, equation_manager), equation_manager
    )
    Y_R, _, _, _, T_R, Tv_R, _ = equation_manager_utils.extract_primitives_from_U(
        _merge_normal_to_full(U_R, equation_manager), equation_manager
    )

    a_L = compute_speed_of_sound(rho_L, p_L, Y_L, T_L, Tv_L, equation_manager)
    a_R = compute_speed_of_sound(rho_R, p_R, Y_R, T_R, Tv_R, equation_manager)

    a_max = jnp.maximum(a_L, a_R)
    S_L = jnp.minimum(u_n_L, u_n_R) - a_max
    S_R = jnp.maximum(u_n_L, u_n_R) + a_max

    S_star = (
        p_R - p_L + rho_L * u_n_L * (S_L - u_n_L) - rho_R * u_n_R * (S_R - u_n_R)
    ) / (rho_L * (S_L - u_n_L) - rho_R * (S_R - u_n_R) + 1e-14)

    F_L = compute_physical_flux_normal(U_L, p_L, equation_manager)
    F_R = compute_physical_flux_normal(U_R, p_R, equation_manager)

    n_faces, n_vars = U_L.shape
    F = jnp.zeros((n_faces, n_vars))

    mask1 = (S_L >= 0.0)[:, None]
    F = jnp.where(mask1, F_L, F)

    U_star_L = compute_star_state_normal(
        U_L, S_L, S_star, p_L, rho_L, u_n_L, equation_manager
    )
    U_star_R = compute_star_state_normal(
        U_R, S_R, S_star, p_R, rho_R, u_n_R, equation_manager
    )

    F_star_L = F_L + S_L[:, None] * (U_star_L - U_L)
    mask2 = ((S_L < 0.0) & (S_star >= 0.0))[:, None]
    F = jnp.where(mask2, F_star_L, F)

    F_star_R = F_R + S_R[:, None] * (U_star_R - U_R)
    mask3 = ((S_star < 0.0) & (S_R >= 0.0))[:, None]
    F = jnp.where(mask3, F_star_R, F)

    mask4 = (S_R < 0.0)[:, None]
    F = jnp.where(mask4, F_R, F)

    return F


def compute_physical_flux_normal(
    U: Float[Array, "n_faces n_vars"],
    p: Float[Array, "n_faces"],
    equation_manager: EquationManager2D,
) -> Float[Array, "n_faces n_vars"]:
    n_species = equation_manager.species.n_species
    n_faces, n_vars = U.shape

    rho_s = U[:, :n_species]
    rho_u_n = U[:, n_species]
    rho_u_t = U[:, n_species + 1]
    rho_E = U[:, n_species + 2]
    rho_Ev = U[:, n_species + 3]

    rho = jnp.sum(rho_s, axis=1)
    u_n = rho_u_n / rho
    u_t = rho_u_t / rho

    F = jnp.zeros((n_faces, n_vars))
    F = F.at[:, :n_species].set(rho_s * u_n[:, None])
    F = F.at[:, n_species].set(rho_u_n * u_n + p)
    F = F.at[:, n_species + 1].set(rho_u_n * u_t)
    F = F.at[:, n_species + 2].set((rho_E + p) * u_n)
    F = F.at[:, n_species + 3].set(rho_Ev * u_n)

    return F


def compute_star_state_normal(
    U: Float[Array, "n_faces n_vars"],
    S: Float[Array, "n_faces"],
    S_star: Float[Array, "n_faces"],
    p: Float[Array, "n_faces"],
    rho: Float[Array, "n_faces"],
    u_n: Float[Array, "n_faces"],
    equation_manager: EquationManager2D,
) -> Float[Array, "n_faces n_vars"]:
    n_species = equation_manager.species.n_species

    rho_star = rho * (S - u_n) / (S - S_star + 1e-14)

    rho_s = U[:, :n_species]
    rho_u_t = U[:, n_species + 1]
    rho_E = U[:, n_species + 2]
    rho_Ev = U[:, n_species + 3]

    U_star = jnp.zeros_like(U)
    factor = ((S - u_n) / (S - S_star + 1e-14))[:, None]
    U_star = U_star.at[:, :n_species].set(rho_s * factor)
    U_star = U_star.at[:, n_species].set(rho_star * S_star)
    U_star = U_star.at[:, n_species + 1].set(rho_u_t * factor[:, 0])

    p_star_term = p / (S - u_n + 1e-14)
    rho_E_star = factor[:, 0] * (
        rho_E + (S_star - u_n) * (rho_star * S_star + p_star_term)
    )
    U_star = U_star.at[:, n_species + 2].set(rho_E_star)

    rho_Ev_star = (rho_star / rho) * rho_Ev
    U_star = U_star.at[:, n_species + 3].set(rho_Ev_star)

    return U_star


def _merge_normal_to_full(
    U_n: Float[Array, "n_faces n_vars"],
    equation_manager: EquationManager2D,
) -> Float[Array, "n_faces n_variables"]:
    # Convert extended normal state back to full state for thermodynamic extraction
    n_species = equation_manager.species.n_species
    U_full = jnp.zeros((U_n.shape[0], n_species + 4))
    U_full = U_full.at[:, :n_species].set(U_n[:, :n_species])
    # Assume tangential momentum stored at n_species+1 maps to v component
    U_full = U_full.at[:, n_species].set(U_n[:, n_species])
    U_full = U_full.at[:, n_species + 1].set(U_n[:, n_species + 1])
    U_full = U_full.at[:, n_species + 2].set(U_n[:, n_species + 2])
    U_full = U_full.at[:, n_species + 3].set(U_n[:, n_species + 3])
    return U_full
