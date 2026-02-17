"""Solver module for computing numerical fluxes.

Implements HLLC Riemann solver for multi-species two-temperature Euler equations.
"""

import jax.numpy as jnp
from jaxtyping import Float, Array

from compressible_1d import (
    equation_manager_types,
    equation_manager_utils,
)
from compressible_core import thermodynamic_relations


def compute_flux(
    U_L: Float[Array, "n_interfaces n_variables"],
    U_R: Float[Array, "n_interfaces n_variables"],
    equation_manager: equation_manager_types.EquationManager,
    primitives_L: equation_manager_utils.Primitives1D | None = None,
    primitives_R: equation_manager_utils.Primitives1D | None = None,
) -> Float[Array, "n_interfaces n_variables"]:
    """Compute numerical flux at cell interfaces using HLLC Riemann solver.

    Args:
        U_L: Left states at interfaces [n_interfaces, n_variables]
        U_R: Right states at interfaces [n_interfaces, n_variables]
        equation_manager: Contains species table and config

    Returns:
        F: Numerical flux

    Notes:
        - HLLC Riemann solver for multi-species two-temperature Euler
        - State vector: [rho_1, ..., rho_ns, rho*u, rho*E, rho*E_v]
    """
    # Extract primitives from left and right states (allow precomputed)
    if primitives_L is None:
        primitives_L = equation_manager_utils.extract_primitives(U_L, equation_manager)
    if primitives_R is None:
        primitives_R = equation_manager_utils.extract_primitives(U_R, equation_manager)

    Y_L, rho_L, T_L, Tv_L, p_L = primitives_L
    Y_R, rho_R, T_R, Tv_R, p_R = primitives_R

    # Extract velocity
    n_species = equation_manager.species.n_species
    rho_u_L = U_L[:, n_species]
    rho_u_R = U_R[:, n_species]
    u_L = rho_u_L / rho_L
    u_R = rho_u_R / rho_R

    # Compute speed of sound
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

    # Compute physical fluxes
    F_L = compute_physical_flux(U_L, p_L, equation_manager)
    F_R = compute_physical_flux(U_R, p_R, equation_manager)

    # HLLC flux
    n_interfaces, n_variables = U_L.shape
    F = jnp.zeros((n_interfaces, n_variables))

    # Region 1: S_L >= 0
    mask1 = (S_L >= 0.0)[:, None]  # [n_interfaces, 1]
    F = jnp.where(mask1, F_L, F)

    # Compute star states for regions 2 and 3
    U_star_L = compute_star_state(U_L, S_L, S_star, p_L, rho_L, u_L, equation_manager)
    U_star_R = compute_star_state(U_R, S_R, S_star, p_R, rho_R, u_R, equation_manager)

    # Region 2: S_L < 0 <= S_star
    F_star_L = F_L + S_L[:, None] * (U_star_L - U_L)
    mask2 = ((S_L < 0.0) & (S_star >= 0.0))[:, None]
    F = jnp.where(mask2, F_star_L, F)

    # Region 3: S_star < 0 <= S_R
    F_star_R = F_R + S_R[:, None] * (U_star_R - U_R)
    mask3 = ((S_star < 0.0) & (S_R >= 0.0))[:, None]
    F = jnp.where(mask3, F_star_R, F)

    # Region 4: S_R < 0
    mask4 = (S_R < 0.0)[:, None]
    F = jnp.where(mask4, F_R, F)

    return F


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

    # Extract variables
    rho_species = U[:, :n_species]  # [n_cells, n_species]
    rho_u = U[:, n_species]  # [n_cells]
    rho_E = U[:, n_species + 1]  # [n_cells]
    rho_Ev = U[:, n_species + 2]  # [n_cells]

    # Compute density and velocity
    rho = jnp.sum(rho_species, axis=1)  # [n_cells]
    u = rho_u / rho  # [n_cells]

    # Initialize flux
    F = jnp.zeros((n_cells, n_variables))

    # Species flux: F_i = ρ_i * u
    F = F.at[:, :n_species].set(rho_species * u[:, None])

    # Momentum flux: F_ρu = ρu^2 + p
    F = F.at[:, n_species].set(rho_u * u + p)

    # Total energy flux: F_ρE = (ρE + p) * u
    F = F.at[:, n_species + 1].set((rho_E + p) * u)

    # Vibrational energy flux: F_ρEv = ρE_v * u
    F = F.at[:, n_species + 2].set(rho_Ev * u)

    return F


def compute_star_state(
    U: Float[Array, "n_cells n_variables"],
    S: Float[Array, "n_cells"],
    S_star: Float[Array, "n_cells"],
    p: Float[Array, "n_cells"],
    rho: Float[Array, "n_cells"],
    u: Float[Array, "n_cells"],
    equation_manager: equation_manager_types.EquationManager,
) -> Float[Array, "n_cells n_variables"]:
    """Compute star state for HLLC solver.

    Args:
        U: Conserved state [n_cells, n_variables]
        S: Wave speed (S_L or S_R) [n_cells]
        S_star: Contact wave speed [n_cells]
        p: Pressure [n_cells]
        rho: Density [n_cells]
        u: Velocity [n_cells]
        equation_manager: Contains species data

    Returns:
        U_star: Star state [n_cells, n_variables]
    """
    n_species = equation_manager.species.n_species

    # Compute star density
    rho_star = rho * (S - u) / (S - S_star + 1e-14)

    # Extract components
    rho_species = U[:, :n_species]
    rho_E = U[:, n_species + 1]
    rho_Ev = U[:, n_species + 2]

    # Star state components
    U_star = jnp.zeros_like(U)

    # Star species densities: ρ_i^* = ρ_i * (S - u) / (S - u*)
    factor = ((S - u) / (S - S_star + 1e-14))[:, None]
    U_star = U_star.at[:, :n_species].set(rho_species * factor)

    # Star momentum: (ρu)^* = ρ^* * u^*
    U_star = U_star.at[:, n_species].set(rho_star * S_star)

    # Star total energy: (ρE)^* = (S - u)/(S - u^*) * [ρE + (u^* - u)*(ρu^* + p/(S - u))]
    p_star_term = p / (S - u + 1e-14)
    rho_E_star = factor[:, 0] * (
        rho_E + (S_star - u) * (rho_star * S_star + p_star_term)
    )
    U_star = U_star.at[:, n_species + 1].set(rho_E_star)

    # Star vibrational energy: (ρE_v)^* = ρ^* / ρ * ρE_v (advected with contact)
    rho_Ev_star = (rho_star / rho) * rho_Ev
    U_star = U_star.at[:, n_species + 2].set(rho_Ev_star)

    return U_star


def compute_speed_of_sound(
    rho: Float[Array, "n_cells"],
    p: Float[Array, "n_cells"],
    Y_s: Float[Array, "n_cells n_species"],
    T: Float[Array, "n_cells"],
    Tv: Float[Array, "n_cells"],
    equation_manager: equation_manager_types.EquationManager,
) -> Float[Array, "n_cells"]:
    """Compute speed of sound for two-temperature multi-species gas.

    a = sqrt(gamma * p / rho)

    where gamma is the mixture-averaged specific heat ratio.

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
    # Compute specific heats
    cp = thermodynamic_relations.compute_cp(
        T, equation_manager.species
    )  # [n_species, n_cells]
    cv_tr = thermodynamic_relations.compute_cv_tr(
        T, equation_manager.species
    )  # [n_species, n_cells]

    # Convert mole fractions to mass fractions for mass-based specific heats
    M_s = equation_manager.species.molar_masses
    Y_M = Y_s * M_s[None, :]
    c_s = Y_M / jnp.sum(Y_M, axis=1, keepdims=True)

    # Mixture-averaged specific heats
    cp_mix = jnp.sum(c_s * cp.T, axis=1)  # [n_cells]
    cv_tr_mix = jnp.sum(c_s * cv_tr.T, axis=1)  # [n_cells]

    # Frozen specific heat ratio (two-temperature model)
    # γ_frozen = cp_mix / cv_tr_mix (vibrational modes frozen on short timescales)
    gamma_frozen = cp_mix / (cv_tr_mix + 1e-14)

    # Speed of sound
    a = jnp.sqrt(gamma_frozen * p / (rho + 1e-14))

    return a
