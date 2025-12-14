"""Solver adapter for two-temperature model.

This module provides state conversion functions and integration with the
finite volume solver for the two-temperature multi-species system.
"""

import jax.numpy as jnp
from jaxtyping import Float, Array

from compressible_1d.two_temperature.config import (
    SpeciesData,
    TwoTemperatureModelConfig,
)
from compressible_1d.two_temperature import thermodynamics


def U_to_primitives(
    U: Float[Array, "n_conserved ..."],
    species_list: list[SpeciesData],
    config: TwoTemperatureModelConfig,
) -> tuple:
    """Convert conserved variables to primitive variables.

    State vector: U = [ρ_1, ρ_2, ..., ρ_ns, ρu, ρE, ρE_v]
    Primitives: (Y, u, T, Tv, p)

    Args:
        U: Conserved variables [n_conserved, ...]
        species_list: List of species data
        config: Model configuration

    Returns:
        Y: Mass fractions [n_species, ...]
        u: Velocity [m/s]
        T: Translational temperature [K]
        Tv: Vibrational temperature [K]
        p: Pressure [Pa]
        rho: Density [kg/m^3]
    """
    from compressible_1d.two_temperature.source_terms import extract_primitives_from_U

    Y, rho, T, Tv, p = extract_primitives_from_U(U, species_list, config)

    # Extract velocity
    n_species = config.n_species
    rho_u = U[n_species]
    u = rho_u / rho

    return Y, u, T, Tv, p, rho


def primitives_to_U(
    Y: Float[Array, "n_species ..."],
    u: Float[Array, "..."],
    T: Float[Array, "..."],
    Tv: Float[Array, "..."],
    rho: Float[Array, "..."],
    species_list: list[SpeciesData],
    config: TwoTemperatureModelConfig,
) -> Float[Array, "n_conserved ..."]:
    """Convert primitive variables to conserved variables.

    Primitives: (Y, u, T, Tv, rho)
    State vector: U = [ρ_1, ρ_2, ..., ρ_ns, ρu, ρE, ρE_v]

    Args:
        Y: Mass fractions [n_species, ...]
        u: Velocity [m/s]
        T: Translational temperature [K]
        Tv: Vibrational temperature [K]
        rho: Density [kg/m^3]
        species_list: List of species data
        config: Model configuration

    Returns:
        U: Conserved variables [n_conserved, ...]
    """
    n_species = config.n_species
    n_conserved = config.n_conserved

    # Initialize state vector
    U = jnp.zeros((n_conserved,) + rho.shape)

    # Partial densities
    for i in range(n_species):
        U = U.at[i].set(rho * Y[i])

    # Momentum density
    U = U.at[n_species].set(rho * u)

    # Translational-rotational energy per unit mass
    e_tr = jnp.zeros_like(rho)
    for i, species in enumerate(species_list):
        c_v_tr = thermodynamics.compute_cv_trans_rot(T, species)
        e_tr = e_tr + Y[i] * c_v_tr * T

    # Vibrational energy per unit mass
    e_v = jnp.zeros_like(rho)
    for i, species in enumerate(species_list):
        e_v_species = thermodynamics.compute_e_vib(Tv, species)
        e_v = e_v + Y[i] * e_v_species

    # Kinetic energy per unit mass
    e_kin = 0.5 * u**2

    # Total energy per unit mass
    E_total = e_tr + e_v + e_kin

    # Total energy density
    U = U.at[n_species + 1].set(rho * E_total)

    # Vibrational energy density
    U = U.at[n_species + 2].set(rho * e_v)

    return U


def initialize_two_temperature_shock_tube(
    Y_left: Float[Array, "n_species"],
    u_left: float,
    T_left: float,
    Tv_left: float,
    rho_left: float,
    Y_right: Float[Array, "n_species"],
    u_right: float,
    T_right: float,
    Tv_right: float,
    rho_right: float,
    n_cells: int,
    species_list: list[SpeciesData],
    config: TwoTemperatureModelConfig,
) -> Float[Array, "n_conserved n_cells"]:
    """Initialize a shock tube with two different states.

    Args:
        Y_left: Mass fractions on the left [n_species]
        u_left: Velocity on the left [m/s]
        T_left: Translational temperature on the left [K]
        Tv_left: Vibrational temperature on the left [K]
        rho_left: Density on the left [kg/m^3]
        Y_right: Mass fractions on the right [n_species]
        u_right: Velocity on the right [m/s]
        T_right: Translational temperature on the right [K]
        Tv_right: Vibrational temperature on the right [K]
        rho_right: Density on the right [kg/m^3]
        n_cells: Number of cells
        species_list: List of species data
        config: Model configuration

    Returns:
        U: Initial conserved state [n_conserved, n_cells]
    """
    n_conserved = config.n_conserved
    mid = n_cells // 2

    # Create arrays for left and right states
    Y_field = jnp.zeros((config.n_species, n_cells))
    u_field = jnp.zeros(n_cells)
    T_field = jnp.zeros(n_cells)
    Tv_field = jnp.zeros(n_cells)
    rho_field = jnp.zeros(n_cells)

    # Set left state
    for i in range(config.n_species):
        Y_field = Y_field.at[i, :mid].set(Y_left[i])
    u_field = u_field.at[:mid].set(u_left)
    T_field = T_field.at[:mid].set(T_left)
    Tv_field = Tv_field.at[:mid].set(Tv_left)
    rho_field = rho_field.at[:mid].set(rho_left)

    # Set right state
    for i in range(config.n_species):
        Y_field = Y_field.at[i, mid:].set(Y_right[i])
    u_field = u_field.at[mid:].set(u_right)
    T_field = T_field.at[mid:].set(T_right)
    Tv_field = Tv_field.at[mid:].set(Tv_right)
    rho_field = rho_field.at[mid:].set(rho_right)

    # Convert to conserved variables
    U = primitives_to_U(
        Y_field, u_field, T_field, Tv_field, rho_field, species_list, config
    )

    return U


def compute_flux_two_temperature(
    U_L: Float[Array, "n_conserved ..."],
    U_R: Float[Array, "n_conserved ..."],
    species_list: list[SpeciesData],
    config: TwoTemperatureModelConfig,
) -> Float[Array, "n_conserved ..."]:
    """Compute numerical flux for two-temperature system.

    Uses HLLC Riemann solver extended to multi-species system.

    Args:
        U_L: Left state [n_conserved, ...]
        U_R: Right state [n_conserved, ...]
        species_list: List of species data
        config: Model configuration

    Returns:
        F: Numerical flux [n_conserved, ...]
    """
    # Convert to primitives
    Y_L, u_L, T_L, Tv_L, p_L, rho_L = U_to_primitives(U_L, species_list, config)
    Y_R, u_R, T_R, Tv_R, p_R, rho_R = U_to_primitives(U_R, species_list, config)

    # Compute speed of sound
    a_L = thermodynamics.compute_speed_of_sound(
        rho_L, p_L, Y_L, T_L, Tv_L, species_list
    )
    a_R = thermodynamics.compute_speed_of_sound(
        rho_R, p_R, Y_R, T_R, Tv_R, species_list
    )

    # Wave speed estimates
    a_max = jnp.maximum(a_L, a_R)
    S_L = jnp.minimum(u_L, u_R) - a_max
    S_R = jnp.maximum(u_L, u_R) + a_max

    # Contact wave speed
    S_star = (p_R - p_L + rho_L * u_L * (S_L - u_L) - rho_R * u_R * (S_R - u_R)) / (
        rho_L * (S_L - u_L) - rho_R * (S_R - u_R)
    )

    # Compute physical fluxes
    F_L = compute_physical_flux(U_L, p_L, species_list, config)
    F_R = compute_physical_flux(U_R, p_R, species_list, config)

    # HLLC flux
    n_conserved = config.n_conserved
    F = jnp.zeros_like(U_L)

    # Broadcast conditions to match conserved variable shape [n_conserved, ...]
    # S_L, S_R, S_star have shape [...], need to broadcast to [n_conserved, ...]
    cond_shape = (n_conserved,) + S_L.shape

    # Region 1: S_L >= 0
    cond1 = jnp.broadcast_to(S_L >= 0.0, cond_shape)
    F = jnp.where(cond1, F_L, F)

    # Compute star states for regions 2 and 3
    U_star_L = compute_star_state(U_L, S_L, S_star, p_L, rho_L, u_L)
    U_star_R = compute_star_state(U_R, S_R, S_star, p_R, rho_R, u_R)

    # Region 2: S_L < 0 <= S_star
    F_star_L = F_L + S_L * (U_star_L - U_L)
    cond2 = jnp.broadcast_to(jnp.logical_and(S_L < 0.0, S_star >= 0.0), cond_shape)
    F = jnp.where(cond2, F_star_L, F)

    # Region 3: S_star < 0 <= S_R
    F_star_R = F_R + S_R * (U_star_R - U_R)
    cond3 = jnp.broadcast_to(jnp.logical_and(S_star < 0.0, S_R >= 0.0), cond_shape)
    F = jnp.where(cond3, F_star_R, F)

    # Region 4: S_R < 0
    cond4 = jnp.broadcast_to(S_R < 0.0, cond_shape)
    F = jnp.where(cond4, F_R, F)

    return F


def compute_physical_flux(
    U: Float[Array, "n_conserved ..."],
    p: Float[Array, "..."],
    species_list: list[SpeciesData],
    config: TwoTemperatureModelConfig,
) -> Float[Array, "n_conserved ..."]:
    """Compute physical flux for conserved variables.

    F = [ρ_i*u, ρu^2 + p, (ρE + p)*u, ρE_v*u]

    Args:
        U: Conserved variables [n_conserved, ...]
        p: Pressure [Pa]
        species_list: List of species data
        config: Model configuration

    Returns:
        F: Physical flux [n_conserved, ...]
    """
    n_species = config.n_species
    n_conserved = config.n_conserved

    # Extract variables
    rho_species = U[:n_species]
    rho_u = U[n_species]
    rho_E = U[n_species + 1]
    rho_Ev = U[n_species + 2]

    # Compute density and velocity
    rho = jnp.sum(rho_species, axis=0)
    u = rho_u / jnp.clip(rho, 1e-10, 1e10)

    # Initialize flux
    F = jnp.zeros_like(U)

    # Species fluxes
    for i in range(n_species):
        F = F.at[i].set(rho_species[i] * u)

    # Momentum flux
    F = F.at[n_species].set(rho_u * u + p)

    # Total energy flux
    F = F.at[n_species + 1].set((rho_E + p) * u)

    # Vibrational energy flux
    F = F.at[n_species + 2].set(rho_Ev * u)

    return F


def compute_star_state(
    U: Float[Array, "n_conserved ..."],
    S: Float[Array, "..."],
    S_star: Float[Array, "..."],
    p: Float[Array, "..."],
    rho: Float[Array, "..."],
    u: Float[Array, "..."],
) -> Float[Array, "n_conserved ..."]:
    """Compute star region state for HLLC solver.

    Args:
        U: Conserved state [n_conserved, ...]
        S: Wave speed (S_L or S_R)
        S_star: Contact wave speed
        p: Pressure
        rho: Density
        u: Velocity

    Returns:
        U_star: Star region state [n_conserved, ...]
    """
    n_conserved = U.shape[0]

    # Avoid division by zero (but preserve sign for negative denominators!)
    eps = 1e-10
    denom = S - S_star
    # Use sign-preserving epsilon: if |denom| < eps, use sign(denom) * eps
    denom_safe = jnp.where(jnp.abs(denom) < eps, jnp.sign(denom) * eps, denom)

    factor = rho * (S - u) / denom_safe

    U_star = jnp.zeros_like(U)

    # Species: ρ_i* = ρ_i * (S - u) / (S - S_star)
    for i in range(n_conserved - 3):  # All species
        U_star = U_star.at[i].set(U[i] * (S - u) / denom_safe)

    # Momentum: (ρu)* = ρ* · S_star
    U_star = U_star.at[n_conserved - 3].set(factor * S_star)

    # Star pressure
    p_star = p + rho * (S - u) * (S_star - u)

    # Total energy: (ρE)* = [(S - u) * ρE - p * u + p_star * S_star] / (S - S_star)
    rho_E = U[n_conserved - 2]
    rho_u = U[n_conserved - 3]
    rho_E_star = ((S - u) * rho_E - p * u + p_star * S_star) / denom_safe
    U_star = U_star.at[n_conserved - 2].set(rho_E_star)

    # Vibrational energy: (ρE_v)* = (ρE_v) * (S - u) / (S - S_star)
    U_star = U_star.at[n_conserved - 1].set(U[n_conserved - 1] * (S - u) / denom_safe)

    return U_star
