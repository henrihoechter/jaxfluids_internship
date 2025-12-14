"""Vibrational relaxation for two-temperature model.

This module implements Millikan-White vibrational relaxation with Park's
high-temperature correction and preferential dissociation model, following
Gnoffo TP-2867.
"""

import jax.numpy as jnp
from jaxtyping import Float, Array

from compressible_1d.two_temperature.config import (
    SpeciesData,
    TwoTemperatureModelConfig,
)
from compressible_1d.two_temperature.thermodynamics import (
    compute_e_vib,
    mass_fractions_to_mole_fractions,
)


def compute_millikan_white_time(
    T: Float[Array, "..."],
    p: Float[Array, "..."],
    species_i: SpeciesData,
    species_j: SpeciesData,
) -> Float[Array, "..."]:
    """Compute Millikan-White vibrational relaxation time for species pair.

    Gnoffo TP-2867, Eq. 36:
    τ_MW,ij = (1/p) * exp[A_ij(T^(-1/3) - 0.015*μ_ij^(1/4)) - 18.42]

    where:
    - A_ij = 1.16e-3 * μ_ij^(1/2) * θ_v^(4/3)
    - μ_ij = M_i * M_j / (M_i + M_j) is reduced mass
    - p is in atmospheres
    - τ is in seconds

    Args:
        T: Translational temperature [K]
        p: Pressure [Pa]
        species_i: Vibrating species (must have theta_v > 0)
        species_j: Collision partner

    Returns:
        tau_MW: Millikan-White relaxation time [s]
    """
    if species_i.theta_v == 0.0:
        # No vibrational mode, return infinite relaxation time
        return jnp.full_like(T, 1e20)

    # Convert pressure to atmospheres
    p_atm = p / 101325.0

    # Reduced mass [kg/mol]
    M_i = species_i.molecular_mass
    M_j = species_j.molecular_mass
    mu_ij = M_i * M_j / (M_i + M_j)

    # Characteristic vibrational temperature
    theta_v = species_i.theta_v

    # Millikan-White parameter
    A_ij = 1.16e-3 * mu_ij**0.5 * theta_v ** (4.0 / 3.0)

    # Clip T to avoid numerical issues
    T_clip = jnp.clip(T, 100.0, 20000.0)

    # Millikan-White correlation
    exponent = A_ij * (T_clip ** (-1.0 / 3.0) - 0.015 * mu_ij**0.25) - 18.42
    exponent = jnp.clip(exponent, -50.0, 50.0)  # Prevent overflow

    tau_MW = (1.0 / jnp.clip(p_atm, 1e-6, 1e6)) * jnp.exp(exponent)

    return tau_MW


def compute_park_correction(
    tau_MW: Float[Array, "..."], tau_chem: Float[Array, "..."]
) -> Float[Array, "..."]:
    """Compute Park's high-temperature correction to relaxation time.

    Gnoffo TP-2867, Eq. 37:
    1/τ_v = 1/τ_MW + 1/τ_chem

    At high temperatures, chemical reactions provide an additional
    pathway for vibrational energy exchange (preferential dissociation).

    Args:
        tau_MW: Millikan-White relaxation time [s]
        tau_chem: Chemical timescale [s]

    Returns:
        tau_v: Effective vibrational relaxation time with Park correction [s]
    """
    inv_tau_v = 1.0 / jnp.clip(tau_MW, 1e-20, 1e20) + 1.0 / jnp.clip(
        tau_chem, 1e-20, 1e20
    )
    tau_v = 1.0 / jnp.clip(inv_tau_v, 1e-20, 1e20)

    return tau_v


def compute_mixture_relaxation_time(
    Y: Float[Array, "n_species ..."],
    T: Float[Array, "..."],
    p: Float[Array, "..."],
    tau_chem: Float[Array, "..."],
    species_list: list[SpeciesData],
    config: TwoTemperatureModelConfig,
) -> Float[Array, "..."]:
    """Compute mixture-averaged vibrational relaxation time.

    For N2-N mixture:
    1/τ_v,mix = Σ_j (X_j / τ_MW,N2-j)

    with Park's correction applied to the mixture average.

    Args:
        Y: Mass fractions [n_species, ...]
        T: Translational temperature [K]
        p: Pressure [Pa]
        tau_chem: Chemical timescale [s]
        species_list: List of species data
        config: Model configuration

    Returns:
        tau_v: Mixture vibrational relaxation time [s]
    """
    if config.relaxation_model == "none":
        # No relaxation
        return jnp.full_like(T, 1e20)

    # Convert to mole fractions
    X = mass_fractions_to_mole_fractions(Y, species_list)

    # Find vibrating species (N2)
    vib_idx = None
    for i, species in enumerate(species_list):
        if species.theta_v > 0:
            vib_idx = i
            break

    if vib_idx is None:
        # No vibrating species
        return jnp.full_like(T, 1e20)

    vib_species = species_list[vib_idx]

    # Compute mixture-averaged Millikan-White time
    inv_tau_MW_mix = jnp.zeros_like(T)

    for j, collision_partner in enumerate(species_list):
        tau_MW_ij = compute_millikan_white_time(T, p, vib_species, collision_partner)
        inv_tau_MW_mix = inv_tau_MW_mix + X[j] / jnp.clip(tau_MW_ij, 1e-20, 1e20)

    tau_MW_mix = 1.0 / jnp.clip(inv_tau_MW_mix, 1e-20, 1e20)

    # Apply Park's correction
    if config.relaxation_model == "millikan_white_park":
        tau_v = compute_park_correction(tau_MW_mix, tau_chem)
    else:
        tau_v = tau_MW_mix

    return tau_v


def compute_vibrational_energy_source(
    Y: Float[Array, "n_species ..."],
    rho: Float[Array, "..."],
    T: Float[Array, "..."],
    Tv: Float[Array, "..."],
    p: Float[Array, "..."],
    omega_dot: Float[Array, "n_species ..."],
    tau_v: Float[Array, "..."],
    species_list: list[SpeciesData],
    config: TwoTemperatureModelConfig,
) -> Float[Array, "..."]:
    """Compute vibrational energy source term.

    Gnoffo TP-2867, Eq. 32:
    Q̇_v = ρ * (e_v(T) - e_v(Tv)) / τ_v - Σ_i(e_v,i * ω̇_i)

    The first term represents thermal relaxation toward equilibrium.
    The second term accounts for vibrational energy carried away by
    dissociating molecules (preferential dissociation).

    Args:
        Y: Mass fractions [n_species, ...]
        rho: Density [kg/m^3]
        T: Translational temperature [K]
        Tv: Vibrational temperature [K]
        p: Pressure [Pa]
        omega_dot: Species production rates [n_species, ...] [kg/(m^3·s)]
        tau_v: Vibrational relaxation time [s]
        species_list: List of species data
        config: Model configuration

    Returns:
        Q_dot_v: Vibrational energy source [W/m^3]
    """
    Q_dot_v = jnp.zeros_like(T)

    for i, species in enumerate(species_list):
        if species.theta_v > 0:  # Only vibrating species contribute
            # Equilibrium vibrational energy at T
            e_v_eq = compute_e_vib(T, species)

            # Current vibrational energy at Tv
            e_v_current = compute_e_vib(Tv, species)

            # Thermal relaxation term
            Q_relax = rho * Y[i] * (e_v_eq - e_v_current) / jnp.clip(tau_v, 1e-20, 1e20)
            Q_dot_v = Q_dot_v + Q_relax

            # Preferential dissociation term (vibrational energy lost with dissociating molecules)
            if config.dissociation_model == "preferential":
                Q_diss = -e_v_current * omega_dot[i]
                Q_dot_v = Q_dot_v + Q_diss

    return Q_dot_v


def compute_vibrational_energy_density(
    Y: Float[Array, "n_species ..."],
    rho: Float[Array, "..."],
    Tv: Float[Array, "..."],
    species_list: list[SpeciesData],
) -> Float[Array, "..."]:
    """Compute total vibrational energy per unit volume.

    ρ*E_v = Σ_i (ρ * Y_i * e_v,i(Tv))

    Args:
        Y: Mass fractions [n_species, ...]
        rho: Density [kg/m^3]
        Tv: Vibrational temperature [K]
        species_list: List of species data

    Returns:
        rho_Ev: Vibrational energy density [J/m^3]
    """
    rho_Ev = jnp.zeros_like(rho)

    for i, species in enumerate(species_list):
        if species.theta_v > 0:
            e_v = compute_e_vib(Tv, species)
            rho_Ev = rho_Ev + rho * Y[i] * e_v

    return rho_Ev
