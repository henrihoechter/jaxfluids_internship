"""Chemical kinetics for two-temperature model.

This module implements Park's two-temperature chemical kinetic model with
configurable controlling temperatures for dissociation reactions, following
Gnoffo TP-2867.
"""

import jax.numpy as jnp
from jaxtyping import Float, Array

from compressible_1d.two_temperature.config import (
    SpeciesData,
    Reaction,
    TwoTemperatureModelConfig,
)


def compute_controlling_temperature(
    T: Float[Array, "..."], Tv: Float[Array, "..."], mode: str
) -> Float[Array, "..."]:
    """Compute rate-controlling temperature for dissociation reactions.

    Gnoffo TP-2867, Eq. 29: Park's geometric mean model
    T_control = sqrt(T * Tv)

    Args:
        T: Translational-rotational temperature [K]
        Tv: Vibrational temperature [K]
        mode: Controlling temperature mode
            "geometric_mean": T_c = sqrt(T * Tv) (Park model)
            "translational": T_c = T
            "harmonic_mean": T_c = 2 * T * Tv / (T + Tv)

    Returns:
        T_control: Controlling temperature [K]
    """
    if mode == "geometric_mean":
        # Park's model: geometric mean
        T_control = jnp.sqrt(T * Tv)
    elif mode == "translational":
        # Use translational temperature only
        T_control = T
    elif mode == "harmonic_mean":
        # Harmonic mean
        T_control = 2.0 * T * Tv / jnp.clip(T + Tv, 1.0, 1e10)
    else:
        raise ValueError(f"Unknown controlling temperature mode: {mode}")

    return T_control


def compute_forward_rate(
    T_control: Float[Array, "..."], reaction: Reaction
) -> Float[Array, "..."]:
    """Compute forward reaction rate coefficient.

    Arrhenius form with controlling temperature:
    k_f = A * T_control^n * exp(-θ_d / T_control)

    Gnoffo TP-2867, Eq. 29-30

    Args:
        T_control: Controlling temperature [K]
        reaction: Reaction data

    Returns:
        k_f: Forward rate coefficient [m^3/(mol·s)] or appropriate units
    """
    A = reaction.A
    n = reaction.n
    theta_d = reaction.theta_d

    # Clip T_control to avoid numerical issues
    T_control = jnp.clip(T_control, 100.0, 1e5)

    k_f = A * T_control**n * jnp.exp(-theta_d / T_control)

    return k_f


def compute_equilibrium_constant(
    T: Float[Array, "..."], reaction: Reaction, species_list: list[SpeciesData]
) -> Float[Array, "..."]:
    """Compute equilibrium constant for a reaction.

    K_eq = exp(-ΔG° / (R*T))

    where ΔG° = Δh_f° - T * Δs°

    Simplified approach using partition functions for N2 dissociation:
    N2 ⇌ 2N
    K_eq(T) = (Q_N^2 / Q_N2) * exp(-D_0 / (k_B * T))

    Args:
        T: Temperature [K]
        reaction: Reaction data
        species_list: List of species data

    Returns:
        K_eq: Equilibrium constant [appropriate units]
    """
    R = 8.314462618  # J/(mol·K)

    # For N2 dissociation: N2 + M → N + N + M
    # This is a simplified form using dissociation energy
    if "N2" in reaction.reactants and "N" in reaction.products:
        # Find N2 species
        N2_species = next(s for s in species_list if s.name == "N2")
        theta_d = N2_species.theta_d  # Dissociation temperature

        # Equilibrium constant (statistical mechanics approach)
        # K_eq ≈ (k_B*T/p_ref)^Δn * Q_stat * exp(-D_0/(k_B*T))
        # Simplified: K_eq ≈ C * T^3.5 * exp(-θ_d/T)
        p_ref = 101325.0  # Pa, reference pressure
        C = 1.5e10  # Pre-exponential constant [appropriate units]

        K_eq = C * T**3.5 * jnp.exp(-theta_d / T)
    else:
        # Generic equilibrium constant (placeholder)
        K_eq = jnp.ones_like(T)

    return K_eq


def compute_backward_rate(
    T: Float[Array, "..."],
    k_f: Float[Array, "..."],
    reaction: Reaction,
    species_list: list[SpeciesData],
) -> Float[Array, "..."]:
    """Compute backward reaction rate coefficient from equilibrium.

    k_b = k_f / K_eq

    Gnoffo TP-2867, Eq. 31

    Args:
        T: Temperature [K]
        k_f: Forward rate coefficient
        reaction: Reaction data
        species_list: List of species data

    Returns:
        k_b: Backward rate coefficient [appropriate units]
    """
    K_eq = compute_equilibrium_constant(T, reaction, species_list)
    k_b = k_f / jnp.clip(K_eq, 1e-30, 1e30)

    return k_b


def get_n2_dissociation_reaction(config: TwoTemperatureModelConfig) -> Reaction:
    """Get N2 dissociation reaction with Park's rate coefficients.

    N2 + M → N + N + M

    Park (1993) rates:
    k_f = 7.0e21 * T^(-1.6) * exp(-113200/T_control) [cm^3/(mol·s)]

    Args:
        config: Model configuration

    Returns:
        Reaction object for N2 dissociation
    """
    # Convert from cm^3/(mol·s) to m^3/(mol·s)
    A = 7.0e21 * 1e-6  # m^3/(mol·s)
    n = -1.6
    theta_d = 113200.0  # K

    # Third body efficiencies (Park 1993)
    third_body_eff = {
        "N2": 1.0,
        "N": 4.0,  # Atoms are more efficient catalysts
    }

    reaction = Reaction(
        name="N2_dissociation",
        reactants=["N2", "M"],
        products=["N", "N", "M"],
        stoich_reactants=[1.0, 1.0],
        stoich_products=[2.0, 1.0],
        A=A,
        n=n,
        theta_d=theta_d,
        third_body_efficiencies=third_body_eff,
    )

    return reaction


def compute_production_rates(
    Y: Float[Array, "n_species ..."],
    rho: Float[Array, "..."],
    T: Float[Array, "..."],
    Tv: Float[Array, "..."],
    species_list: list[SpeciesData],
    config: TwoTemperatureModelConfig,
) -> Float[Array, "n_species ..."]:
    """Compute species production rates from chemical reactions.

    Gnoffo TP-2867, Eq. 1: ω̇_i is the chemical source term

    For N2 dissociation: N2 + M → N + N + M
    ω̇_N2 = -k_f * [N2] * [M] + k_b * [N]^2 * [M]
    ω̇_N = -2 * ω̇_N2

    Args:
        Y: Mass fractions [n_species, ...]
        rho: Density [kg/m^3]
        T: Translational temperature [K]
        Tv: Vibrational temperature [K]
        species_list: List of species data
        config: Model configuration

    Returns:
        omega_dot: Production rates [n_species, ...] [kg/(m^3·s)]
    """
    n_species = len(species_list)
    omega_dot = jnp.zeros_like(Y)

    # Get N2 dissociation reaction
    reaction = get_n2_dissociation_reaction(config)

    # Compute controlling temperature
    T_control = compute_controlling_temperature(
        T, Tv, config.controlling_temperature_mode
    )

    # Forward rate coefficient
    k_f = compute_forward_rate(T_control, reaction)

    # Backward rate coefficient
    k_b = compute_backward_rate(T, k_f, reaction, species_list)

    # Find species indices
    N2_idx = next(i for i, s in enumerate(species_list) if s.name == "N2")
    N_idx = next(i for i, s in enumerate(species_list) if s.name == "N")

    # Molar concentrations [mol/m^3]
    M_N2 = species_list[N2_idx].molecular_mass
    M_N = species_list[N_idx].molecular_mass
    c_N2 = rho * Y[N2_idx] / M_N2  # [mol/m^3]
    c_N = rho * Y[N_idx] / M_N  # [mol/m^3]

    # Third body concentration [mol/m^3]
    c_M = jnp.zeros_like(rho)
    for i, species in enumerate(species_list):
        eff = reaction.third_body_efficiencies.get(species.name, 1.0)
        c_M = c_M + eff * rho * Y[i] / species.molecular_mass

    # Reaction rate [mol/(m^3·s)]
    rate_forward = k_f * c_N2 * c_M
    rate_backward = k_b * c_N**2 * c_M
    rate_net = rate_forward - rate_backward

    # Species production rates [kg/(m^3·s)]
    # N2: consumed at rate -rate_net
    omega_dot = omega_dot.at[N2_idx].set(-M_N2 * rate_net)

    # N: produced at rate +2*rate_net (2 atoms per molecule)
    omega_dot = omega_dot.at[N_idx].set(2.0 * M_N * rate_net)

    return omega_dot


def compute_chemical_timescale(
    Y: Float[Array, "n_species ..."],
    rho: Float[Array, "..."],
    T: Float[Array, "..."],
    Tv: Float[Array, "..."],
    species_list: list[SpeciesData],
    config: TwoTemperatureModelConfig,
) -> Float[Array, "..."]:
    """Compute characteristic chemical timescale.

    τ_chem ≈ ρ_i / |ω̇_i|

    Used for determining subcycling steps and Park's correction to
    vibrational relaxation.

    Args:
        Y: Mass fractions [n_species, ...]
        rho: Density [kg/m^3]
        T: Translational temperature [K]
        Tv: Vibrational temperature [K]
        species_list: List of species data
        config: Model configuration

    Returns:
        tau_chem: Chemical timescale [s]
    """
    omega_dot = compute_production_rates(Y, rho, T, Tv, species_list, config)

    # Find N2 index
    N2_idx = next(i for i, s in enumerate(species_list) if s.name == "N2")

    # Characteristic time based on N2 consumption
    rho_N2 = rho * Y[N2_idx]
    omega_N2 = jnp.abs(omega_dot[N2_idx])

    tau_chem = rho_N2 / jnp.clip(omega_N2, 1e-20, 1e20)

    return tau_chem
