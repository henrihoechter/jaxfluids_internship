"""Transport properties for two-temperature model.

This module implements transport property calculations including binary diffusion,
mixture-averaged diffusion, and thermal conductivity following Gnoffo TP-2867.
"""

import jax.numpy as jnp
from jaxtyping import Float, Array

from compressible_1d.two_temperature.config import (
    SpeciesData,
    TwoTemperatureModelConfig,
)


def compute_binary_diffusion(
    T: Float[Array, "..."],
    p: Float[Array, "..."],
    species_i: SpeciesData,
    species_j: SpeciesData,
) -> Float[Array, "..."]:
    """Compute binary diffusion coefficient.

    Gnoffo TP-2867, Eq. 16 (simplified form):
    D_ij = C_D * T^1.5 / (p * σ_ij^2)

    where C_D ≈ 1.43e-16 [Pa·m^2/K^1.5] includes collision integral effects.

    Args:
        T: Temperature [K]
        p: Pressure [Pa]
        species_i: First species data
        species_j: Second species data

    Returns:
        D_ij: Binary diffusion coefficient [m^2/s]
    """
    # Collision diameter (average of the two species)
    sigma_i = species_i.collision_diameter  # m
    sigma_j = species_j.collision_diameter  # m
    sigma_ij = 0.5 * (sigma_i + sigma_j)  # m

    # Constant from kinetic theory (includes collision integral approximation)
    C_D = 1.43e-16  # Pa·m^2/K^1.5

    # Binary diffusion coefficient
    D_ij = C_D * T**1.5 / (p * sigma_ij**2)

    return D_ij


def compute_mixture_diffusion_coeffs(
    Y: Float[Array, "n_species ..."],
    T: Float[Array, "..."],
    p: Float[Array, "..."],
    species_list: list[SpeciesData],
    config: TwoTemperatureModelConfig,
) -> Float[Array, "n_species ..."]:
    """Compute mixture-averaged diffusion coefficients.

    Gnoffo TP-2867, Eq. 13 (mixture-averaged formulation):
    D_i = (1 - X_i) / Σ_{j≠i} (X_j / D_ij)

    Args:
        Y: Mass fractions [n_species, ...]
        T: Temperature [K]
        p: Pressure [Pa]
        species_list: List of species data
        config: Model configuration

    Returns:
        D_i: Mixture diffusion coefficients [n_species, ...] [m^2/s]
    """
    if config.diffusion_model == "constant":
        # Use constant diffusion coefficient for all species
        D_const = config.constant_diffusion_coeff
        return jnp.full_like(Y, D_const)

    # Convert mass fractions to mole fractions
    from compressible_1d.two_temperature.thermodynamics import (
        mass_fractions_to_mole_fractions,
    )

    X = mass_fractions_to_mole_fractions(Y, species_list)

    n_species = len(species_list)
    D_mix = jnp.zeros_like(Y)

    for i in range(n_species):
        # Compute Σ_{j≠i} (X_j / D_ij)
        sum_term = jnp.zeros_like(T)

        for j in range(n_species):
            if i != j:
                D_ij = compute_binary_diffusion(T, p, species_list[i], species_list[j])
                sum_term = sum_term + X[j] / jnp.clip(D_ij, 1e-10, 1e10)

        # D_i = (1 - X_i) / sum_term
        D_i = (1.0 - X[i]) / jnp.clip(sum_term, 1e-10, 1e10)
        D_mix = D_mix.at[i].set(D_i)

    return D_mix


def compute_thermal_conductivity_eucken(
    T: Float[Array, "..."], species: SpeciesData
) -> Float[Array, "..."]:
    """Compute thermal conductivity using Eucken formula.

    Eucken formula for monatomic and diatomic gases:
    λ = (15/4) * (R/M) * μ  for atoms
    λ = (5/2) * c_v * μ + (R/M) * μ  for molecules

    Using Chapman-Enskog theory for viscosity:
    μ ≈ C_μ * sqrt(M*T) / σ^2

    Args:
        T: Temperature [K]
        species: Species data

    Returns:
        λ: Thermal conductivity [W/(m·K)]
    """
    R = 8.314462618  # J/(mol·K)
    M = species.molecular_mass  # kg/mol
    sigma = species.collision_diameter  # m

    # Chapman-Enskog viscosity approximation
    C_mu = 2.67e-6  # kg·K^(-0.5)/(m·s)
    mu = C_mu * jnp.sqrt(M * T) / sigma**2

    if species.theta_rot > 0:  # Molecule
        # For diatomic: λ = 2.5 * c_v * μ + (R/M) * μ
        c_v = 2.5 * R / M  # Translational + rotational
        lambda_thermal = 2.5 * c_v * mu + (R / M) * mu
    else:  # Atom
        # For monatomic: λ = 3.75 * (R/M) * μ
        lambda_thermal = 3.75 * (R / M) * mu

    return lambda_thermal


def compute_mixture_thermal_conductivity(
    Y: Float[Array, "n_species ..."],
    T: Float[Array, "..."],
    species_list: list[SpeciesData],
) -> Float[Array, "..."]:
    """Compute mixture thermal conductivity.

    Using Wilke's mixing rule:
    λ_mix = Σ_i (X_i * λ_i / φ_i)
    where φ_i = Σ_j (X_j * [1 + sqrt(μ_i/μ_j) * (M_j/M_i)^(1/4)]^2 / sqrt(8*(1 + M_i/M_j)))

    Simplified approximation: λ_mix ≈ Σ_i (Y_i * λ_i)

    Args:
        Y: Mass fractions [n_species, ...]
        T: Temperature [K]
        species_list: List of species data

    Returns:
        λ_mix: Mixture thermal conductivity [W/(m·K)]
    """
    lambda_mix = jnp.zeros_like(T)

    for i, species in enumerate(species_list):
        lambda_i = compute_thermal_conductivity_eucken(T, species)
        lambda_mix = lambda_mix + Y[i] * lambda_i

    return lambda_mix


def compute_viscosity_mixture(
    Y: Float[Array, "n_species ..."],
    T: Float[Array, "..."],
    species_list: list[SpeciesData],
) -> Float[Array, "..."]:
    """Compute mixture dynamic viscosity.

    Using Chapman-Enskog theory with mass-weighted averaging:
    μ_mix ≈ Σ_i (Y_i * μ_i)

    Args:
        Y: Mass fractions [n_species, ...]
        T: Temperature [K]
        species_list: List of species data

    Returns:
        μ_mix: Mixture dynamic viscosity [Pa·s]
    """
    mu_mix = jnp.zeros_like(T)
    C_mu = 2.67e-6  # kg·K^(-0.5)/(m·s)

    for i, species in enumerate(species_list):
        M = species.molecular_mass
        sigma = species.collision_diameter
        mu_i = C_mu * jnp.sqrt(M * T) / sigma**2
        mu_mix = mu_mix + Y[i] * mu_i

    return mu_mix
