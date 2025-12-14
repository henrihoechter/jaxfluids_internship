"""Thermodynamic properties for two-temperature model.

This module handles species data loading and thermodynamic property calculations
including specific heats, enthalpies, and vibrational energies following the
formulations in Gnoffo TP-2867.
"""

import json
from pathlib import Path
import jax.numpy as jnp
from jaxtyping import Float, Array

from compressible_1d.two_temperature.config import (
    SpeciesData,
)


def load_species_data(species_name: str) -> SpeciesData:
    """Load species thermodynamic data from JSON file.

    Args:
        species_name: Name of the species (e.g., "N2", "N")

    Returns:
        SpeciesData object containing thermodynamic properties

    Raises:
        FileNotFoundError: If species data file not found
    """
    # Find the data file
    current_file = Path(__file__)
    data_dir = current_file.parent.parent / "data" / "species"
    data_file = data_dir / f"{species_name}.json"

    if not data_file.exists():
        raise FileNotFoundError(
            f"Species data file not found: {data_file}. "
            f"Available species: {[f.stem for f in data_dir.glob('*.json')]}"
        )

    with open(data_file, "r") as f:
        data = json.load(f)

    return SpeciesData(
        name=data["name"],
        molecular_mass=data["molecular_mass"],
        formation_enthalpy=data["formation_enthalpy"],
        theta_v=data.get("theta_v", 0.0),
        theta_rot=data.get("theta_rot", 0.0),
        theta_d=data.get("theta_d", 0.0),
        collision_diameter=data.get("collision_diameter", 0.0),
        nasa_low=tuple(data.get("nasa_low", [0.0] * 7)),
        nasa_high=tuple(data.get("nasa_high", [0.0] * 7)),
        temp_mid=data.get("temp_mid", 1000.0),
    )


def compute_cv_trans_rot(
    T: Float[Array, "..."], species: SpeciesData
) -> Float[Array, "..."]:
    """Compute translational-rotational specific heat at constant volume.

    For diatomic molecules: c_v,tr = (5/2) R/M
    For atoms: c_v,tr = (3/2) R/M

    Args:
        T: Temperature [K]
        species: Species data

    Returns:
        c_v,tr: Translational-rotational specific heat [J/(kg·K)]
    """
    R = 8.314462618  # J/(mol·K)
    M = species.molecular_mass  # kg/mol

    if species.theta_rot > 0:  # Molecule
        c_v_tr = 2.5 * R / M
    else:  # Atom
        c_v_tr = 1.5 * R / M

    return jnp.full_like(T, c_v_tr)


def compute_cv_vib(
    Tv: Float[Array, "..."], species: SpeciesData
) -> Float[Array, "..."]:
    """Compute vibrational specific heat at constant volume.

    Using harmonic oscillator model:
    c_v,vib = R/M * (θ_v/Tv)^2 * exp(θ_v/Tv) / (exp(θ_v/Tv) - 1)^2

    Args:
        Tv: Vibrational temperature [K]
        species: Species data

    Returns:
        c_v,vib: Vibrational specific heat [J/(kg·K)]
    """
    if species.theta_v == 0.0:  # Atom, no vibrational mode
        return jnp.zeros_like(Tv)

    R = 8.314462618  # J/(mol·K)
    M = species.molecular_mass  # kg/mol
    theta_v = species.theta_v

    # Clip Tv to avoid numerical issues
    Tv = jnp.clip(Tv, 10.0, 1e5)

    x = theta_v / Tv
    exp_x = jnp.exp(jnp.clip(x, -50.0, 50.0))

    c_v_vib = (R / M) * x**2 * exp_x / (exp_x - 1.0) ** 2

    return c_v_vib


def compute_e_vib(Tv: Float[Array, "..."], species: SpeciesData) -> Float[Array, "..."]:
    """Compute vibrational energy per unit mass.

    Using harmonic oscillator model (Gnoffo Eq. 8):
    e_v = (R/M) * θ_v / (exp(θ_v/Tv) - 1)

    Args:
        Tv: Vibrational temperature [K]
        species: Species data

    Returns:
        e_v: Vibrational energy [J/kg]
    """
    if species.theta_v == 0.0:  # Atom, no vibrational mode
        return jnp.zeros_like(Tv)

    R = 8.314462618  # J/(mol·K)
    M = species.molecular_mass  # kg/mol
    theta_v = species.theta_v

    # Clip Tv to avoid numerical issues
    Tv = jnp.clip(Tv, 10.0, 1e5)

    x = theta_v / Tv
    exp_x = jnp.exp(jnp.clip(x, -50.0, 50.0))

    e_v = (R / M) * theta_v / (exp_x - 1.0)

    return e_v


def compute_enthalpy_species(
    T: Float[Array, "..."], Tv: Float[Array, "..."], species: SpeciesData
) -> Float[Array, "..."]:
    """Compute total enthalpy per unit mass for a species.

    h = h_f + c_v,tr * T + e_v(Tv) + RT/M

    where h_f is formation enthalpy, and RT/M accounts for p·v work.

    Args:
        T: Translational temperature [K]
        Tv: Vibrational temperature [K]
        species: Species data

    Returns:
        h: Specific enthalpy [J/kg]
    """
    R = 8.314462618  # J/(mol·K)
    M = species.molecular_mass  # kg/mol

    h_f = species.formation_enthalpy
    c_v_tr = compute_cv_trans_rot(T, species)
    e_v = compute_e_vib(Tv, species)

    h = h_f + c_v_tr * T + e_v + (R / M) * T

    return h


def compute_mixture_gamma(
    Y: Float[Array, "n_species ..."],
    T: Float[Array, "..."],
    Tv: Float[Array, "..."],
    species_list: list[SpeciesData],
) -> Float[Array, "..."]:
    """Compute mixture-averaged specific heat ratio.

    γ_mix = c_p,mix / c_v,mix
    where c_v,mix = Σ Y_i (c_v,tr,i + c_v,vib,i)
    and c_p,mix = c_v,mix + R_mix

    Args:
        Y: Mass fractions [n_species, ...]
        T: Translational temperature [K]
        Tv: Vibrational temperature [K]
        species_list: List of species data

    Returns:
        γ_mix: Mixture specific heat ratio
    """
    R = 8.314462618  # J/(mol·K)

    c_v_mix = jnp.zeros_like(T)
    R_mix = jnp.zeros_like(T)

    for i, species in enumerate(species_list):
        c_v_tr = compute_cv_trans_rot(T, species)
        c_v_vib = compute_cv_vib(Tv, species)
        c_v_species = c_v_tr + c_v_vib

        c_v_mix = c_v_mix + Y[i] * c_v_species
        R_mix = R_mix + Y[i] * (R / species.molecular_mass)

    c_p_mix = c_v_mix + R_mix
    gamma_mix = c_p_mix / c_v_mix

    return gamma_mix


def compute_mixture_molecular_mass(
    Y: Float[Array, "n_species ..."], species_list: list[SpeciesData]
) -> Float[Array, "..."]:
    """Compute mixture molecular mass.

    M_mix = 1 / Σ(Y_i / M_i)

    Args:
        Y: Mass fractions [n_species, ...]
        species_list: List of species data

    Returns:
        M_mix: Mixture molecular mass [kg/mol]
    """
    inv_M_mix = jnp.zeros_like(Y[0])

    for i, species in enumerate(species_list):
        inv_M_mix = inv_M_mix + Y[i] / species.molecular_mass

    M_mix = 1.0 / jnp.clip(inv_M_mix, 1e-10, 1e10)

    return M_mix


def compute_speed_of_sound(
    rho: Float[Array, "..."],
    p: Float[Array, "..."],
    Y: Float[Array, "n_species ..."],
    T: Float[Array, "..."],
    Tv: Float[Array, "..."],
    species_list: list[SpeciesData],
) -> Float[Array, "..."]:
    """Compute mixture speed of sound.

    a = sqrt(γ_mix * p / ρ)

    Args:
        rho: Density [kg/m^3]
        p: Pressure [Pa]
        Y: Mass fractions [n_species, ...]
        T: Translational temperature [K]
        Tv: Vibrational temperature [K]
        species_list: List of species data

    Returns:
        a: Speed of sound [m/s]
    """
    gamma_mix = compute_mixture_gamma(Y, T, Tv, species_list)
    a = jnp.sqrt(gamma_mix * p / jnp.clip(rho, 1e-10, 1e10))

    return a


def mass_fractions_to_mole_fractions(
    Y: Float[Array, "n_species ..."], species_list: list[SpeciesData]
) -> Float[Array, "n_species ..."]:
    """Convert mass fractions to mole fractions.

    X_i = (Y_i / M_i) / Σ(Y_j / M_j)

    Args:
        Y: Mass fractions [n_species, ...]
        species_list: List of species data

    Returns:
        X: Mole fractions [n_species, ...]
    """
    n_species = len(species_list)
    Y_over_M = jnp.zeros_like(Y)

    for i, species in enumerate(species_list):
        Y_over_M = Y_over_M.at[i].set(Y[i] / species.molecular_mass)

    sum_Y_over_M = jnp.sum(Y_over_M, axis=0, keepdims=True)
    X = Y_over_M / jnp.clip(sum_Y_over_M, 1e-10, 1e10)

    return X


def mole_fractions_to_mass_fractions(
    X: Float[Array, "n_species ..."], species_list: list[SpeciesData]
) -> Float[Array, "n_species ..."]:
    """Convert mole fractions to mass fractions.

    Y_i = (X_i * M_i) / Σ(X_j * M_j)

    Args:
        X: Mole fractions [n_species, ...]
        species_list: List of species data

    Returns:
        Y: Mass fractions [n_species, ...]
    """
    n_species = len(species_list)
    X_times_M = jnp.zeros_like(X)

    for i, species in enumerate(species_list):
        X_times_M = X_times_M.at[i].set(X[i] * species.molecular_mass)

    sum_X_times_M = jnp.sum(X_times_M, axis=0, keepdims=True)
    Y = X_times_M / jnp.clip(sum_X_times_M, 1e-10, 1e10)

    return Y
