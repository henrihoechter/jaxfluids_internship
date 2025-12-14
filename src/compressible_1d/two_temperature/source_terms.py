"""Source term integration for two-temperature model.

This module implements chemistry integration with subcycling and operator
splitting for the two-temperature model source terms.
"""

import jax.numpy as jnp
from jaxtyping import Float, Array

from compressible_1d.two_temperature.config import (
    SpeciesData,
    TwoTemperatureModelConfig,
)
from compressible_1d.two_temperature import thermodynamics, kinetics, relaxation


def extract_primitives_from_U(
    U: Float[Array, "n_conserved ..."],
    species_list: list[SpeciesData],
    config: TwoTemperatureModelConfig,
) -> tuple[
    Float[Array, "n_species ..."],
    Float[Array, "..."],
    Float[Array, "..."],
    Float[Array, "..."],
    Float[Array, "..."],
]:
    """Extract primitive variables from conserved state vector.

    State vector: U = [ρ_1, ρ_2, ..., ρ_ns, ρu, ρE, ρE_v]

    Args:
        U: Conserved variables [n_conserved, ...]
        species_list: List of species data
        config: Model configuration

    Returns:
        Y: Mass fractions [n_species, ...]
        rho: Total density [kg/m^3]
        T: Translational temperature [K]
        Tv: Vibrational temperature [K]
        p: Pressure [Pa]
    """
    n_species = config.n_species

    # Extract partial densities
    rho_species = U[:n_species]  # [n_species, ...]
    rho_u = U[n_species]
    rho_E = U[n_species + 1]
    rho_Ev = U[n_species + 2]

    # Total density
    rho = jnp.sum(rho_species, axis=0)
    rho = jnp.clip(rho, 1e-10, 1e10)

    # Mass fractions
    Y = rho_species / rho

    # Velocity
    u = rho_u / rho

    # Kinetic energy
    E_kin = 0.5 * u**2

    # Vibrational energy per unit mass
    E_v = rho_Ev / rho

    # Total energy per unit mass
    E_total = rho_E / rho

    # Translational-rotational energy
    E_tr = E_total - E_v - E_kin

    # Solve for translational temperature iteratively
    # E_tr = Σ_i (Y_i * c_v,tr,i * T)
    # Simplified: T ≈ E_tr / c_v,tr,mix
    c_v_tr_mix = jnp.zeros_like(rho)
    R = 8.314462618
    for i, species in enumerate(species_list):
        c_v_tr = thermodynamics.compute_cv_trans_rot(
            jnp.ones_like(rho) * 300.0, species
        )
        c_v_tr_mix = c_v_tr_mix + Y[i] * c_v_tr

    T = E_tr / jnp.clip(c_v_tr_mix, 1e-10, 1e10)
    T = jnp.clip(T, 100.0, 50000.0)

    # Solve for vibrational temperature
    # E_v = Σ_i (Y_i * e_v,i(Tv))
    # Use Newton iteration (simplified: assume single vibrating species)
    Tv = T  # Initial guess
    for _ in range(5):  # Newton iterations
        e_v_mix = jnp.zeros_like(rho)
        de_v_dTv_mix = jnp.zeros_like(rho)

        for i, species in enumerate(species_list):
            if species.theta_v > 0:
                e_v = thermodynamics.compute_e_vib(Tv, species)
                c_v_vib = thermodynamics.compute_cv_vib(Tv, species)
                e_v_mix = e_v_mix + Y[i] * e_v
                de_v_dTv_mix = de_v_dTv_mix + Y[i] * c_v_vib

        residual = e_v_mix - E_v
        Tv = Tv - residual / jnp.clip(de_v_dTv_mix, 1e-10, 1e10)
        Tv = jnp.clip(Tv, 100.0, 50000.0)

    # Compute pressure
    M_mix = thermodynamics.compute_mixture_molecular_mass(Y, species_list)
    p = rho * (R / M_mix) * T
    p = jnp.clip(p, 1.0, 1e10)

    return Y, rho, T, Tv, p


def apply_chemistry_substep(
    U: Float[Array, "n_conserved ..."],
    dt: float,
    species_list: list[SpeciesData],
    config: TwoTemperatureModelConfig,
) -> Float[Array, "n_conserved ..."]:
    """Apply a single chemistry substep.

    Updates species densities and vibrational energy based on:
    - Chemical production/consumption
    - Vibrational relaxation
    - Preferential dissociation

    Args:
        U: Conserved state vector [n_conserved, ...]
        dt: Time step [s]
        species_list: List of species data
        config: Model configuration

    Returns:
        U_new: Updated state vector [n_conserved, ...]
    """
    n_species = config.n_species

    # Extract primitives
    Y, rho, T, Tv, p = extract_primitives_from_U(U, species_list, config)

    # Compute chemical production rates
    omega_dot = kinetics.compute_production_rates(Y, rho, T, Tv, species_list, config)

    # Compute chemical timescale
    tau_chem = kinetics.compute_chemical_timescale(Y, rho, T, Tv, species_list, config)

    # Compute vibrational relaxation time
    tau_v = relaxation.compute_mixture_relaxation_time(
        Y, T, p, tau_chem, species_list, config
    )

    # Compute vibrational energy source
    Q_dot_v = relaxation.compute_vibrational_energy_source(
        Y, rho, T, Tv, p, omega_dot, tau_v, species_list, config
    )

    # Update species densities (explicit Euler)
    U_new = U.at[:n_species].set(U[:n_species] + dt * omega_dot)

    # Ensure non-negative densities and normalize
    U_new = U_new.at[:n_species].set(jnp.clip(U_new[:n_species], 0.0, 1e10))
    rho_total = jnp.sum(U_new[:n_species], axis=0, keepdims=True)
    U_new = U_new.at[:n_species].set(U_new[:n_species] * rho / rho_total)

    # Update vibrational energy (explicit Euler)
    U_new = U_new.at[n_species + 2].set(U[n_species + 2] + dt * Q_dot_v)

    # Ensure non-negative vibrational energy
    U_new = U_new.at[n_species + 2].set(jnp.clip(U_new[n_species + 2], 0.0, 1e10))

    return U_new


def apply_chemistry_source(
    U: Float[Array, "n_conserved ..."],
    dt: float,
    species_list: list[SpeciesData],
    config: TwoTemperatureModelConfig,
) -> Float[Array, "n_conserved ..."]:
    """Apply chemistry source terms with subcycling.

    Integrates chemistry and vibrational relaxation over time dt using
    multiple substeps for numerical stability with stiff source terms.

    Args:
        U: Conserved state vector [n_conserved, ...]
        dt: Time step [s]
        species_list: List of species data
        config: Model configuration

    Returns:
        U_new: Updated state vector [n_conserved, ...]
    """
    n_substeps = config.chemistry_substeps
    dt_sub = dt / n_substeps

    U_current = U
    for _ in range(n_substeps):
        U_current = apply_chemistry_substep(U_current, dt_sub, species_list, config)

    return U_current


def strang_splitting_step(
    U: Float[Array, "n_conserved ..."],
    dt: float,
    delta_x: float,
    species_list: list[SpeciesData],
    config: TwoTemperatureModelConfig,
    hydrodynamics_step_fn,
) -> Float[Array, "n_conserved ..."]:
    """Perform one time step using Strang operator splitting.

    Strang splitting (2nd order accurate):
    1. Chemistry for dt/2
    2. Hydrodynamics for dt
    3. Chemistry for dt/2

    Args:
        U: Conserved state vector [n_conserved, ...]
        dt: Time step [s]
        delta_x: Spatial grid spacing [m]
        species_list: List of species data
        config: Model configuration
        hydrodynamics_step_fn: Function to perform hydrodynamics step

    Returns:
        U_new: Updated state vector [n_conserved, ...]
    """
    # Step 1: Half-step chemistry
    U_half = apply_chemistry_source(U, dt / 2.0, species_list, config)

    # Step 2: Full-step hydrodynamics
    U_hydro = hydrodynamics_step_fn(U_half, dt, delta_x)

    # Step 3: Half-step chemistry
    U_new = apply_chemistry_source(U_hydro, dt / 2.0, species_list, config)

    return U_new


def godunov_splitting_step(
    U: Float[Array, "n_conserved ..."],
    dt: float,
    delta_x: float,
    species_list: list[SpeciesData],
    config: TwoTemperatureModelConfig,
    hydrodynamics_step_fn,
) -> Float[Array, "n_conserved ..."]:
    """Perform one time step using Godunov operator splitting.

    Godunov splitting (1st order accurate):
    1. Hydrodynamics for dt
    2. Chemistry for dt

    Args:
        U: Conserved state vector [n_conserved, ...]
        dt: Time step [s]
        delta_x: Spatial grid spacing [m]
        species_list: List of species data
        config: Model configuration
        hydrodynamics_step_fn: Function to perform hydrodynamics step

    Returns:
        U_new: Updated state vector [n_conserved, ...]
    """
    # Step 1: Hydrodynamics
    U_hydro = hydrodynamics_step_fn(U, dt, delta_x)

    # Step 2: Chemistry
    U_new = apply_chemistry_source(U_hydro, dt, species_list, config)

    return U_new
