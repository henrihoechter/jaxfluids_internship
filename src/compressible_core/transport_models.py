"""Dispatch utilities for transport property models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jaxtyping import Array, Float

from compressible_core import transport_model_gnoffo as transport_gnoffo
from compressible_core import transport_model_casseau
from compressible_core import thermodynamic_relations
from compressible_core.transport_models_types import TransportModel
if TYPE_CHECKING:
    from compressible_1d.equation_manager_types import EquationManager
else:
    EquationManager = object


def _zeros_transport(
    n_cells: int, n_species: int
) -> tuple[
    Float[Array, " n_cells"],
    Float[Array, " n_cells"],
    Float[Array, " n_cells"],
    Float[Array, " n_cells"],
    Float[Array, "n_cells n_species"],
]:
    return (
        jnp.zeros(n_cells),
        jnp.zeros(n_cells),
        jnp.zeros(n_cells),
        jnp.zeros(n_cells),
        jnp.zeros((n_cells, n_species)),
    )


def _compute_molar_concentrations(
    Y_s: Float[Array, "n_cells n_species"],
    molar_masses: Float[Array, " n_species"],
) -> tuple[
    Float[Array, "n_cells n_species"], Float[Array, "n_cells n_species"]
]:
    Y_M = Y_s * molar_masses[None, :]
    c_s = Y_M / jnp.sum(Y_M, axis=1, keepdims=True)
    gamma_s = c_s / molar_masses[None, :]
    return c_s, gamma_s


def compute_transport_properties_gnoffo(
    T: Float[Array, " n_cells"],
    T_v: Float[Array, " n_cells"],
    p: Float[Array, " n_cells"],
    Y_s: Float[Array, "n_cells n_species"],
    rho: Float[Array, " n_cells"],
    *,
    species_table,
    collision_integrals,
    include_diffusion: bool,
) -> tuple[
    Float[Array, " n_cells"],
    Float[Array, " n_cells"],
    Float[Array, " n_cells"],
    Float[Array, " n_cells"],
    Float[Array, "n_cells n_species"],
]:
    """Compute transport properties using the Gnoffo model."""
    del rho
    n_cells = T.shape[0]
    n_species = species_table.n_species

    if collision_integrals is None:
        return _zeros_transport(n_cells, n_species)

    M_s = species_table.molar_masses
    _c_s, gamma_s = _compute_molar_concentrations(Y_s, M_s)

    pair_indices = transport_gnoffo.build_pair_index_matrix(
        species_table.names, collision_integrals
    )
    pi_omega_11 = transport_gnoffo.interpolate_collision_integral(
        T,
        collision_integrals.omega_11_2000K,
        collision_integrals.omega_11_4000K,
    )
    pi_omega_22 = transport_gnoffo.interpolate_collision_integral(
        T,
        collision_integrals.omega_22_2000K,
        collision_integrals.omega_22_4000K,
    )
    pi_omega_11_Tv = transport_gnoffo.interpolate_collision_integral(
        T_v,
        collision_integrals.omega_11_2000K,
        collision_integrals.omega_11_4000K,
    )

    delta_1 = transport_gnoffo.compute_modified_collision_integral_1(
        T, M_s, M_s, pi_omega_11, pair_indices
    )
    delta_2 = transport_gnoffo.compute_modified_collision_integral_2(
        T, M_s, M_s, pi_omega_22, pair_indices
    )
    delta_1_Tv = transport_gnoffo.compute_modified_collision_integral_1(
        T_v, M_s, M_s, pi_omega_11_Tv, pair_indices
    )

    mu = transport_gnoffo.compute_mixture_viscosity(T, gamma_s, M_s, delta_2)
    eta_t = transport_gnoffo.compute_translational_thermal_conductivity(
        T, gamma_s, M_s, delta_2
    )
    is_molecule = ~species_table.is_monoatomic.astype(bool)
    eta_r = transport_gnoffo.compute_rotational_thermal_conductivity(
        T, gamma_s, is_molecule, delta_1
    )
    eta_v = transport_gnoffo.compute_vibrational_thermal_conductivity(
        T_v, gamma_s, is_molecule, delta_1_Tv
    )

    if include_diffusion:
        D_sr = transport_gnoffo.compute_binary_diffusion_coefficient(T, p, delta_1)
        D_s = transport_gnoffo.compute_effective_diffusion_coefficient(
            gamma_s, M_s, D_sr
        )
    else:
        D_s = jnp.zeros((n_cells, n_species))

    return mu, eta_t, eta_r, eta_v, D_s


def compute_transport_properties_casseau(
    T: Float[Array, " n_cells"],
    T_v: Float[Array, " n_cells"],
    p: Float[Array, " n_cells"],
    Y_s: Float[Array, "n_cells n_species"],
    rho: Float[Array, " n_cells"],
    *,
    species_table,
    casseau_transport,
    collision_integrals,
    include_diffusion: bool,
) -> tuple[
    Float[Array, " n_cells"],
    Float[Array, " n_cells"],
    Float[Array, " n_cells"],
    Float[Array, " n_cells"],
    Float[Array, "n_cells n_species"],
]:
    """Compute transport properties using the Casseau model."""
    del rho
    n_cells = T.shape[0]
    n_species = species_table.n_species

    if casseau_transport is None:
        raise ValueError("Casseau transport selected but no data provided.")

    M_s = species_table.molar_masses
    _c_s, gamma_s = _compute_molar_concentrations(Y_s, M_s)

    # Normalize mole fractions in case clipping or numerical drift breaks sum=1.
    Y_sum = jnp.sum(Y_s, axis=1, keepdims=True)
    X_s = Y_s / jnp.clip(Y_sum, 1e-30, None)

    cv_ve = thermodynamic_relations.compute_cv_ve(T_v, species_table)
    mu, eta_t, eta_r, eta_v = transport_model_casseau.compute_casseau_transport_properties(
        T,
        T_v,
        X_s,
        M_s,
        species_table.is_monoatomic,
        cv_ve,
        casseau_transport,
    )

    if include_diffusion and collision_integrals is not None:
        pair_indices = transport_gnoffo.build_pair_index_matrix(
            species_table.names, collision_integrals
        )
        pi_omega_11 = transport_gnoffo.interpolate_collision_integral(
            T,
            collision_integrals.omega_11_2000K,
            collision_integrals.omega_11_4000K,
        )
        delta_1 = transport_gnoffo.compute_modified_collision_integral_1(
            T, M_s, M_s, pi_omega_11, pair_indices
        )
        D_sr = transport_gnoffo.compute_binary_diffusion_coefficient(T, p, delta_1)
        D_s = transport_gnoffo.compute_effective_diffusion_coefficient(
            gamma_s, M_s, D_sr
        )
    else:
        D_s = jnp.zeros((n_cells, n_species))

    return mu, eta_t, eta_r, eta_v, D_s


def compute_transport_properties(
    T: Float[Array, " n_cells"],
    T_v: Float[Array, " n_cells"],
    p: Float[Array, " n_cells"],
    Y_s: Float[Array, "n_cells n_species"],
    rho: Float[Array, " n_cells"],
    equation_manager: EquationManager,
) -> tuple[
    Float[Array, " n_cells"],
    Float[Array, " n_cells"],
    Float[Array, " n_cells"],
    Float[Array, " n_cells"],
    Float[Array, "n_cells n_species"],
]:
    """Compute transport properties for the selected model.

    Returns:
        mu, eta_t, eta_r, eta_v, D_s
    """
    transport_model = equation_manager.transport_model
    if transport_model is None:
        return _zeros_transport(T.shape[0], equation_manager.species.n_species)

    return transport_model.compute_transport_properties(T, T_v, p, Y_s, rho)
