"""Dispatch utilities for transport property models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jaxtyping import Array, Float

from compressible_core import transport
from compressible_core import transport_casseau
from compressible_core import thermodynamic_relations
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

    species_table = equation_manager.species
    M_s = species_table.molar_masses
    n_cells = T.shape[0]
    n_species = species_table.n_species

    Y_M = Y_s * M_s[None, :]
    c_s = Y_M / jnp.sum(Y_M, axis=1, keepdims=True)
    gamma_s = c_s / M_s[None, :]

    if transport_model.model == "gnoffo":
        if equation_manager.collision_integrals is None:
            return _zeros_transport(n_cells, n_species)

        collision_integrals = equation_manager.collision_integrals
        pair_indices = transport.build_pair_index_matrix(
            species_table.names, collision_integrals
        )
        pi_omega_11 = transport.interpolate_collision_integral(
            T,
            collision_integrals.omega_11_2000K,
            collision_integrals.omega_11_4000K,
        )
        pi_omega_22 = transport.interpolate_collision_integral(
            T,
            collision_integrals.omega_22_2000K,
            collision_integrals.omega_22_4000K,
        )
        pi_omega_11_Tv = transport.interpolate_collision_integral(
            T_v,
            collision_integrals.omega_11_2000K,
            collision_integrals.omega_11_4000K,
        )

        delta_1 = transport.compute_modified_collision_integral_1(
            T, M_s, M_s, pi_omega_11, pair_indices
        )
        delta_2 = transport.compute_modified_collision_integral_2(
            T, M_s, M_s, pi_omega_22, pair_indices
        )
        delta_1_Tv = transport.compute_modified_collision_integral_1(
            T_v, M_s, M_s, pi_omega_11_Tv, pair_indices
        )

        mu = transport.compute_mixture_viscosity(T, gamma_s, M_s, delta_2)
        eta_t = transport.compute_translational_thermal_conductivity(
            T, gamma_s, M_s, delta_2
        )
        is_molecule = ~species_table.is_monoatomic.astype(bool)
        eta_r = transport.compute_rotational_thermal_conductivity(
            T, gamma_s, is_molecule, delta_1
        )
        eta_v = transport.compute_vibrational_thermal_conductivity(
            T_v, gamma_s, is_molecule, delta_1_Tv
        )

        if transport_model.include_diffusion:
            D_sr = transport.compute_binary_diffusion_coefficient(T, p, delta_1)
            D_s = transport.compute_effective_diffusion_coefficient(
                gamma_s, M_s, D_sr
            )
        else:
            D_s = jnp.zeros((n_cells, n_species))

        return mu, eta_t, eta_r, eta_v, D_s

    if transport_model.model == "casseau":
        if equation_manager.casseau_transport is None:
            raise ValueError("Casseau transport selected but no data provided.")

        cv_ve = thermodynamic_relations.compute_cv_ve(T_v, species_table)
        mu, eta_t, eta_r, eta_v = transport_casseau.compute_casseau_transport_properties(
            T,
            T_v,
            Y_s,
            M_s,
            species_table.is_monoatomic,
            cv_ve,
            equation_manager.casseau_transport,
        )

        if transport_model.include_diffusion and equation_manager.collision_integrals is not None:
            collision_integrals = equation_manager.collision_integrals
            pair_indices = transport.build_pair_index_matrix(
                species_table.names, collision_integrals
            )
            pi_omega_11 = transport.interpolate_collision_integral(
                T,
                collision_integrals.omega_11_2000K,
                collision_integrals.omega_11_4000K,
            )
            delta_1 = transport.compute_modified_collision_integral_1(
                T, M_s, M_s, pi_omega_11, pair_indices
            )
            D_sr = transport.compute_binary_diffusion_coefficient(T, p, delta_1)
            D_s = transport.compute_effective_diffusion_coefficient(
                gamma_s, M_s, D_sr
            )
        else:
            D_s = jnp.zeros((n_cells, n_species))

        return mu, eta_t, eta_r, eta_v, D_s

    raise ValueError(f"Unknown transport model: {transport_model.model}")
