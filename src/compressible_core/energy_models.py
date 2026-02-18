from __future__ import annotations

import functools

import jax.numpy as jnp
from jaxtyping import Array, Float

from compressible_core import constants, thermodynamic_relations
from compressible_core.energy_models_types import EnergyFn, EnergyModel, EnergyModelConfig
from compressible_core.energy_models_utils import (
    build_energy_model_from_config,
    load_bird_characteristic_temperatures,
)


def _zeros_like_species(
    T_V: Float[Array, " N"], n_species: int
) -> Float[Array, "n_species N"]:
    T = jnp.atleast_1d(T_V)
    return jnp.zeros((n_species, T.shape[0]))


def build_gnoffo_energy_model(
    *,
    T_ref: float,
    T_limit_low: Float[Array, "n_species n_ranges"],
    T_limit_high: Float[Array, "n_species n_ranges"],
    enthalpy_coeffs: Float[Array, "n_species n_ranges n_coeffs"],
    is_monoatomic: Float[Array, " n_species"],
    molar_masses: Float[Array, " n_species"],
) -> EnergyModel:
    """Build the Gnoffo vibrational-electronic energy model."""
    e_ve = functools.partial(
        thermodynamic_relations.compute_e_vib_electronic,
        T_ref=T_ref,
        T_limit_low=T_limit_low,
        T_limit_high=T_limit_high,
        parameters=enthalpy_coeffs,
        is_monoatomic=is_monoatomic,
        molar_masses=molar_masses,
    )
    cv_ve = functools.partial(
        thermodynamic_relations.compute_cv_vib_electronic,
        T_limit_low=T_limit_low,
        T_limit_high=T_limit_high,
        parameters=enthalpy_coeffs,
        is_monoatomic=is_monoatomic,
        molar_masses=molar_masses,
    )
    cp = functools.partial(
        thermodynamic_relations.compute_cp_from_polynomial,
        T_limit_low=T_limit_low,
        T_limit_high=T_limit_high,
        parameters=enthalpy_coeffs,
        molar_masses=molar_masses,
    )

    n_species = molar_masses.shape[0]

    def e_vib(T_V: Float[Array, " N"]) -> Float[Array, "n_species N"]:
        return e_ve(T_V)

    def e_el(T_V: Float[Array, " N"]) -> Float[Array, "n_species N"]:
        return _zeros_like_species(T_V, n_species)

    return EnergyModel(e_vib=e_vib, e_el=e_el, e_ve=e_ve, cv_ve=cv_ve, cp=cp)


def build_bird_energy_model(
    *,
    characteristic_temperature: Float[Array, " n_species"],
    g_i: Float[Array, "n_species n_levels"],
    theta_el_i: Float[Array, "n_species n_levels"],
    molar_masses: Float[Array, " n_species"],
    is_monoatomic: Float[Array, " n_species"],
    include_electronic: bool = True,
) -> EnergyModel:
    """Build the Bird vibrational + electronic energy model."""
    e_vib = functools.partial(
        thermodynamic_relations.compute_e_vibrational_from_harmonic_oscillator,
        characteristic_temperature=characteristic_temperature,
        M=molar_masses,
    )
    e_el = functools.partial(
        thermodynamic_relations.compute_electronic_energy_from_levels_batched,
        g_i=g_i,
        theta_el_i=theta_el_i,
        molar_masses=molar_masses,
    )

    cv_vib = functools.partial(
        thermodynamic_relations.compute_cv_vibrational_from_harmonic_oscillator,
        characteristic_temperature=characteristic_temperature,
        M=molar_masses,
    )
    cv_el = functools.partial(
        thermodynamic_relations.compute_cv_electronic_from_levels_batched,
        g_i=g_i,
        theta_el_i=theta_el_i,
        molar_masses=molar_masses,
    )

    def e_ve(T_V: Float[Array, " N"]) -> Float[Array, "n_species N"]:
        if include_electronic:
            return e_vib(T_V) + e_el(T_V)
        else:
            return e_vib(T_V)

    def cv_ve(T_V: Float[Array, " N"]) -> Float[Array, "n_species N"]:
        if include_electronic:
            return cv_vib(T_V) + cv_el(T_V)
        else:
            return cv_vib(T_V)

    def cp(T: Float[Array, " N"]) -> Float[Array, "n_species N"]:
        if include_electronic:
            return (
                thermodynamic_relations.compute_cv_trans_rot(
                    T=T, is_monoatomic=is_monoatomic, molar_masses=molar_masses
                )
                + cv_vib(T)
                + cv_el(T)
                + constants.R_universal / molar_masses[:, None]
            )
        else:
            return (
                thermodynamic_relations.compute_cv_trans_rot(
                    T=T, is_monoatomic=is_monoatomic, molar_masses=molar_masses
                )
                + cv_vib(T)
                + constants.R_universal / molar_masses[:, None]
            )

    return EnergyModel(e_vib=e_vib, e_el=e_el, e_ve=e_ve, cv_ve=cv_ve, cp=cp)
