from __future__ import annotations

from dataclasses import dataclass
import functools
import json
from pathlib import Path
from typing import Callable, Literal, Sequence

import jax.numpy as jnp
from jaxtyping import Array, Float

from compressible_core import thermodynamic_relations, constants


EnergyFn = Callable[[Float[Array, " N"]], Float[Array, "n_species N"]]


@dataclass(frozen=True, eq=False)
class EnergyModel:
    """Container for vibrational/electronic energy model callables."""

    e_vib: EnergyFn
    e_el: EnergyFn
    e_ve: EnergyFn
    cv_ve: EnergyFn
    cp: EnergyFn


@dataclass(frozen=True)
class EnergyModelConfig:
    """Configuration for selecting and building energy models."""

    model: Literal["gnoffo", "bird"] = "gnoffo"
    include_electronic: bool = True
    data_path: str | None = None


def _pad_levels(values: Sequence[float], max_levels: int) -> jnp.ndarray:
    values_array = jnp.array(values)
    pad_width = max_levels - values_array.shape[0]
    if pad_width < 0:
        raise ValueError("max_levels must be >= len(values).")
    return jnp.pad(values_array, (0, pad_width))


def _load_gnoffo_energy_data(
    data_path: str, species_names: Sequence[str]
) -> tuple[
    Float[Array, "n_species n_ranges"],
    Float[Array, "n_species n_ranges"],
    Float[Array, "n_species n_ranges n_coeffs"],
]:
    raw_data = json.loads(Path(data_path).read_text(encoding="utf-8"))
    by_name: dict[str, list[dict]] = {}
    for entry in raw_data:
        by_name.setdefault(entry["name"], []).append(entry)

    T_limit_low_list = []
    T_limit_high_list = []
    parameters_list = []
    n_ranges = None

    for name in species_names:
        entries = by_name.get(name)
        if not entries:
            raise ValueError(f"Gnoffo data missing for species '{name}'.")
        T_low = [entry["T_limit_low"] for entry in entries]
        T_high = [entry["T_limit_high"] for entry in entries]
        params = [entry["parameters"] for entry in entries]

        if n_ranges is None:
            n_ranges = len(T_low)
        elif len(T_low) != n_ranges:
            raise ValueError("All species must have the same number of ranges.")

        T_limit_low_list.append(T_low)
        T_limit_high_list.append(T_high)
        parameters_list.append(params)

    return (
        jnp.array(T_limit_low_list),
        jnp.array(T_limit_high_list),
        jnp.array(parameters_list),
    )


def _load_bird_energy_data(
    data_path: str,
    species_names: Sequence[str],
    include_electronic: bool,
) -> tuple[
    Float[Array, " n_species"],
    Float[Array, "n_species n_levels"],
    Float[Array, "n_species n_levels"],
]:
    raw_data = json.loads(Path(data_path).read_text(encoding="utf-8"))
    by_name = {entry["name"]: entry for entry in raw_data}

    theta_vib_list = []
    g_i_list = []
    theta_el_list = []

    for name in species_names:
        entry = by_name.get(name)
        if entry is None:
            raise ValueError(f"Bird data missing for species '{name}'.")

        theta_vib_list.append(entry.get("theta_vib", 0.0))
        g_i = entry.get("g_i", [])
        theta_el = entry.get("theta_el_i", [])

        if include_electronic and (not g_i or not theta_el):
            raise ValueError(
                f"Electronic levels required for species '{name}' with include_electronic=True."
            )
        if include_electronic and len(g_i) != len(theta_el):
            raise ValueError(f"Electronic level lengths mismatch for species '{name}'.")

        if not g_i or not theta_el:
            g_i = [1.0]
            theta_el = [0.0]

        g_i_list.append(g_i)
        theta_el_list.append(theta_el)

    max_levels = max(len(values) for values in g_i_list)
    g_i = jnp.stack([_pad_levels(values, max_levels) for values in g_i_list])
    theta_el_i = jnp.stack(
        [_pad_levels(values, max_levels) for values in theta_el_list]
    )
    theta_vib = jnp.array(theta_vib_list)

    return theta_vib, g_i, theta_el_i


def load_bird_characteristic_temperatures(
    data_path: str, species_names: Sequence[str]
) -> Float[Array, " n_species"]:
    """Load Bird characteristic vibrational temperatures for the given species."""
    theta_vib, _, _ = _load_bird_energy_data(
        data_path, species_names, include_electronic=False
    )
    return theta_vib


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


def build_energy_model_from_config(
    config: EnergyModelConfig | None,
    *,
    species_names: Sequence[str],
    T_ref: float,
    is_monoatomic: Float[Array, " n_species"],
    molar_masses: Float[Array, " n_species"],
) -> EnergyModel:
    """Build an energy model from a configuration object."""
    if config is None:
        config = EnergyModelConfig()

    model = config.model.lower()
    if model == "gnoffo":
        if not config.include_electronic:
            raise ValueError(
                "Gnoffo energy model does not support include_electronic=False."
            )
        if not config.data_path:
            raise ValueError("Gnoffo energy model requires data_path.")

        T_limit_low, T_limit_high, enthalpy_coeffs = _load_gnoffo_energy_data(
            config.data_path, species_names
        )
        return build_gnoffo_energy_model(
            T_ref=T_ref,
            T_limit_low=T_limit_low,
            T_limit_high=T_limit_high,
            enthalpy_coeffs=enthalpy_coeffs,
            is_monoatomic=is_monoatomic,
            molar_masses=molar_masses,
        )

    if model == "bird":
        if not config.data_path:
            raise ValueError("Bird energy model requires data_path.")

        theta_vib, g_i, theta_el_i = _load_bird_energy_data(
            config.data_path,
            species_names,
            include_electronic=config.include_electronic,
        )
        return build_bird_energy_model(
            characteristic_temperature=theta_vib,
            g_i=g_i,
            theta_el_i=theta_el_i,
            molar_masses=molar_masses,
            is_monoatomic=is_monoatomic,
            include_electronic=config.include_electronic,
        )

    raise ValueError(f"Unknown energy model '{config.model}'.")
