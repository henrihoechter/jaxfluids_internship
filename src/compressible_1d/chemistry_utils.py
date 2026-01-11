import json
from pathlib import Path
from typing import Sequence
import jax.numpy as jnp
from jaxtyping import Float, Array


from compressible_1d.chemistry_types import SpeciesTable
from compressible_1d import constants, energy_models


def _load_molar_mass(entry: dict) -> float:
    """Convert molar mass from amu to kg/mol."""
    return entry["molar_mass"] / 1000.0  # amu -> kg/mol


def _load_h_s0(entry: dict) -> float:
    """Convert h_s0 from kcal/mol to J/kg."""
    kcal_per_mol = entry["h_s0"]
    J_per_mol = kcal_per_mol * 4184.0  # kcal/mol → J/mol
    M_s = _load_molar_mass(entry)  # kg/mol
    J_per_kg = J_per_mol / M_s
    return J_per_kg


def _load_dissociation_energy(entry: dict) -> float | None:
    """Convert dissociation energy from eV to J."""
    if entry.get("dissociation_energy") is not None:
        return entry["dissociation_energy"] * constants.e
    return None


def _load_ionization_energy(entry: dict) -> float | None:
    """Convert ionization energy from eV to J."""
    if entry.get("ionization_energy") is not None:
        return entry["ionization_energy"] * constants.e
    return None


def _select_species_entries(
    raw_data: list[dict], species_names: Sequence[str]
) -> list[dict]:
    entries = {entry["name"]: entry for entry in raw_data}
    missing = [name for name in species_names if name not in entries]
    if missing:
        raise ValueError(f"Species not found in data: {missing}")
    return [entries[name] for name in species_names]


def load_equilibrium_enthalpy_curve_fits(
    json_path: str, species_name: str
) -> tuple[
    Float[Array, " n_ranges"],
    Float[Array, " n_ranges"],
    Float[Array, "n_parameters n_ranges"],
]:
    json_path = Path(json_path)
    with json_path.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    T_limit_low_list = []
    T_limit_high_list = []
    parameters_list = []
    for entry in raw_data:
        if entry["name"] == species_name:
            T_limit_low_list.append(entry["T_limit_low"])
            T_limit_high_list.append(entry["T_limit_high"])
            parameters_list.append(entry["parameters"])

    T_limit_low, T_limit_high, parameters = (
        jnp.array(T_limit_low_list),
        jnp.array(T_limit_high_list),
        jnp.array(parameters_list),
    )

    # verify data consistency
    if (
        not T_limit_high.shape[0] == T_limit_low.shape[0]
        or not T_limit_high.shape[0] == parameters.shape[0]
    ):
        raise ValueError(
            f"Import of equilibrium enthaply curve fits for species {species_name} "
            "failed."
        )

    if T_limit_high.ndim == 0:
        raise ValueError(f"No curve fit data found for species {species_name}.")

    return T_limit_low, T_limit_high, parameters


def load_species_table(
    species_names: Sequence[str],
    general_data_path: str,
    energy_model_config: energy_models.EnergyModelConfig,
) -> SpeciesTable:
    """Load species data and return as SpeciesTable.

    Args:
        general_data_path: Path to JSON file with general species data
        equilibrium_enthalpy: Path to JSON file with enthalpy curve fits
        species_names: Names of species to load (defaults to all)
        energy_model_config: Energy model configuration (data_path selects model data)

    Returns:
        SpeciesTable with all data as vectorized arrays
    """
    general_data_path = Path(general_data_path)
    with general_data_path.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if species_names is None:
        species_names = [entry["name"] for entry in raw_data]
    if not species_names:
        raise ValueError("species_names cannot be empty.")

    selected_entries = _select_species_entries(raw_data, species_names)

    names = []
    molar_masses = []
    h_s0_array = []
    dissociation_energy = []
    ionization_energy = []
    vibrational_relaxation_factor = []
    charge = []
    sigma_es_a = []
    sigma_es_b = []
    sigma_es_c = []

    for entry in selected_entries:
        names.append(entry["name"])
        molar_masses.append(_load_molar_mass(entry))
        h_s0_array.append(_load_h_s0(entry))
        dissociation_energy_value = _load_dissociation_energy(entry)
        ionization_energy_value = _load_ionization_energy(entry)
        vibrational_relaxation_value = entry.get("vibrational_relaxation_factor")
        sigma_es_a_value = entry.get("sigma_es_a")
        sigma_es_b_value = entry.get("sigma_es_b")
        sigma_es_c_value = entry.get("sigma_es_c")

        dissociation_energy.append(
            dissociation_energy_value
            if dissociation_energy_value is not None
            else jnp.nan
        )
        ionization_energy.append(
            ionization_energy_value if ionization_energy_value is not None else jnp.nan
        )
        vibrational_relaxation_factor.append(
            vibrational_relaxation_value
            if vibrational_relaxation_value is not None
            else jnp.nan
        )
        charge.append(entry.get("charge", 0))
        sigma_es_a.append(sigma_es_a_value if sigma_es_a_value is not None else jnp.nan)
        sigma_es_b.append(sigma_es_b_value if sigma_es_b_value is not None else jnp.nan)
        sigma_es_c.append(sigma_es_c_value if sigma_es_c_value is not None else jnp.nan)

    names_tuple = tuple(names)
    molar_masses = jnp.array(molar_masses)
    h_s0_array = jnp.array(h_s0_array)
    dissociation_energy = jnp.array(dissociation_energy)
    ionization_energy = jnp.array(ionization_energy)
    vibrational_relaxation_factor = jnp.array(vibrational_relaxation_factor)
    charge = jnp.array(charge)
    sigma_es_a = jnp.array(sigma_es_a)
    sigma_es_b = jnp.array(sigma_es_b)
    sigma_es_c = jnp.array(sigma_es_c)

    is_monoatomic = ~jnp.isfinite(dissociation_energy)
    T_ref = 298.16

    energy_model = energy_models.build_energy_model_from_config(
        energy_model_config,
        species_names=names_tuple,
        T_ref=T_ref,
        is_monoatomic=is_monoatomic,
        molar_masses=molar_masses,
    )

    return SpeciesTable(
        names=names_tuple,
        molar_masses=molar_masses,
        h_s0=h_s0_array,
        dissociation_energy=dissociation_energy,
        ionization_energy=ionization_energy,
        vibrational_relaxation_factor=vibrational_relaxation_factor,
        charge=charge,
        sigma_es_a=sigma_es_a,
        sigma_es_b=sigma_es_b,
        sigma_es_c=sigma_es_c,
        is_monoatomic=is_monoatomic,
        T_ref=T_ref,
        energy_model=energy_model,
    )
