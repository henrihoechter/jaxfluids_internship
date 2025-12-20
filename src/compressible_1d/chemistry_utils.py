import json
from pathlib import Path
from typing import List
import jax.numpy as jnp
from jaxtyping import Float, Array


from compressible_1d.chemistry_types import Species
from compressible_1d import constants


def load_species_from_gnoffo(
    general_data_path: str, equilibrium_enthaply: str
) -> List[Species]:
    """

    Format description `general_data_path`
    {
        "name": string,
        "charge": float  # charge inelementary charge units
        "molar_mass": float,  # relative atmoic mass in amu
        "h_s0": float,  # standard enthalpy of formation at 0 K in J/g/mol
        "ionization_energy": float | None,  # ionization energy in eV, None for
            ionized species
        "dissociation_energy": float | None,  # dissociation energy in eV, None for
            monoatomic species
        "vibrational_relaxation_factor": float | None  # vibrational relaxation factor,
            None for monoatomic species
    }

    Format description `equilibrium_enthaply`

    """

    def _load_molar_mass(entry: dict) -> float:
        """Convert molar mass from amu to g."""
        return entry["molar_mass"] * 1e3  # g/mol -> kg/mol

    def _load_h_s0(entry: dict) -> float:
        """Convert h_s0 from J/g/mol to J/kg."""
        return entry["h_s0"] / (entry["molar_mass"] * 1e3)  # J/g/mol -> J/kg

    def _load_dissociation_energy(entry: dict) -> float | None:
        """Convert dissociation energy from eV to J."""
        if entry.get("dissociation_energy") is not None:
            return entry["dissociation_energy"] * constants.e
        else:
            return None

    def _load_ionization_energy(entry: dict) -> float | None:
        """Convert ionization energy from eV to J."""
        if entry.get("ionization_energy") is not None:
            return entry["ionization_energy"] * constants.e
        else:
            return None

    general_data_path = Path(general_data_path)

    with general_data_path.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    species_list: List[Species] = []

    for entry in raw_data:
        T_limit_low, T_limit_high, parameters = load_equilibrium_enthalpy_curve_fits(
            json_path=equilibrium_enthaply, species_name=entry["name"]
        )
        species = Species(
            name=entry["name"],
            molar_mass=_load_molar_mass(entry),
            h_s0=_load_h_s0(entry),
            T_limit_low=T_limit_low,
            T_limit_high=T_limit_high,
            parameters=parameters,
            dissociation_energy=_load_dissociation_energy(entry),
            ionization_energy=_load_ionization_energy(entry),
            vibrational_relaxation_factor=entry.get("vibrational_relaxation_factor"),
        )
        species_list.append(species)

    return species_list


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
