import json
from pathlib import Path
from typing import List
import jax.numpy as jnp
from jaxtyping import Float, Array


from compressible_1d.chemistry_types import Species, SpeciesTable
from compressible_1d import constants


def load_species_from_gnoffo(
    general_data_path: str, equilibrium_enthalpy: str
) -> List[Species]:
    """Load species data including characteristic temperatures.

    Extended to load:
    - Enthalpy polynomial fits (existing)
    - Characteristic temperatures theta_v, theta_rot (new - from general_data JSON)

    Format description `general_data_path`
    {
        "name": string,
        "charge": float  # charge in elementary charge units
        "molar_mass": float,  # relative atomic mass in amu
        "h_s0": float,  # standard enthalpy of formation at 0 K in J/g/mol
        "ionization_energy": float | None,  # ionization energy in eV, None for
            ionized species
        "dissociation_energy": float | None,  # dissociation energy in eV, None for
            monoatomic species
        "vibrational_relaxation_factor": float | None  # vibrational relaxation factor,
            None for monoatomic species
        "theta_v": float  # vibrational characteristic temperature [K], 0.0 for atoms
        "theta_rot": float  # rotational characteristic temperature [K], 0.0 for atoms
    }

    Format description `equilibrium_enthaply`

    Note:
        - NO separate C_p curve fits needed!
        - C_p computed via dh/dT from enthalpy polynomial
    """

    def _load_molar_mass(entry: dict) -> float:
        """Convert molar mass from amu to kg/kmol.

        Note: 1 amu = 1 g/mol = 1 kg/kmol (numerically identical)
        """
        return entry["molar_mass"]  # amu -> kg/kmol (no conversion needed)

    def _load_h_s0(entry: dict) -> float:
        """Convert h_s0 from kcal/mol to J/kg.

        The JSON contains h_s0 in kcal/g-mole (kcal/mol).
        Conversion: kcal/mol → J/kg
          1. kcal → J: multiply by 4184 J/kcal
          2. /mol → /kg: divide by molar_mass [kg/mol]
        """
        kcal_per_mol = entry["h_s0"]
        J_per_mol = kcal_per_mol * 4184.0  # kcal/mol → J/mol
        M_kg_per_mol = (
            entry["molar_mass"] / 1000.0
        )  # g/mol → kg/mol (molar_mass is in amu = g/mol)
        J_per_kg = J_per_mol / M_kg_per_mol
        return J_per_kg

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
            json_path=equilibrium_enthalpy, species_name=entry["name"]
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


def load_species_table_from_gnoffo(
    general_data_path: str, equilibrium_enthalpy: str
) -> SpeciesTable:
    """Load species data and return as SpeciesTable.

    Convenience wrapper around load_species_from_gnoffo that returns
    a vectorized SpeciesTable for efficient JAX operations.

    Args:
        general_data_path: Path to JSON file with general species data
        equilibrium_enthalpy: Path to JSON file with enthalpy curve fits

    Returns:
        SpeciesTable with all data as vectorized arrays
    """
    species_list = load_species_from_gnoffo(general_data_path, equilibrium_enthalpy)
    return SpeciesTable.from_species_list(species_list)
