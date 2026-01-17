import json
from pathlib import Path
from typing import Sequence
import jax.numpy as jnp
from jaxtyping import Float, Array


from compressible_1d.chemistry_types import SpeciesTable, ReactionTable
from compressible_1d import (
    constants,
    energy_models,
    reaction_rates,
    chemistry_types,
)


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


def load_equilibrium_enthalpy_curve_fits(json_path: str, species_name: str) -> tuple[
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

    theta_vib = jnp.full((len(names_tuple),), jnp.nan)
    if energy_model_config.model.lower() == "bird" and energy_model_config.data_path:
        theta_vib = energy_models.load_bird_characteristic_temperatures(
            energy_model_config.data_path, names_tuple
        )

    return SpeciesTable(
        names=names_tuple,
        molar_masses=molar_masses,
        h_s0=h_s0_array,
        dissociation_energy=dissociation_energy,
        ionization_energy=ionization_energy,
        vibrational_relaxation_factor=vibrational_relaxation_factor,
        theta_vib=theta_vib,
        charge=charge,
        sigma_es_a=sigma_es_a,
        sigma_es_b=sigma_es_b,
        sigma_es_c=sigma_es_c,
        is_monoatomic=is_monoatomic,
        T_ref=T_ref,
        energy_model=energy_model,
    )


def check_reaction_coverage(
    json_path: str,
    species_names: Sequence[str],
) -> tuple[list[dict], list[dict]]:
    """Check which reactions are covered by the given species list.

    Use this function before loading reactions to understand which reactions
    will be included and which will be skipped due to missing species.

    Args:
        json_path: Path to JSON file with reaction data.
        species_names: List of species names to check against.

    Returns:
        A tuple of (included_reactions, excluded_reactions) where each is a list
        of dicts containing:
            - "index": Original reaction index in the JSON file
            - "equation": Human-readable reaction equation
            - "missing_species": Set of species not in species_names (only for excluded)
    """
    json_path = Path(json_path)
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    reactions = data.get("reactions", data)
    if not isinstance(reactions, list):
        raise ValueError("JSON must contain a 'reactions' array or be an array.")

    species_set = set(species_names)
    included = []
    excluded = []

    for r, rxn in enumerate(reactions):
        # Collect all species involved in this reaction
        reaction_species = set(rxn["reactants"].keys()) | set(rxn["products"].keys())
        missing = reaction_species - species_set

        info = {
            "index": r,
            "equation": rxn.get("equation", f"Reaction {r}"),
        }

        if missing:
            info["missing_species"] = missing
            excluded.append(info)
        else:
            included.append(info)

    return included, excluded


def load_reactions_from_json(
    json_path: str,
    species_table: chemistry_types.SpeciesTable,
    chemistry_model_config: reaction_rates.ChemistryModelConfig = reaction_rates.ChemistryModelConfig(),
    preferential_factor: float = 1.0,
) -> ReactionTable:
    """Load reaction mechanism from JSON file.

    Reactions involving species not in species_table are silently skipped.
    Use check_reaction_coverage() beforehand to see which reactions will be
    included or excluded.

    The JSON file should contain an array of reaction objects with:
        - equation: Human-readable reaction equation
        - reactants: Dict mapping species name to stoichiometric coefficient
        - products: Dict mapping species name to stoichiometric coefficient
        - is_dissociation: Whether reaction uses T_d = sqrt(T*T_v) for rate
        - is_electron_impact: Whether reaction uses T_v for rate
        - C_f: Pre-exponential factor [cgs units]
        - n_f: Temperature exponent
        - E_f_over_k: Activation energy / k [K]
        - equilibrium_coeffs: [B1, B2, B3, B4, B5] for K_c polynomial

    Args:
        json_path: Path to JSON file with reaction data.
        species_table: SpeciesTable containing species information.
        preferential_factor: Factor for vibrational reactive source (default 1.0).

    Returns:
        ReactionTable with reactions that can be represented by the given species.
        Reactions with missing species (reactants or products) are excluded.
    """
    json_path = Path(json_path)
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    reactions = data.get("reactions", data)
    if not isinstance(reactions, list):
        raise ValueError("JSON must contain a 'reactions' array or be an array.")

    species_names = list(species_table.names)
    n_species = len(species_names)
    species_index = {name: i for i, name in enumerate(species_names)}
    species_set = set(species_names)

    # First pass: filter reactions to only those with all species present
    # (both reactants AND products must be in species_names)
    valid_reactions = []
    for rxn in reactions:
        reaction_species = set(rxn["reactants"].keys()) | set(rxn["products"].keys())
        if reaction_species.issubset(species_set):
            valid_reactions.append(rxn)

    n_reactions = len(valid_reactions)

    # Initialize arrays for valid reactions only
    reactant_stoich = jnp.zeros((n_reactions, n_species))
    product_stoich = jnp.zeros((n_reactions, n_species))
    C_f = jnp.zeros(n_reactions)
    n_f = jnp.zeros(n_reactions)
    E_f_over_k = jnp.zeros(n_reactions)
    equilibrium_coeffs = jnp.zeros((n_reactions, 5))
    is_dissociation = jnp.zeros(n_reactions)
    is_electron_impact = jnp.zeros(n_reactions)

    for r, rxn in enumerate(valid_reactions):
        # Parse reactants (all guaranteed to exist in species_index)
        for species_name, coeff in rxn["reactants"].items():
            s = species_index[species_name]
            reactant_stoich = reactant_stoich.at[r, s].set(coeff)

        # Parse products (all guaranteed to exist in species_index)
        for species_name, coeff in rxn["products"].items():
            s = species_index[species_name]
            product_stoich = product_stoich.at[r, s].set(coeff)

        # Arrhenius parameters
        C_f = C_f.at[r].set(rxn["C_f"])
        n_f = n_f.at[r].set(rxn["n_f"])
        E_f_over_k = E_f_over_k.at[r].set(rxn["E_f_over_k"])

        # Equilibrium coefficients
        coeffs = rxn["equilibrium_coeffs"]
        for i, coeff in enumerate(coeffs):
            equilibrium_coeffs = equilibrium_coeffs.at[r, i].set(coeff)

        # Reaction type flags
        is_dissociation = is_dissociation.at[r].set(
            1.0 if rxn.get("is_dissociation", False) else 0.0
        )
        is_electron_impact = is_electron_impact.at[r].set(
            1.0 if rxn.get("is_electron_impact", False) else 0.0
        )

    chemistry_model = reaction_rates.build_chemistry_model_from_config(
        config=chemistry_model_config,
        species_table=species_table,  # not needed here
        net_stoich=product_stoich - reactant_stoich,
        is_dissociation=is_dissociation,
    )

    return ReactionTable(
        species_names=tuple(species_names),
        reactant_stoich=reactant_stoich,
        product_stoich=product_stoich,
        C_f=C_f,
        n_f=n_f,
        E_f_over_k=E_f_over_k,
        equilibrium_coeffs=equilibrium_coeffs,
        is_dissociation=is_dissociation,
        is_electron_impact=is_electron_impact,
        preferential_factor=preferential_factor,
        chemistry_model=chemistry_model,
    )


def slice_reactions(
    reaction_table: ReactionTable,
    indices: Sequence[int],
) -> ReactionTable:
    """Create a new ReactionTable containing only the selected reactions.

    Args:
        reaction_table: The original ReactionTable to slice.
        indices: List of reaction indices to include in the new table.
            Indices are 0-based and must be valid for the original table.

    Returns:
        A new ReactionTable containing only the selected reactions.
        All arrays are sliced along the reaction dimension.

    Raises:
        IndexError: If any index is out of bounds.

    Example:
        >>> # Keep only the first 3 reactions
        >>> sliced = slice_reactions(reaction_table, [0, 1, 2])
        >>> sliced.n_reactions
        3

        >>> # Keep specific reactions by index
        >>> sliced = slice_reactions(reaction_table, [0, 5, 10])
    """
    indices_array = jnp.array(indices)

    return ReactionTable(
        species_names=reaction_table.species_names,
        reactant_stoich=reaction_table.reactant_stoich[indices_array],
        product_stoich=reaction_table.product_stoich[indices_array],
        C_f=reaction_table.C_f[indices_array],
        n_f=reaction_table.n_f[indices_array],
        E_f_over_k=reaction_table.E_f_over_k[indices_array],
        equilibrium_coeffs=reaction_table.equilibrium_coeffs[indices_array],
        is_dissociation=reaction_table.is_dissociation[indices_array],
        is_electron_impact=reaction_table.is_electron_impact[indices_array],
        preferential_factor=reaction_table.preferential_factor,
        chemistry_model=reaction_table.chemistry_model,
    )
