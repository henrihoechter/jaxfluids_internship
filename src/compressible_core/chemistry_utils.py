import json
from pathlib import Path
from typing import Sequence
import jax.numpy as jnp
from jaxtyping import Float, Array, Int


from compressible_core.chemistry_types import SpeciesTable, ReactionTable
from compressible_core import (
    constants,
    energy_models_types,
    energy_models_utils,
    chemistry,
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
    """Load dissociation energy in J/kg."""
    if entry.get("dissociation_energy") is not None:
        return float(entry["dissociation_energy"])
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


def _extract_vibrational_relaxation_scalar(
    species_name: str, raw_value: object
) -> float | None:
    """Return a per-species scalar for backward compatibility."""
    if isinstance(raw_value, dict):
        self_pair = raw_value.get(species_name, None)
        if isinstance(self_pair, dict):
            a_value = self_pair.get("a")
            return float(a_value) if a_value is not None else None
        return None
    if raw_value is None:
        return None
    return float(raw_value)


def _build_vibrational_relaxation_tables(
    names: Sequence[str],
    charge: Float[Array, " n_species"],
    is_monoatomic: Float[Array, " n_species"],
    vibrational_relaxation_raw: Sequence[object],
) -> tuple[
    Float[Array, " n_molecules n_partners"],
    Float[Array, " n_molecules n_partners"],
    Int[Array, " n_molecules"],
    Int[Array, " n_partners"],
]:
    non_e_indices = [i for i, c in enumerate(charge.tolist()) if c != -1]
    molecule_indices = [i for i, mono in enumerate(is_monoatomic.tolist()) if not mono]
    non_e_names = [names[i] for i in non_e_indices]

    a_rows: list[list[float]] = []
    b_rows: list[list[float]] = []
    missing: dict[str, list[str]] = {}

    for m_idx in molecule_indices:
        m_name = names[m_idx]
        raw_value = vibrational_relaxation_raw[m_idx]
        if isinstance(raw_value, dict):
            row_a: list[float] = []
            row_b: list[float] = []
            missing_partners: list[str] = []
            for s_name in non_e_names:
                pair_entry = raw_value.get(s_name, None)
                if (
                    not isinstance(pair_entry, dict)
                    or pair_entry.get("a") is None
                    or pair_entry.get("b") is None
                ):
                    missing_partners.append(s_name)
                else:
                    row_a.append(float(pair_entry["a"]))
                    row_b.append(float(pair_entry["b"]))
            if missing_partners:
                missing[m_name] = missing_partners
            else:
                a_rows.append(row_a)
                b_rows.append(row_b)
        elif raw_value is None:
            missing[m_name] = non_e_names
        else:
            a_rows.append([float(raw_value)] * len(non_e_names))
            b_rows.append([jnp.nan] * len(non_e_names))

    if missing:
        details = "; ".join(
            f"{name}: {partners}" for name, partners in missing.items()
        )
        raise ValueError(
            "Missing vibrational relaxation factors for collision partners: "
            f"{details}"
        )

    if non_e_names:
        a_ms = jnp.array(a_rows) if a_rows else jnp.zeros((0, len(non_e_names)))
        b_ms = jnp.array(b_rows) if b_rows else jnp.zeros((0, len(non_e_names)))
    else:
        a_ms = jnp.zeros((len(molecule_indices), 0))
        b_ms = jnp.zeros((len(molecule_indices), 0))

    return (
        a_ms,
        b_ms,
        jnp.array(molecule_indices, dtype=int),
        jnp.array(non_e_indices, dtype=int),
    )


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
    energy_model_config: energy_models_types.EnergyModelConfig,
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
    vibrational_relaxation_raw = []
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
        vibrational_relaxation_raw.append(vibrational_relaxation_value)
        vibrational_relaxation_scalar = _extract_vibrational_relaxation_scalar(
            entry["name"], vibrational_relaxation_value
        )
        vibrational_relaxation_factor.append(
            vibrational_relaxation_scalar
            if vibrational_relaxation_scalar is not None
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

    (
        vibrational_relaxation_a_ms,
        vibrational_relaxation_b_ms,
        vibrational_relaxation_molecule_indices,
        vibrational_relaxation_partner_indices,
    ) = _build_vibrational_relaxation_tables(
        names_tuple, charge, is_monoatomic, vibrational_relaxation_raw
    )

    energy_model = energy_models_utils.build_energy_model_from_config(
        energy_model_config,
        species_names=names_tuple,
        T_ref=T_ref,
        is_monoatomic=is_monoatomic,
        molar_masses=molar_masses,
    )

    theta_vib = jnp.full((len(names_tuple),), jnp.nan)
    if energy_model_config.model.lower() == "bird" and energy_model_config.data_path:
        theta_vib = energy_models_utils.load_bird_characteristic_temperatures(
            energy_model_config.data_path, names_tuple
        )

    return SpeciesTable(
        names=names_tuple,
        molar_masses=molar_masses,
        h_s0=h_s0_array,
        dissociation_energy=dissociation_energy,
        ionization_energy=ionization_energy,
        vibrational_relaxation_factor=vibrational_relaxation_factor,
        vibrational_relaxation_a_ms=vibrational_relaxation_a_ms,
        vibrational_relaxation_b_ms=vibrational_relaxation_b_ms,
        vibrational_relaxation_molecule_indices=vibrational_relaxation_molecule_indices,
        vibrational_relaxation_partner_indices=vibrational_relaxation_partner_indices,
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
    chemistry_model_config: chemistry.ChemistryModelConfig = chemistry.ChemistryModelConfig(),
) -> ReactionTable:
    """Load reaction mechanism from JSON file.

    Reactions involving species not in species_table are silently skipped.
    Use check_reaction_coverage() beforehand to see which reactions will be
    included or excluded.

    The JSON file should contain an array of reaction objects with:
        - equation: Human-readable reaction equation
        - reactants: Dict mapping species name to stoichiometric coefficient
        - products: Dict mapping species name to stoichiometric coefficient
        - is_dissociation: Whether reaction uses CVDV-QP formulation
        - is_electron_impact: Whether reaction uses T_v for rate control
        - C_f: Pre-exponential factor [m^3/molecule/s or m^6/molecule^2/s]
        - n_f: Temperature exponent [-]
        - E_f_over_k: Activation energy / k [K]
        - equilibrium_coeffs_casseau: Casseau coefficients table with rows
            [n_ref_cm3 or n_ref_m3, A0..A4] for Eq. 2.69. Preferred as a
            per-reaction field. A top-level equilibrium_coeffs_casseau is
            still accepted for backward compatibility.

    Notes:
        - C_f is converted from per-molecule to per-mol units on load.
        - n_ref_cm3 is converted to n_ref_m3 (×1e6).

    Args:
        json_path: Path to JSON file with reaction data.
        species_table: SpeciesTable containing species information.

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
    is_dissociation = jnp.zeros(n_reactions)
    is_electron_impact = jnp.zeros(n_reactions)

    coeffs_data = data.get("equilibrium_coeffs_casseau")

    def _parse_coeff_table(coeff_rows: list[dict]) -> jnp.ndarray:
        rows = []
        for row in coeff_rows:
            if "n_ref_m3" in row:
                n_ref = float(row["n_ref_m3"])
            elif "n_ref_cm3" in row:
                # User-requested SI scaling: cm^3 -> m^3
                n_ref = float(row["n_ref_cm3"]) * 1e6
            else:
                raise ValueError("Casseau coeff row missing n_ref_cm3/n_ref_m3.")
            rows.append(
                [
                    n_ref,
                    float(row["A0"]),
                    float(row["A1"]),
                    float(row["A2"]),
                    float(row["A3"]),
                    float(row["A4"]),
                ]
            )
        rows.sort(key=lambda r: r[0])
        return jnp.array(rows)

    # Preferred format: coefficients embedded per reaction.
    per_reaction_coeffs = [
        rxn.get("equilibrium_coeffs_casseau") for rxn in valid_reactions
    ]
    if any(rows is not None for rows in per_reaction_coeffs):
        if not all(rows is not None for rows in per_reaction_coeffs):
            raise ValueError(
                "All reactions must define equilibrium_coeffs_casseau if any do."
            )
        coeff_lists = per_reaction_coeffs
    else:
        if coeffs_data is None:
            raise ValueError("Missing equilibrium_coeffs_casseau in reaction JSON.")
        if isinstance(coeffs_data, list):
            if coeffs_data and isinstance(coeffs_data[0], dict):
                coeff_lists = [coeffs_data] * n_reactions
            else:
                if len(coeffs_data) != n_reactions:
                    raise ValueError(
                        "equilibrium_coeffs_casseau list length must match reactions."
                    )
                coeff_lists = coeffs_data
        elif isinstance(coeffs_data, dict):
            if len(coeffs_data) == 1:
                coeff_lists = [next(iter(coeffs_data.values()))] * n_reactions
            else:
                coeff_lists = []
                for rxn in valid_reactions:
                    key = rxn.get("equilibrium_key", rxn.get("equation", ""))
                    if key in coeffs_data:
                        coeff_lists.append(coeffs_data[key])
                        continue
                    key_norm = key.replace(" ", "")
                    match = None
                    for k, v in coeffs_data.items():
                        if k.replace(" ", "") == key_norm:
                            match = v
                            break
                    if match is None:
                        raise ValueError(f"No Casseau coeffs for reaction '{key}'.")
                    coeff_lists.append(match)
        else:
            raise ValueError("equilibrium_coeffs_casseau must be a list or dict.")

    parsed_tables = [_parse_coeff_table(rows) for rows in coeff_lists]
    n_refs = parsed_tables[0].shape[0] if parsed_tables else 0
    if any(table.shape[0] != n_refs for table in parsed_tables):
        raise ValueError("All Casseau coefficient tables must have the same length.")
    equilibrium_coeffs_casseau = jnp.stack(parsed_tables, axis=0)

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

        # Reaction type flags
        is_dissociation = is_dissociation.at[r].set(
            1.0 if rxn.get("is_dissociation", False) else 0.0
        )
        is_electron_impact = is_electron_impact.at[r].set(
            1.0 if rxn.get("is_electron_impact", False) else 0.0
        )

    # Convert Arrhenius prefactors from per-molecule to per-mol units.
    reaction_order = jnp.sum(reactant_stoich, axis=1)
    C_f = C_f * jnp.power(constants.N_A, reaction_order - 1.0)


    if chemistry_model_config.model.lower() == "cvdv_qp":
        chemistry_model = chemistry.build_cvdv_qp_chemistry_model()
    elif chemistry_model_config.model.lower() == "park":
        chemistry_model = chemistry.build_park_chemistry_model(chemistry_model_config)
    else:
        raise ValueError(
            f"Unsupported chemistry model: {chemistry_model_config.model}"
        )

    return ReactionTable(
        species_names=tuple(species_names),
        reactant_stoich=reactant_stoich,
        product_stoich=product_stoich,
        C_f=C_f,
        n_f=n_f,
        E_f_over_k=E_f_over_k,
        equilibrium_coeffs_casseau=equilibrium_coeffs_casseau,
        is_dissociation=is_dissociation,
        is_electron_impact=is_electron_impact,
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
        equilibrium_coeffs_casseau=reaction_table.equilibrium_coeffs_casseau[
            indices_array
        ],
        is_dissociation=reaction_table.is_dissociation[indices_array],
        is_electron_impact=reaction_table.is_electron_impact[indices_array],
        preferential_factor=reaction_table.preferential_factor,
        chemistry_model=reaction_table.chemistry_model,
    )
