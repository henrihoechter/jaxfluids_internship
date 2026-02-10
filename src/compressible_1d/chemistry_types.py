from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import jax
import jax.numpy as jnp
import jaxtyping as jt
from jaxtyping import Float

if TYPE_CHECKING:
    from compressible_1d.energy_models import EnergyModel
    from compressible_1d.reaction_rates import ChemistryModel


@jax.tree_util.register_dataclass
@dataclass
class SpeciesTable:
    """Vectorized species data structure for JAX processing.

    All data is stored as JAX arrays for efficient vectorized operations.
    Thermodynamic properties are computed via pure functions in thermodynamic_relations.py
    that take this SpeciesTable as an argument.

    This design keeps SpeciesTable JIT-compatible by storing energy model callables
    as static fields (not traced by JAX).
    """

    # Basic properties [n_species]
    names: tuple[str, ...] = field(metadata=dict(static=True))
    # names: tuple[str, ...]
    molar_masses: Float[jt.Array, " n_species"]  # [kg/mol]
    h_s0: Float[jt.Array, " n_species"]  # [J/kg]

    # Optional properties [n_species] - use NaN for None
    dissociation_energy: Float[jt.Array, " n_species"]  # [J/kg]
    ionization_energy: Float[jt.Array, " n_species"]  # [J]
    vibrational_relaxation_factor: Float[jt.Array, " n_species"]  # [-]
    vibrational_relaxation_a_ms: Float[jt.Array, " n_molecules n_species"]  # [-]
    vibrational_relaxation_b_ms: Float[jt.Array, " n_molecules n_species"]  # [-]
    vibrational_relaxation_molecule_indices: Int[jt.Array, " n_molecules"]
    vibrational_relaxation_partner_indices: Int[jt.Array, " n_species"]
    theta_vib: Float[jt.Array, " n_species"]  # [K]

    # Charge state [n_species]
    charge: Float[
        jt.Array, " n_species"
    ]  # [-1 for electron, 0 for neutral, +1 for ion]

    # Electron-neutral collision cross-section coefficients (Eq. 66 from NASA TP-2867)
    # σₑₛ = sigma_es_a + sigma_es_b * T_e + sigma_es_c * T_e²
    # NaN for ions and electrons (use Coulomb formula instead)
    sigma_es_a: Float[jt.Array, " n_species"]  # [m²]
    sigma_es_b: Float[jt.Array, " n_species"]  # [m²/K]
    sigma_es_c: Float[jt.Array, " n_species"]  # [m²/K²]

    # Derived properties
    is_monoatomic: Float[
        jt.Array, " n_species"
    ]  # Boolean: True = atom, False = molecule

    # Reference temperature for vibrational energy integration
    T_ref: float  # [K], typically 298.16

    # Energy model callables (configured upfront; static for JIT)
    energy_model: EnergyModel = field(metadata=dict(static=True))

    def __post_init__(self):
        """Validate data consistency."""
        n_sp = len(self.names)

        # Check shape consistency for basic properties
        assert self.molar_masses.shape == (
            n_sp,
        ), f"molar_masses shape {self.molar_masses.shape} != ({n_sp},)"
        assert self.h_s0.shape == (n_sp,), f"h_s0 shape {self.h_s0.shape} != ({n_sp},)"
        assert self.dissociation_energy.shape == (
            n_sp,
        ), f"dissociation_energy shape {self.dissociation_energy.shape} != ({n_sp},)"
        assert self.ionization_energy.shape == (
            n_sp,
        ), f"ionization_energy shape {self.ionization_energy.shape} != ({n_sp},)"
        assert (
            self.vibrational_relaxation_factor.shape == (n_sp,)
        ), f"vibrational_relaxation_factor shape {self.vibrational_relaxation_factor.shape} != ({n_sp},)"
        n_mol = self.vibrational_relaxation_molecule_indices.shape[0]
        n_partners = self.vibrational_relaxation_partner_indices.shape[0]
        assert self.vibrational_relaxation_a_ms.shape == (
            n_mol,
            n_partners,
        ), (
            "vibrational_relaxation_a_ms shape "
            f"{self.vibrational_relaxation_a_ms.shape} != ({n_mol}, {n_partners})"
        )
        assert self.vibrational_relaxation_b_ms.shape == (
            n_mol,
            n_partners,
        ), (
            "vibrational_relaxation_b_ms shape "
            f"{self.vibrational_relaxation_b_ms.shape} != ({n_mol}, {n_partners})"
        )
        assert self.theta_vib.shape == (
            n_sp,
        ), f"theta_vib shape {self.theta_vib.shape} != ({n_sp},)"
        assert self.charge.shape == (
            n_sp,
        ), f"charge shape {self.charge.shape} != ({n_sp},)"
        assert self.sigma_es_a.shape == (
            n_sp,
        ), f"sigma_es_a shape {self.sigma_es_a.shape} != ({n_sp},)"
        assert self.sigma_es_b.shape == (
            n_sp,
        ), f"sigma_es_b shape {self.sigma_es_b.shape} != ({n_sp},)"
        assert self.sigma_es_c.shape == (
            n_sp,
        ), f"sigma_es_c shape {self.sigma_es_c.shape} != ({n_sp},)"

        assert self.is_monoatomic.shape == (
            n_sp,
        ), f"is_monoatomic shape {self.is_monoatomic.shape} != ({n_sp},)"

        # Check physical constraints
        # assert jnp.all(self.molar_masses > 0), "All molar masses must be positive"
        # assert self.T_ref > 0, "T_ref must be positive"

        if self.energy_model is None:
            raise ValueError("energy_model must be configured for SpeciesTable.")

    @property
    def n_species(self) -> int:
        """Number of species in the table."""
        return len(self.names)

    def with_energy_model(self, energy_model: "EnergyModel") -> "SpeciesTable":
        """Return a new SpeciesTable with a different energy model."""
        return dataclasses.replace(self, energy_model=energy_model)

    @property
    def M_s(self) -> Float[jt.Array, " n_species"]:
        """Alias for molar_masses for backward compatibility."""
        return self.molar_masses

    @property
    def has_dissociation_energy(self) -> Float[jt.Array, " n_species"]:
        """Boolean mask indicating which species have dissociation energy."""
        return jnp.isfinite(self.dissociation_energy)

    @property
    def has_ionization_energy(self) -> Float[jt.Array, " n_species"]:
        """Boolean mask indicating which species have ionization energy."""
        return jnp.isfinite(self.ionization_energy)

    @property
    def has_vibrational_mode(self) -> Float[jt.Array, " n_species"]:
        """Boolean mask indicating which species have vibrational modes."""
        return jnp.isfinite(self.vibrational_relaxation_factor)

    @property
    def electron_index(self) -> int | None:
        """Index of electron species, or None if not present."""
        try:
            return self.names.index("e-")
        except ValueError:
            return None

    @property
    def is_electron(self) -> Float[jt.Array, " n_species"]:
        """Boolean mask for electron species (charge == -1)."""
        return self.charge == -1

    @property
    def is_ion(self) -> Float[jt.Array, " n_species"]:
        """Boolean mask for ionized species (charge > 0)."""
        return self.charge > 0

    @property
    def is_neutral(self) -> Float[jt.Array, " n_species"]:
        """Boolean mask for neutral species (charge == 0)."""
        return self.charge == 0

    def get_species_index(self, name: str) -> int:
        """Get the index of a species by name.

        Args:
            name: Species name (e.g., "N2", "O")

        Returns:
            Index of the species in the table

        Raises:
            ValueError: If species name not found
        """
        try:
            return self.names.index(name)
        except ValueError:
            raise ValueError(
                f"Species '{name}' not found. Available species: {self.names}"
            )


@jax.tree_util.register_dataclass
@dataclass
class ReactionTable:
    """Vectorized reaction data for JAX processing.

    Implements the chemical kinetic model from NASA TP-2867 (Gnoffo et al. 1989).

    The stoichiometry arrays define reactions of the form:
        Σ_s α_{s,r} [Species_s] ↔ Σ_s β_{s,r} [Species_s]

    Rate coefficients follow the Arrhenius form (Eq. 46a):
        k_{f,r} = C_{f,r} * T_q^{n_{f,r}} * exp(-E_{f,r} / k*T_q)

    where T_q is the rate-controlling temperature:
        - Dissociation reactions: T_d = √(T * T_v) (Park model)
        - Electron impact reactions: T_v
        - Other reactions: T

    Equilibrium constants use polynomial curve fit (Eq. 47):
        K_{c,r} = exp(B_1 + B_2*ln(Z) + B_3*Z + B_4*Z² + B_5*Z³)
        where Z = 10000/T

    Attributes:
        species_names: Ordered tuple of species names matching stoichiometry columns.
        reactant_stoich: Stoichiometric coefficients for reactants α_{s,r}.
            Shape [n_reactions, n_species].
        product_stoich: Stoichiometric coefficients for products β_{s,r}.
            Shape [n_reactions, n_species].
        C_f: Arrhenius pre-exponential factor [cgs units: cm³/mol/s or cm⁶/mol²/s].
            Shape [n_reactions].
        n_f: Arrhenius temperature exponent [-]. Shape [n_reactions].
        E_f_over_k: Activation energy divided by Boltzmann constant [K].
            Shape [n_reactions].
        equilibrium_coeffs: Polynomial coefficients B_1 to B_5 for K_c.
            Shape [n_reactions, 5].
        is_dissociation: Flag for dissociation reactions (use T_d = √(T*T_v)).
            Shape [n_reactions].
        is_electron_impact: Flag for electron impact reactions (use T_v).
            Shape [n_reactions].
        preferential_factor: Factor ĉ_2 for vibrational energy reactive source (Eq. 51).
            1.0 = nonpreferential, >1.0 = preferential dissociation.
    """

    # Species ordering (must match SpeciesTable)
    species_names: tuple[str, "n_species"] = field(metadata=dict(static=True))

    # Stoichiometry [n_reactions, n_species]
    reactant_stoich: Float[jt.Array, "n_reactions n_species"]  # α_{s,r}
    product_stoich: Float[jt.Array, "n_reactions n_species"]  # β_{s,r}

    # Arrhenius parameters (Eq. 46a)
    C_f: Float[jt.Array, " n_reactions"]  # Pre-exponential [cgs units]
    n_f: Float[jt.Array, " n_reactions"]  # Temperature exponent [-]
    E_f_over_k: Float[jt.Array, " n_reactions"]  # Activation energy / k [K]

    # Equilibrium constant polynomial (Eq. 47)
    # K_c = exp(B_1 + B_2*ln(Z) + B_3*Z + B_4*Z² + B_5*Z³), Z = 10000/T
    equilibrium_coeffs: Float[jt.Array, "n_reactions 5"]  # [B_1, B_2, B_3, B_4, B_5]

    # Reaction type flags
    is_dissociation: Float[jt.Array, " n_reactions"]  # Use T_d = √(T*T_v)
    is_electron_impact: Float[jt.Array, " n_reactions"]  # Use T_v

    # Chemistry model callables (configured upfront; static for JIT)
    chemistry_model: ChemistryModel = field(metadata=dict(static=True))

    # Preferential dissociation factor (Eq. 51)
    preferential_factor: float = 1.0  # ĉ_2: 1.0 = nonpreferential

    def __post_init__(self):
        """Validate data consistency."""
        n_reactions = self.C_f.shape[0]
        n_species = len(self.species_names)

        # Check stoichiometry shapes
        assert (
            self.reactant_stoich.shape
            == (
                n_reactions,
                n_species,
            )
        ), f"reactant_stoich shape {self.reactant_stoich.shape} != ({n_reactions}, {n_species})"
        assert (
            self.product_stoich.shape
            == (
                n_reactions,
                n_species,
            )
        ), f"product_stoich shape {self.product_stoich.shape} != ({n_reactions}, {n_species})"

        # Check Arrhenius parameter shapes
        assert self.C_f.shape == (
            n_reactions,
        ), f"C_f shape {self.C_f.shape} != ({n_reactions},)"
        assert self.n_f.shape == (
            n_reactions,
        ), f"n_f shape {self.n_f.shape} != ({n_reactions},)"
        assert self.E_f_over_k.shape == (
            n_reactions,
        ), f"E_f_over_k shape {self.E_f_over_k.shape} != ({n_reactions},)"

        # Check equilibrium coefficients shape
        assert (
            self.equilibrium_coeffs.shape
            == (
                n_reactions,
                5,
            )
        ), f"equilibrium_coeffs shape {self.equilibrium_coeffs.shape} != ({n_reactions}, 5)"

        # Check flag shapes
        assert self.is_dissociation.shape == (
            n_reactions,
        ), f"is_dissociation shape {self.is_dissociation.shape} != ({n_reactions},)"
        assert (
            self.is_electron_impact.shape == (n_reactions,)
        ), f"is_electron_impact shape {self.is_electron_impact.shape} != ({n_reactions},)"

    @property
    def n_reactions(self) -> int:
        """Number of reactions in the table."""
        return self.C_f.shape[0]

    @property
    def n_species(self) -> int:
        """Number of species in the reaction mechanism."""
        return len(self.species_names)

    @property
    def net_stoich(self) -> Float[jt.Array, "n_reactions n_species"]:
        """Net stoichiometric coefficients (β - α) for each reaction."""
        return self.product_stoich - self.reactant_stoich

    def with_chemistry_model(self, chemistry_model: "ChemistryModel") -> "ReactionTable":
        """Return a new ReactionTable with a different chemistry model."""
        return dataclasses.replace(self, chemistry_model=chemistry_model)

    def get_species_index(self, name: str) -> int:
        """Get the index of a species by name."""
        try:
            return self.species_names.index(name)
        except ValueError:
            raise ValueError(
                f"Species '{name}' not found. Available: {self.species_names}"
            )


@dataclass(frozen=True, slots=True)
class CollisionIntegralTable:
    """Collision integral data for transport property calculations.

    Based on NASA TP-2867 Table VI. Stores log10(π·Ω^(k,k)_sr) at reference
    temperatures (2000K and 4000K) for linear interpolation in ln(T).

    The interpolation formula (Eq. 67 from TP-2867) is:
        log10(πΩ^(k,k)_sr(T)) = log10(πΩ(2000)) + slope × [ln(T) - ln(2000)]
    where:
        slope = [log10(πΩ(4000)) - log10(πΩ(2000))] / [ln(4000) - ln(2000)]

    All collision integrals are stored in cm² units (as in TP-2867 Table VI).
    """

    # Species pair names (s, r) - order matters for lookup
    species_pairs: tuple[tuple[str, str], ...]

    # Collision integrals at reference temperatures [n_pairs]
    # Values are log10(π·Ω^(k,k)_sr) in cm²
    omega_11_2000K: Float[jt.Array, " n_pairs"]  # log10(πΩ^(1,1)) at T=2000K
    omega_11_4000K: Float[jt.Array, " n_pairs"]  # log10(πΩ^(1,1)) at T=4000K
    omega_22_2000K: Float[jt.Array, " n_pairs"]  # log10(πΩ^(2,2)) at T=2000K
    omega_22_4000K: Float[jt.Array, " n_pairs"]  # log10(πΩ^(2,2)) at T=4000K

    def __post_init__(self):
        """Validate data consistency."""
        n_pairs = len(self.species_pairs)

        assert self.omega_11_2000K.shape == (
            n_pairs,
        ), f"omega_11_2000K shape {self.omega_11_2000K.shape} != ({n_pairs},)"
        assert self.omega_11_4000K.shape == (
            n_pairs,
        ), f"omega_11_4000K shape {self.omega_11_4000K.shape} != ({n_pairs},)"
        assert self.omega_22_2000K.shape == (
            n_pairs,
        ), f"omega_22_2000K shape {self.omega_22_2000K.shape} != ({n_pairs},)"
        assert self.omega_22_4000K.shape == (
            n_pairs,
        ), f"omega_22_4000K shape {self.omega_22_4000K.shape} != ({n_pairs},)"

    @property
    def n_pairs(self) -> int:
        """Number of species pairs in the table."""
        return len(self.species_pairs)

    def get_pair_index(self, species_s: str, species_r: str) -> int:
        """Get the index of a species pair.

        Tries both orderings (s,r) and (r,s).

        Args:
            species_s: First species name
            species_r: Second species name

        Returns:
            Index of the pair in the table

        Raises:
            ValueError: If pair not found
        """
        try:
            return self.species_pairs.index((species_s, species_r))
        except ValueError:
            pass

        try:
            return self.species_pairs.index((species_r, species_s))
        except ValueError:
            raise ValueError(
                f"Species pair ({species_s}, {species_r}) not found. "
                f"Available pairs: {self.species_pairs[:5]}..."
            )
