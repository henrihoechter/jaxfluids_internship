import dataclasses
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import jax
import jax.numpy as jnp
import jaxtyping as jt
from jaxtyping import Float

if TYPE_CHECKING:
    from compressible_1d.energy_models import EnergyModel


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
    dissociation_energy: Float[jt.Array, " n_species"]  # [J]
    ionization_energy: Float[jt.Array, " n_species"]  # [J]
    vibrational_relaxation_factor: Float[jt.Array, " n_species"]  # [-]

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
    energy_model: "EnergyModel" = field(metadata=dict(static=True))

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


@dataclass(frozen=True, slots=True)
class Reactions:
    reactants: str
    products: str

    def __post_init__(self):
        raise NotImplementedError("Reactions are not implemented yet.")


@dataclass(frozen=True, slots=True)
class ReactionTable:
    """Data structure containing multiple reactions in a format suitable for JAX
    processing."""


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
