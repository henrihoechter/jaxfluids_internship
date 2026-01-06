from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import jaxtyping as jt
from jaxtyping import Float
from compressible_1d import constants


def _compute_is_monoatomic(
    dissociation_energy: Float[jt.Array, " n_species"],
) -> Float[jt.Array, " n_species"]:
    """Compute boolean mask indicating which species are monoatomic (atoms).

    This is a local helper to avoid circular imports. The canonical version
    is in equation_manager_utils.compute_is_monoatomic().

    Args:
        dissociation_energy: Dissociation energy array [J], NaN for atoms

    Returns:
        Boolean array where True = atom (monoatomic), False = molecule
    """
    return ~jnp.isfinite(dissociation_energy)


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class Species:
    name: str = field(metadata=dict(static=True))
    """Name of the species."""
    molar_mass: float
    """Molar mass of species.
      
    [amu]
    """
    h_s0: float
    """Enthalpy at reference state.
    
    [J/kg]
    """
    T_limit_low: Float[jt.Array, " n_ranges"]
    T_limit_high: Float[jt.Array, " n_ranges"]
    parameters: Float[jt.Array, "n_ranges n_parameters"]
    """Function to compute specific enthalpy

    # TODO(hhoechter): i think this is really ugly. i would love a Callable, but this
    # seems not compatible with vmap`ing over a list of Species. This makes the 
    # species unflexible and tailored for this specific enthalpy calculation.
    
    [J/kg]
    """
    dissociation_energy: float | None = None
    """Dissociation energy, None for monoatomic species.
    
    [J]
    """
    ionization_energy: float | None = None
    """Ionization energy, None for ionized species.
    
    [J]
    """
    vibrational_relaxation_factor: float | None = None
    """Vibrational relaxation factor, None for monoatomic species.

    [-]
    """

    def h(self, T_V: Float[jt.Array, " N"]) -> Float[jt.Array, " N"]:
        """Compute specific enthalpy at temperature T_V."""

        h_V = jnp.zeros_like(T_V)

        for i in range(self.T_limit_low.shape[0]):
            mask = (T_V >= self.T_limit_low[i]) & (T_V < self.T_limit_high[i])
            T_range = T_V[mask]

            a = self.parameters[i, :]

            h_range = (
                constants.R_universal
                / self.molar_mass
                * (  # this is ugly
                    # h_range = (
                    a[0] * T_range**1 / 1
                    + a[1] * T_range**2 / 2
                    + a[2] * T_range**3 / 3
                    + a[3] * T_range**4 / 4
                    + a[4] * T_range**5 / 5
                    + a[5]
                )
            )  # [J/kg]

            h_V = h_V.at[mask].set(h_range)

        return h_V


@jax.tree_util.register_dataclass
@dataclass
class SpeciesTable:
    """Vectorized species data structure for JAX processing.

    All data is stored as JAX arrays for efficient vectorized operations.
    Thermodynamic properties are computed via pure functions in thermodynamic_relations.py
    that take this SpeciesTable as an argument.

    This design makes SpeciesTable JIT-compatible (no callables/closures).
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

    # Polynomial coefficient data for thermodynamic calculations
    T_limit_low: Float[jt.Array, "n_species n_ranges"]  # [K]
    T_limit_high: Float[jt.Array, "n_species n_ranges"]  # [K]
    enthalpy_coeffs: Float[jt.Array, "n_species n_ranges n_coeffs"]  # NASA coeffs

    # Derived properties
    is_monoatomic: Float[
        jt.Array, " n_species"
    ]  # Boolean: True = atom, False = molecule

    # Reference temperature for vibrational energy integration
    T_ref: float  # [K], typically 298.16

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

        # Check shape consistency for coefficient arrays
        assert (
            self.T_limit_low.shape[0] == n_sp
        ), f"T_limit_low first dim {self.T_limit_low.shape[0]} != n_species ({n_sp})"
        assert (
            self.T_limit_high.shape[0] == n_sp
        ), f"T_limit_high first dim {self.T_limit_high.shape[0]} != n_species ({n_sp})"
        assert (
            self.enthalpy_coeffs.shape[0] == n_sp
        ), f"enthalpy_coeffs first dim {self.enthalpy_coeffs.shape[0]} != n_species ({n_sp})"
        assert self.is_monoatomic.shape == (
            n_sp,
        ), f"is_monoatomic shape {self.is_monoatomic.shape} != ({n_sp},)"

        # Check physical constraints
        # assert jnp.all(self.molar_masses > 0), "All molar masses must be positive"
        # assert self.T_ref > 0, "T_ref must be positive"

    @property
    def n_species(self) -> int:
        """Number of species in the table."""
        return len(self.names)

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
            return self.names.index("electron")
        except ValueError:
            return None

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

    @classmethod
    def from_species_list(cls, species_list: list[Species]) -> "SpeciesTable":
        """Construct SpeciesTable from a list of Species objects.

        Args:
            species_list: List of Species objects to vectorize

        Returns:
            SpeciesTable with all data as vectorized arrays

        Raises:
            ValueError: If species_list is empty or data is inconsistent
        """
        if not species_list:
            raise ValueError("species_list cannot be empty")

        n_species = len(species_list)

        # Check all species have same number of temperature ranges
        n_ranges_per_species = [s.T_limit_low.shape[0] for s in species_list]
        if len(set(n_ranges_per_species)) != 1:
            raise ValueError(
                f"All species must have same number of temperature ranges. "
                f"Found: {dict(zip([s.name for s in species_list], n_ranges_per_species))}"
            )

        # Extract names as tuple
        names = tuple(s.name for s in species_list)

        # Vectorize scalar properties
        molar_masses = jnp.array([s.molar_mass for s in species_list])
        h_s0_array = jnp.array([s.h_s0 for s in species_list])

        # Vectorize optional properties (NaN for None)
        dissociation_energy = jnp.array(
            [
                s.dissociation_energy if s.dissociation_energy is not None else jnp.nan
                for s in species_list
            ]
        )
        ionization_energy = jnp.array(
            [
                s.ionization_energy if s.ionization_energy is not None else jnp.nan
                for s in species_list
            ]
        )
        vibrational_relaxation_factor = jnp.array(
            [
                (
                    s.vibrational_relaxation_factor
                    if s.vibrational_relaxation_factor is not None
                    else jnp.nan
                )
                for s in species_list
            ]
        )

        # Stack temperature ranges and polynomial coefficients
        T_limit_low = jnp.stack([s.T_limit_low for s in species_list], axis=0)
        T_limit_high = jnp.stack([s.T_limit_high for s in species_list], axis=0)
        enthalpy_coeffs = jnp.stack([s.parameters for s in species_list], axis=0)

        # Compute is_monoatomic using the local helper function
        is_monoatomic = _compute_is_monoatomic(dissociation_energy)

        # Reference temperature for vibrational energy integration
        T_ref = 298.16  # [K]

        return cls(
            names=names,
            molar_masses=molar_masses,
            h_s0=h_s0_array,
            dissociation_energy=dissociation_energy,
            ionization_energy=ionization_energy,
            vibrational_relaxation_factor=vibrational_relaxation_factor,
            T_limit_low=T_limit_low,
            T_limit_high=T_limit_high,
            enthalpy_coeffs=enthalpy_coeffs,
            is_monoatomic=is_monoatomic,
            T_ref=T_ref,
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
