from dataclasses import dataclass
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import jaxtyping as jt
from jaxtyping import Float, Array
from compressible_1d import constants
from compressible_1d import thermodynamic_relations


@dataclass(frozen=True, slots=True)
class Species:
    name: str
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
                / (self.molar_mass / 1e3)
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


@dataclass(frozen=True, slots=True)
class SpeciesTable:
    """Vectorized species data structure for JAX processing.

    All data is stored as JAX arrays for efficient vectorized operations.
    Thermodynamic properties are computed via callables (created with functools.partial)
    to allow flexible, JAX-compatible calculation models.
    """

    # Basic properties [n_species]
    names: tuple[str, ...]
    molar_masses: Float[jt.Array, " n_species"]  # [kg/kmol]
    h_s0: Float[jt.Array, " n_species"]  # [J/kg]

    # Optional properties [n_species] - use NaN for None
    dissociation_energy: Float[jt.Array, " n_species"]  # [J]
    ionization_energy: Float[jt.Array, " n_species"]  # [J]
    vibrational_relaxation_factor: Float[jt.Array, " n_species"]  # [-]

    # Thermodynamic property callables - ONE per property for all species
    h_equilibrium: Callable[[Float[Array, " N"]], Float[Array, "n_species N"]]
    """Equilibrium enthalpy calculator.

    Signature: h_equilibrium(T_V) -> h_all_species
    - Input: Temperature array [K], shape (N,)
    - Output: Enthalpy array [J/kg], shape (n_species, N)
    """

    cp_equilibrium: Callable[[Float[Array, " N"]], Float[Array, "n_species N"]]
    """Specific heat at constant pressure calculator.

    Signature: cp_equilibrium(T) -> cp_all_species
    - Input: Temperature array [K], shape (N,)
    - Output: Specific heat [J/(kg·K)], shape (n_species, N)

    Computed via differentiation of enthalpy polynomial: C_p = dh/dT
    """

    def __post_init__(self):
        """Validate data consistency."""
        n_sp = len(self.names)

        # Check shape consistency
        assert self.molar_masses.shape == (n_sp,), \
            f"molar_masses shape {self.molar_masses.shape} != ({n_sp},)"
        assert self.h_s0.shape == (n_sp,), \
            f"h_s0 shape {self.h_s0.shape} != ({n_sp},)"
        assert self.dissociation_energy.shape == (n_sp,), \
            f"dissociation_energy shape {self.dissociation_energy.shape} != ({n_sp},)"
        assert self.ionization_energy.shape == (n_sp,), \
            f"ionization_energy shape {self.ionization_energy.shape} != ({n_sp},)"
        assert self.vibrational_relaxation_factor.shape == (n_sp,), \
            f"vibrational_relaxation_factor shape {self.vibrational_relaxation_factor.shape} != ({n_sp},)"

        # Check physical constraints
        assert jnp.all(self.molar_masses > 0), "All molar masses must be positive"

        # Verify callables are present
        assert callable(self.h_equilibrium), "h_equilibrium must be callable"
        assert callable(self.cp_equilibrium), "cp_equilibrium must be callable"

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

    def h(self, T_V: Float[jt.Array, " N"]) -> Float[jt.Array, "n_species N"]:
        """Compute specific enthalpy for all species at given temperatures.

        Delegates to the h_equilibrium callable which encapsulates the
        temperature-dependent polynomial evaluation.

        Args:
            T_V: Temperature array [K]

        Returns:
            Enthalpy array [J/kg] for all species
        """
        return self.h_equilibrium(T_V)

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
        dissociation_energy = jnp.array([
            s.dissociation_energy if s.dissociation_energy is not None else jnp.nan
            for s in species_list
        ])
        ionization_energy = jnp.array([
            s.ionization_energy if s.ionization_energy is not None else jnp.nan
            for s in species_list
        ])
        vibrational_relaxation_factor = jnp.array([
            s.vibrational_relaxation_factor if s.vibrational_relaxation_factor is not None else jnp.nan
            for s in species_list
        ])

        # Stack temperature ranges and parameters for enthalpy calculation
        T_limit_low = jnp.stack([s.T_limit_low for s in species_list], axis=0)
        T_limit_high = jnp.stack([s.T_limit_high for s in species_list], axis=0)
        parameters = jnp.stack([s.parameters for s in species_list], axis=0)

        # Create equilibrium enthalpy callable with functools.partial
        # This captures the curve fit data in the closure
        h_equilibrium = partial(
            thermodynamic_relations.compute_equilibrium_enthalpy_polynomial,
            T_limit_low=T_limit_low,
            T_limit_high=T_limit_high,
            parameters=parameters,
            molar_masses=molar_masses,
        )

        # Create specific heat (C_p) callable using SAME polynomial data
        # C_p is computed via differentiation: C_p = dh/dT
        cp_equilibrium = partial(
            thermodynamic_relations.compute_cp_from_polynomial,
            T_limit_low=T_limit_low,
            T_limit_high=T_limit_high,
            parameters=parameters,  # Same coefficients as h_equilibrium!
            molar_masses=molar_masses,
        )

        return cls(
            names=names,
            molar_masses=molar_masses,
            h_s0=h_s0_array,
            dissociation_energy=dissociation_energy,
            ionization_energy=ionization_energy,
            vibrational_relaxation_factor=vibrational_relaxation_factor,
            h_equilibrium=h_equilibrium,
            cp_equilibrium=cp_equilibrium,
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