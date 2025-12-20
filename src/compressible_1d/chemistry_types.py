from dataclasses import dataclass

import jax.numpy as jnp
import jaxtyping as jt
from jaxtyping import Float
from compressible_1d import constants


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
                constants.R_bar
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
    """Data structure containing multiple species in a format suitable for JAX
    processing."""


@dataclass(frozen=True, slots=True)
class Reactions:
    reactants: str
    products: str

    def __post_init__(self):
        raise NotImplementedError("Reactions are not implemented yet.")
