"""Types for transport models and transport data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import jax
from jaxtyping import Array, Float

TransportFn = Callable[
    [
        Float[Array, " n_cells"],
        Float[Array, " n_cells"],
        Float[Array, " n_cells"],
        Float[Array, "n_cells n_species"],
        Float[Array, " n_cells"],
    ],
    tuple[
        Float[Array, " n_cells"],
        Float[Array, " n_cells"],
        Float[Array, " n_cells"],
        Float[Array, " n_cells"],
        Float[Array, "n_cells n_species"],
    ],
]


@jax.tree_util.register_dataclass
@dataclass(frozen=True, eq=False)
class TransportModel:
    """Store the transport-property callable used by the solver.

    Attributes:
        compute_transport_properties: Callable returning the viscosity,
            translational conductivity, rotational conductivity, vibrational
            conductivity, and species diffusion coefficients for a state.
    """

    compute_transport_properties: TransportFn = field(metadata=dict(static=True))


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class TransportModelConfig:
    """Configure how the transport model is built.

    Attributes:
        model: Transport-model family to build.
        include_diffusion: Whether species diffusion coefficients are computed.
        collision_integrals_path: Optional path to collision-integral data for
            the Gnoffo model.
        casseau_data_path: Optional path to species transport coefficients for
            the Casseau model.
    """

    model: Literal["gnoffo", "casseau"] = field(
        default="gnoffo", metadata=dict(static=True)
    )
    include_diffusion: bool = field(default=True, metadata=dict(static=True))
    collision_integrals_path: str | None = field(
        default=None, metadata=dict(static=True)
    )
    casseau_data_path: str | None = field(default=None, metadata=dict(static=True))


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class CollisionIntegralTable:
    """Store collision integrals for the Gnoffo transport model.

    Attributes:
        species_pairs: Ordered species pairs corresponding to each table row.
        omega_11_2000K: Log10 collision integrals `pi*Omega^(1,1)` at 2000 K.
        omega_11_4000K: Log10 collision integrals `pi*Omega^(1,1)` at 4000 K.
        omega_22_2000K: Log10 collision integrals `pi*Omega^(2,2)` at 2000 K.
        omega_22_4000K: Log10 collision integrals `pi*Omega^(2,2)` at 4000 K.
    """

    species_pairs: tuple[tuple[str, str], ...] = field(metadata=dict(static=True))
    omega_11_2000K: Float[Array, " n_pairs"]
    omega_11_4000K: Float[Array, " n_pairs"]
    omega_22_2000K: Float[Array, " n_pairs"]
    omega_22_4000K: Float[Array, " n_pairs"]

    def __post_init__(self) -> None:
        """Validate the collision-integral array shapes."""
        n_pairs = len(self.species_pairs)
        if self.omega_11_2000K.shape != (n_pairs,):
            raise ValueError("omega_11_2000K has an inconsistent shape.")
        if self.omega_11_4000K.shape != (n_pairs,):
            raise ValueError("omega_11_4000K has an inconsistent shape.")
        if self.omega_22_2000K.shape != (n_pairs,):
            raise ValueError("omega_22_2000K has an inconsistent shape.")
        if self.omega_22_4000K.shape != (n_pairs,):
            raise ValueError("omega_22_4000K has an inconsistent shape.")

    @property
    def n_pairs(self) -> int:
        """Return the number of stored species pairs."""
        return len(self.species_pairs)

    def get_pair_index(self, species_s: str, species_r: str) -> int:
        """Return the table index for a species pair.

        Args:
            species_s: Name of the first species in the pair.
            species_r: Name of the second species in the pair.

        Returns:
            Index of the matching pair in the collision-integral arrays.

        Raises:
            ValueError: If neither the ordered nor reversed pair is present.
        """
        try:
            return self.species_pairs.index((species_s, species_r))
        except ValueError:
            pass

        try:
            return self.species_pairs.index((species_r, species_s))
        except ValueError as error:
            raise ValueError(
                f"Species pair ({species_s}, {species_r}) not found."
            ) from error


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class CasseauTransportTable:
    """Store Casseau transport coefficients for a species set.

    Attributes:
        species_names: Ordered species names matching every coefficient array.
        d_ref: Reference diffusion diameters used by the Casseau model.
        omega: Viscosity-temperature exponents used by the Casseau model.
        blottner_A: First Blottner viscosity-fit coefficient per species.
        blottner_B: Second Blottner viscosity-fit coefficient per species.
        blottner_C: Third Blottner viscosity-fit coefficient per species.
        T_ref: Reference temperature used by the Casseau fits [K].
    """

    species_names: tuple[str, ...] = field(metadata=dict(static=True))
    d_ref: Float[Array, " n_species"]
    omega: Float[Array, " n_species"]
    blottner_A: Float[Array, " n_species"]
    blottner_B: Float[Array, " n_species"]
    blottner_C: Float[Array, " n_species"]
    T_ref: float = field(metadata=dict(static=True))
