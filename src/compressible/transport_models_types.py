"""Types for transport models and transport data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import jax
from jaxtyping import Array, Float, Int


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
    """Store the transport property callable used by the solver."""

    compute_transport_properties: TransportFn = field(metadata=dict(static=True))


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class TransportModelConfig:
    """Configure how the transport model is built."""

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
    """Store collision integrals for the Gnoffo transport model."""

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
        """Return the table index for a species pair."""
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
    """Store Casseau transport coefficients for a species set."""

    species_names: tuple[str, ...] = field(metadata=dict(static=True))
    d_ref: Float[Array, " n_species"]
    omega: Float[Array, " n_species"]
    blottner_A: Float[Array, " n_species"]
    blottner_B: Float[Array, " n_species"]
    blottner_C: Float[Array, " n_species"]
    T_ref: float = field(metadata=dict(static=True))
