"""Types for Casseau transport data."""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
from jaxtyping import Array, Float


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class CasseauTransportTable:
    """Casseau transport data for a set of species."""

    species_names: tuple[str, ...] = field(metadata=dict(static=True))
    d_ref: Float[Array, " n_species"]  # [m]
    omega: Float[Array, " n_species"]  # [-]
    blottner_A: Float[Array, " n_species"]  # [-]
    blottner_B: Float[Array, " n_species"]  # [-]
    blottner_C: Float[Array, " n_species"]  # [-]
    T_ref: float = field(metadata=dict(static=True))
