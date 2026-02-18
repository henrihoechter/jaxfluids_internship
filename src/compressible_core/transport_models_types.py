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
    """Container for transport property computation callables."""

    compute_transport_properties: TransportFn = field(metadata=dict(static=True))


@dataclass(frozen=True)
class TransportModelConfig:
    """Configuration for selecting transport property models."""

    model: Literal["gnoffo", "casseau"] = "gnoffo"
    include_diffusion: bool = True
