"""Types for vibrational and electronic energy models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import jax
from jaxtyping import Array, Float

EnergyFn = Callable[[Float[Array, " N"]], Float[Array, "n_species N"]]


@jax.tree_util.register_dataclass
@dataclass(frozen=True, eq=False)
class EnergyModel:
    """Store the energy-model callables used by the solver.

    Attributes:
        e_vib: Callable returning vibrational energy per species [J/kg].
        e_el: Callable returning electronic energy per species [J/kg].
        e_ve: Callable returning combined vibrational-electronic energy per
            species [J/kg].
        cv_ve: Callable returning vibrational-electronic heat capacity per
            species [J/kg/K].
        cp: Callable returning constant-pressure heat capacity per species
            [J/kg/K].
    """

    e_vib: EnergyFn = field(metadata=dict(static=True))
    e_el: EnergyFn = field(metadata=dict(static=True))
    e_ve: EnergyFn = field(metadata=dict(static=True))
    cv_ve: EnergyFn = field(metadata=dict(static=True))
    cp: EnergyFn = field(metadata=dict(static=True))


@dataclass(frozen=True)
class EnergyModelConfig:
    """Configure how the energy model is built.

    Attributes:
        model: Energy-model family to build.
        include_electronic: Whether electronic energy modes are included.
        data_path: Optional path to external model data needed by the selected
            energy model.
    """

    model: Literal["gnoffo", "bird"] = "gnoffo"
    include_electronic: bool = True
    data_path: str | None = None
