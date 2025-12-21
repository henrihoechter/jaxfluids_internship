import dataclasses
from typing import Literal

from compressible_1d import chemistry_types
from compressible_1d import numerics_types


@dataclasses.dataclass(frozen=True, slots=True)
class EquationManager:
    species: list[chemistry_types.Species]
    reactions: list[chemistry_types.Reactions] | None

    numerics_config: numerics_types.NumericsConfig

    boundary_condition: Literal["periodic", "reflective", "transmissive"]
