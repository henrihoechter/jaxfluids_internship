import dataclasses
from typing import Literal
import jax

from compressible_1d import chemistry_types
from compressible_1d import numerics_types


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class EquationManager:
    species: chemistry_types.SpeciesTable
    collision_integrals: chemistry_types.CollisionIntegralTable | None
    reactions: tuple[chemistry_types.Reactions] | None

    numerics_config: numerics_types.NumericsConfig

    boundary_condition: Literal["periodic", "reflective", "transmissive"] = (
        dataclasses.field(metadata=dict(static=True))
    )
