import dataclasses
from typing import Literal
import jax

from compressible_1d import chemistry_types
from compressible_1d import numerics_types


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class EquationManager:
    """Container for all equation-related data and configuration.

    Attributes:
        species: Species thermodynamic and transport data.
        collision_integrals: Collision integral data for transport (optional).
        reactions: Chemical reaction mechanism (None for frozen chemistry).
        numerics_config: Numerical discretization parameters.
        boundary_condition: Type of boundary condition to apply.
    """

    species: chemistry_types.SpeciesTable
    collision_integrals: chemistry_types.CollisionIntegralTable | None
    reactions: tuple[chemistry_types.Reactions] | None

    numerics_config: numerics_types.NumericsConfig

    boundary_condition: Literal["periodic", "reflective", "transmissive"] = (
        dataclasses.field(metadata=dict(static=True))
    )
