import dataclasses
from typing import Literal
import jax

from compressible_core import chemistry_types
from compressible_core import transport_casseau_types
from compressible_core import transport_models_types
from compressible_1d import numerics_types

TransportModel = transport_models_types.TransportModel


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
        transport_model: Transport model callable (or None for inviscid).
        casseau_transport: Casseau transport data (optional).
    """

    species: chemistry_types.SpeciesTable
    collision_integrals: chemistry_types.CollisionIntegralTable | None = (
        dataclasses.field(metadata=dict(static=True))
    )
    reactions: chemistry_types.ReactionTable | None

    numerics_config: numerics_types.NumericsConfig

    boundary_condition: Literal["periodic", "reflective", "transmissive"] = (
        dataclasses.field(metadata=dict(static=True))
    )
    transport_model: TransportModel | None = dataclasses.field(
        default=None, metadata=dict(static=True)
    )
    casseau_transport: transport_casseau_types.CasseauTransportTable | None = None
