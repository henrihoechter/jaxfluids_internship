import dataclasses
from typing import Literal
import jax

from compressible_core import chemistry_types
from compressible_core import transport_casseau
from compressible_1d import numerics_types


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class TransportModelConfig:
    """Configuration for transport property model selection."""

    model: Literal["gnoffo", "casseau"] = dataclasses.field(
        default="gnoffo", metadata=dict(static=True)
    )
    include_diffusion: bool = dataclasses.field(
        default=True, metadata=dict(static=True)
    )


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
        transport_model: Transport model selection and options.
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
    transport_model: TransportModelConfig = dataclasses.field(
        default=TransportModelConfig(), metadata=dict(static=True)
    )
    casseau_transport: transport_casseau.CasseauTransportTable | None = None
