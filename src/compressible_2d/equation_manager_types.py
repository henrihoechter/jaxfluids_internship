import dataclasses
from typing import Literal

from jaxtyping import Array, Bool, Float, Int

from compressible_core import chemistry_types
from compressible_core import transport_casseau
from .numerics_types import NumericsConfig2D


@dataclasses.dataclass(frozen=True, slots=True)
class TransportModelConfig:
    model: Literal["gnoffo", "casseau"] = dataclasses.field(default="gnoffo")
    include_diffusion: bool = dataclasses.field(default=True)


@dataclasses.dataclass(frozen=True, slots=True)
class BoundaryConditionConfig2D:
    """Boundary condition mapping from physical tags to config dictionaries.

    Each entry should be of the form:
        {tag: {"type": "inflow"|"outflow"|"wall"|"wall_slip"|"axisymmetric", ...params}}
    """

    tag_to_bc: dict[int, dict]


@dataclasses.dataclass(frozen=True, slots=True)
class BoundaryConditionArrays2D:
    bc_id: Int[Array, "n_faces"]
    inflow_rho: Float[Array, "n_faces"]
    inflow_u: Float[Array, "n_faces"]
    inflow_v: Float[Array, "n_faces"]
    inflow_T: Float[Array, "n_faces"]
    inflow_Tv: Float[Array, "n_faces"]
    inflow_Y: Float[Array, "n_faces n_species"]
    wall_Tw: Float[Array, "n_faces"]
    wall_Tvw: Float[Array, "n_faces"]
    wall_has_Tw: Bool[Array, "n_faces"]
    wall_has_Tvw: Bool[Array, "n_faces"]
    wall_Y: Float[Array, "n_faces n_species"]
    wall_has_Y: Bool[Array, "n_faces"]
    wall_u: Float[Array, "n_faces"]
    wall_v: Float[Array, "n_faces"]
    wall_sigma_t: Float[Array, "n_faces"]
    wall_sigma_v: Float[Array, "n_faces"]
    wall_dist: Float[Array, "n_faces"]


@dataclasses.dataclass(frozen=True, slots=True)
class EquationManager2D:
    species: chemistry_types.SpeciesTable
    collision_integrals: chemistry_types.CollisionIntegralTable | None
    reactions: chemistry_types.ReactionTable | None
    numerics_config: NumericsConfig2D
    boundary_config: BoundaryConditionConfig2D
    boundary_arrays: BoundaryConditionArrays2D | None = None
    transport_model: TransportModelConfig = dataclasses.field(
        default_factory=TransportModelConfig
    )
    casseau_transport: transport_casseau.CasseauTransportTable | None = None
