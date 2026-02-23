import dataclasses
import jax

from jaxtyping import Array, Bool, Float, Int

from compressible_core import chemistry_types
from compressible_core import transport_casseau_types
from compressible_core import transport_models_types
from .numerics_types import NumericsConfig2D

TransportModel = transport_models_types.TransportModel


@dataclasses.dataclass(frozen=True, slots=True)
class BoundaryConditionConfig2D:
    """Boundary condition mapping from physical tags to config dictionaries.

    Each entry should be of the form:
        {tag: {"type": "inflow"|"outflow"|"wall"|"wall_slip"|"axisymmetric", ...params}}
    """

    tag_to_bc: dict[int, dict]


@jax.tree_util.register_dataclass
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


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class EquationManager2D:
    species: chemistry_types.SpeciesTable
    collision_integrals: chemistry_types.CollisionIntegralTable | None
    reactions: chemistry_types.ReactionTable | None
    numerics_config: NumericsConfig2D
    boundary_arrays: BoundaryConditionArrays2D
    transport_model: TransportModel | None = dataclasses.field(default=None)
    casseau_transport: transport_casseau_types.CasseauTransportTable | None = None
