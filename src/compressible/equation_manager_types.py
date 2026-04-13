"""Unified EquationManager for the compressible solver."""

import dataclasses
import jax

from jaxtyping import Array, Bool, Float, Int

from . import chemistry_types
from . import transport_models_types
from compressible.numerics_types import NumericsConfig

TransportModel = transport_models_types.TransportModel


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class BoundaryConditionArrays:
    """Per-face boundary condition data.

    Works for both 1D degenerate meshes and 2D unstructured meshes.
    For 1D use, set all *_v fields to zero (no transverse velocity).

    bc_id values are defined in boundary_conditions_types.py:
        BC_OUTFLOW = 0       (outflow / zero-gradient)
        BC_INFLOW = 1        (prescribed inflow state)
        BC_WALL = 3          (no-slip isothermal)
        BC_WALL_SLIP = 4     (Maxwell/Smoluchowski slip)
        BC_WALL_EULER = 5    (inviscid slip / symmetry / axis)
        BC_REFLECTIVE = 6    (reflect normal momentum; for 1D reflective BC)

    Attributes:
        bc_id: Boundary-condition identifier for each face.
        inflow_rho: Prescribed inflow density for each face [kg/m^3].
        inflow_u: Prescribed inflow x-velocity for each face [m/s].
        inflow_v: Prescribed inflow y-velocity for each face [m/s].
        inflow_T: Prescribed inflow translational-rotational temperature [K].
        inflow_Tv: Prescribed inflow vibrational-electronic temperature [K].
        inflow_Y: Prescribed inflow species mole fractions for each face.
        wall_Tw: Wall translational temperature used by wall boundary models [K].
        wall_Tvw: Wall vibrational temperature used when a wall model prescribes
            a separate vibrational state [K].
        wall_has_Tw: Mask selecting faces with an explicit wall temperature.
        wall_has_Tvw: Mask selecting faces with an explicit vibrational wall
            temperature.
        wall_Y: Species composition applied at wall faces when provided.
        wall_has_Y: Mask selecting faces with an explicit wall composition.
        wall_u: Wall x-velocity for moving-wall or slip-wall models [m/s].
        wall_v: Wall y-velocity for moving-wall or slip-wall models [m/s].
        wall_sigma: Tangential momentum accommodation coefficient for slip-wall
            models.
        wall_alpha: Thermal accommodation coefficient for slip-wall models.
        wall_dist: Distance from the face centroid to the wall reference point
            used in slip/jump closures [m].
    """

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
    wall_sigma: Float[Array, "n_faces"]
    wall_alpha: Float[Array, "n_faces"]
    wall_dist: Float[Array, "n_faces"]


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class EquationManager:
    """Single unified equation manager for 1D and 2D simulations.

    The dimension of the problem is determined entirely by the Mesh passed to
    run() / advance_one_step(). For 1D, construct the mesh with
    Mesh.from_1d_grid(); the state vector is always n+4 (rhov=0 for 1D).

    Attributes:
        species: Species thermodynamic data used by all closures.
        reactions: Chemical reaction table. Set to `None` for frozen chemistry.
        numerics_config: Time-integration, reconstruction, and clipping settings.
        transport_model: Transport-property callable. Set to `None` for inviscid
            simulations.
        boundary_arrays: Per-face boundary-condition specification. Build with
            boundary_conditions_utils.build_boundary_arrays_1d() for 1D or
            boundary_conditions_utils.build_boundary_arrays_2d() for 2D.
    """

    species: chemistry_types.SpeciesTable
    reactions: chemistry_types.ReactionTable | None
    numerics_config: NumericsConfig
    transport_model: TransportModel | None = dataclasses.field(
        default=None, metadata=dict(static=True)
    )
    boundary_arrays: BoundaryConditionArrays | None = None
