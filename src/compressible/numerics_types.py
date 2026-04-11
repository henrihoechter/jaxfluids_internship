"""Unified numerics configuration for the compressible solver."""

from dataclasses import dataclass, field
from typing import Literal
import jax


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class ClippingConfig:
    """Clipping limits for primitive and conserved variables."""

    # Primitive variables
    rho_min: float = 1e-10
    rho_max: float = 1e10
    p_min: float = 1.0
    p_max: float = 1e10
    T_min: float = 100.0
    T_max: float = 50000.0
    Tv_min: float = 100.0
    Tv_max: float = 50000.0
    Y_min: float = 0.0
    Y_max: float = 1.0

    # Conserved variables
    rho_s_min: float = 1e-15
    rho_s_max: float = 1e10
    rho_u_min: float = -1e10
    rho_u_max: float = 1e10
    rho_v_min: float = -1e10
    rho_v_max: float = 1e10
    # Total energy can be negative in the reference-offset formulation used by
    # this solver, so clipping to a positive minimum injects energy.
    rho_E_min: float = -1e12
    rho_E_max: float = 1e12
    rho_Ev_min: float = 0.0
    rho_Ev_max: float = 1e12

    # Transport properties
    D_s_min: float = 0.0
    D_s_max: float = 1e2


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class NumericsConfig:
    """Unified numerics config. Superset of NumericsConfig (1D) and NumericsConfig2D.

    All Literal/bool fields are static (metadata=dict(static=True)) so JAX JIT
    compiles a separate kernel for each distinct configuration — 1D and 2D cases
    will always produce different compiled kernels.

    Args:
        dt: Fixed timestep (None → use CFL-adaptive stepping).
        cfl: CFL number used when dt is None.
        dt_mode: "fixed" or "cfl".
        integrator_scheme: "forward-euler" or "rk2".
        spatial_scheme: "first_order" or "muscl".
            MUSCL requires Mesh.muscl_ll / muscl_rr stencil arrays to be set
            (populated by Mesh.from_1d_grid; set to -1 for unstructured 2D).
        flux_scheme: "hllc", "exact_riemann", or "lax_friedrichs".
            exact_riemann and lax_friedrichs are 1D-only by convention; they
            operate on the normal-frame state and work for any mesh, but are
            physically meaningful only for 1D problems.
        slope_limiter: "minmod" or "mc". Used only when spatial_scheme="muscl".
        Geometry weighting (Cartesian vs. axisymmetric) is determined by the mesh,
            not by NumericsConfig.
        clipping: Clipping limits applied after primitive extraction.
    """

    dt: float | None = field(metadata=dict(static=True))
    cfl: float = field(default=0.4, metadata=dict(static=True))
    dt_mode: Literal["fixed", "cfl"] = field(
        default="fixed", metadata=dict(static=True)
    )
    integrator_scheme: Literal["forward-euler", "rk2"] = field(
        default="rk2", metadata=dict(static=True)
    )
    spatial_scheme: Literal["first_order", "muscl"] = field(
        default="first_order", metadata=dict(static=True)
    )
    flux_scheme: Literal["hllc", "exact_riemann", "lax_friedrichs"] = field(
        default="hllc", metadata=dict(static=True)
    )
    slope_limiter: Literal["minmod", "mc"] = field(
        default="mc", metadata=dict(static=True)
    )
    clipping: ClippingConfig = field(default_factory=ClippingConfig)
