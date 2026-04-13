"""Unified numerics configuration for the compressible solver."""

from dataclasses import dataclass, field
from typing import Literal
import jax


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class ClippingConfig:
    """Store clipping limits for primitive, conserved, and transport variables.

    Attributes:
        rho_min: Minimum allowed mixture density during primitive extraction.
        rho_max: Maximum allowed mixture density during primitive extraction.
        p_min: Minimum allowed pressure [Pa].
        p_max: Maximum allowed pressure [Pa].
        T_min: Minimum allowed translational-rotational temperature [K].
        T_max: Maximum allowed translational-rotational temperature [K].
        Tv_min: Minimum allowed vibrational-electronic temperature [K].
        Tv_max: Maximum allowed vibrational-electronic temperature [K].
        Y_min: Minimum allowed species mole fraction.
        Y_max: Maximum allowed species mole fraction.
        rho_s_min: Minimum allowed species partial density [kg/m^3].
        rho_s_max: Maximum allowed species partial density [kg/m^3].
        rho_u_min: Minimum allowed x-momentum density [kg/m^2/s].
        rho_u_max: Maximum allowed x-momentum density [kg/m^2/s].
        rho_v_min: Minimum allowed y-momentum density [kg/m^2/s].
        rho_v_max: Maximum allowed y-momentum density [kg/m^2/s].
        rho_E_min: Minimum allowed total energy density [J/m^3].
        rho_E_max: Maximum allowed total energy density [J/m^3].
        rho_Ev_min: Minimum allowed vibrational energy density [J/m^3].
        rho_Ev_max: Maximum allowed vibrational energy density [J/m^3].
        D_s_min: Minimum allowed effective species diffusion coefficient [m^2/s].
        D_s_max: Maximum allowed effective species diffusion coefficient [m^2/s].
    """

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
    """Store the numerical settings used by the solver.

    All Literal/bool fields are static (metadata=dict(static=True)) so JAX JIT
    compiles a separate kernel for each distinct configuration - 1D and 2D cases
    will always produce different compiled kernels.

    Attributes:
        dt: Fixed timestep [s]. Use `None` for CFL-based stepping.
        cfl: CFL number used when `dt` is `None`.
        dt_mode: Timestep selection mode, either `"fixed"` or `"cfl"`.
        integrator_scheme: Time integrator, either `"forward-euler"` or `"rk2"`.
        spatial_scheme: Spatial reconstruction scheme, either `"first_order"` or
            `"muscl"`.
            MUSCL requires Mesh.muscl_ll / muscl_rr stencil arrays to be set
            (populated by Mesh.from_1d_grid; set to -1 for unstructured 2D).
        flux_scheme: Numerical flux scheme, either `"hllc"`,
            `"exact_riemann"`, or `"lax_friedrichs"`.
            exact_riemann and lax_friedrichs are 1D-only by convention; they
            operate on the normal-frame state and work for any mesh, but are
            physically meaningful only for 1D problems.
        slope_limiter: Limiter used by MUSCL reconstruction, either `"minmod"`
            or `"mc"`.
        Geometry weighting (Cartesian vs. axisymmetric) is determined by the mesh,
            not by NumericsConfig.
        clipping: Clipping limits applied after primitive extraction and state
            updates.
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
