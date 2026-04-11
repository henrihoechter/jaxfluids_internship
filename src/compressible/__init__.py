"""Unified compressible solver for 1D and 2D multi-species two-temperature flow.

Public API
----------
Core types:
    EquationManager           — physics + numerics configuration
    BoundaryConditionArrays   — per-face BC data
    Mesh                      — unified mesh (1D via from_1d_grid, 2D via from_gmsh)
    NumericsConfig            — numerical scheme configuration
    ClippingConfig            — variable clipping limits

Simulation entry points:
    run(U_init, mesh, equation_manager, t_final, ...)
    run_scan(U_init, mesh, equation_manager, t_final, ...)   # fixed dt, JIT
    advance_one_step(U, mesh, equation_manager, dt)

State utilities:
    compute_U_from_primitives(Y_s, rho, u, v, T_tr, T_V, equation_manager)
    extract_primitives(U, equation_manager)      → Primitives
    upgrade_state_1d_to_unified(U_1d, n_species)

BC construction helpers:
    build_boundary_arrays_1d(mesh, bc_left, bc_right, n_species, ...)
    build_boundary_arrays_1d_periodic(mesh, n_species)
    build_boundary_arrays_2d(mesh, tag_to_bc, species)
"""

from compressible.equation_manager_types import (
    EquationManager,
    BoundaryConditionArrays,
)
from compressible.mesh import Mesh
from compressible.numerics_types import NumericsConfig, ClippingConfig
from compressible.state import (
    Primitives,
    compute_U_from_primitives,
    extract_primitives,
    extract_primitives_from_U,
)
from compressible.boundary_conditions_utils import (
    build_boundary_arrays_1d,
    build_boundary_arrays_1d_periodic,
    build_boundary_arrays_2d,
)
from compressible.equation_manager import (
    run,
    run_scan,
    advance_one_step,
    compute_divergence,
    compute_cfl_dt,
)
from compressible.utils import upgrade_state_1d_to_unified

__all__ = [
    # Types
    "EquationManager",
    "BoundaryConditionArrays",
    "Mesh",
    "NumericsConfig",
    "ClippingConfig",
    "Primitives",
    # State
    "compute_U_from_primitives",
    "extract_primitives",
    "extract_primitives_from_U",
    # BC helpers
    "build_boundary_arrays_1d",
    "build_boundary_arrays_1d_periodic",
    "build_boundary_arrays_2d",
    # Simulation
    "run",
    "run_scan",
    "advance_one_step",
    "compute_divergence",
    "compute_cfl_dt",
    # Migration
    "upgrade_state_1d_to_unified",
]
