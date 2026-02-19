"""Estimate wall-clock time for a full blunt-cone run by timing 20 steps.

Usage:
    python timing_20steps.py

The script mirrors the setup in subsonic_inviscid.ipynb exactly, runs one
warm-up step to trigger JIT compilation, then times 20 steps and extrapolates
to the full t_final.
"""

import os
os.environ["JAX_PLATFORM_NAME"] = "gpu"
# os.environ["XLA_FLAGS"] = "--jax_num_cpu_devices=1"
# os.environ["JAX_NUM_CPU_DEVICES"] = "16"

import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", False)

from compressible_core import chemistry_utils, energy_models
from compressible_2d import (
    mesh_gmsh,
    equation_manager,
    equation_manager_types,
    equation_manager_utils,
    numerics_types,
)

# ---- same constants as the notebook ----
T_inf   = 300.0
p_inf   = 101325.0
R_N2    = 8314.46 / 28.014
gamma   = 1.4
a_inf   = (gamma * R_N2 * T_inf) ** 0.5
mach    = 0.5
V_inf   = mach * a_inf
rho_inf = p_inf / (R_N2 * T_inf)

TAG_INFLOW       = 1
TAG_OUTFLOW      = 2
TAG_WALL         = 3
TAG_AXISYMMETRIC = 4

dt      = 1e-10
t_final = 1e-5

# ---- mesh ----
data_dir = repo_root / "data"
print("Loading mesh ...", flush=True)
t0 = time.perf_counter()
mesh = mesh_gmsh.read_gmsh_v2_wedge_plane(
    str(data_dir / "bluntedCone.msh"),
    wedge_plane_tag=4,
    remap_tags={7: TAG_AXISYMMETRIC},
    axis_tag=TAG_AXISYMMETRIC,
)
print(f"  {mesh.cell_areas.shape[0]} cells  ({time.perf_counter()-t0:.1f}s)")

# ---- species ----
energy_cfg = energy_models.EnergyModelConfig(
    model="bird",
    include_electronic=False,
    data_path=str(data_dir / "air_5_bird_energy.json"),
)
species = chemistry_utils.load_species_table(
    species_names=("N2",),
    general_data_path=str(data_dir / "species.json"),
    energy_model_config=energy_cfg,
)

# ---- equation manager ----
numerics_config = numerics_types.NumericsConfig2D(
    dt=dt,
    cfl=0.4,
    dt_mode="fixed",
    integrator_scheme="rk2",
    spatial_scheme="muscl",
    flux_scheme="hllc",
    axisymmetric=True,
    clipping=numerics_types.ClippingConfig2D(),
)
boundary_config = equation_manager_types.BoundaryConditionConfig2D(
    tag_to_bc={
        TAG_INFLOW: {
            "type": "inflow",
            "rho": rho_inf, "u": 0.0, "v": V_inf,
            "T": T_inf, "Tv": T_inf, "Y": [1.0],
        },
        TAG_OUTFLOW:       {"type": "outflow"},
        TAG_WALL:          {"type": "wall", "Tw": T_inf},
        TAG_AXISYMMETRIC:  {"type": "axisymmetric"},
    }
)
eq_manager = equation_manager_utils.build_equation_manager(
    mesh, species=species, collision_integrals=None, reactions=None,
    numerics_config=numerics_config, boundary_config=boundary_config,
    transport_model=None, casseau_transport=None,
)

# ---- initial condition ----
n_cells = mesh.cell_areas.shape[0]
U = equation_manager_utils.compute_U_from_primitives(
    Y_s=jnp.ones((n_cells, 1)),
    rho=jnp.full((n_cells,), rho_inf),
    u=jnp.zeros((n_cells,)),
    v=jnp.full((n_cells,), V_inf),
    T_tr=jnp.full((n_cells,), T_inf),
    T_V=jnp.full((n_cells,), T_inf),
    equation_manager=eq_manager,
)

# ---- warm-up (triggers JIT compilation) ----
print("Warm-up step (JIT compilation) ...", flush=True)
t0 = time.perf_counter()
U = equation_manager.advance_one_step(U, mesh, eq_manager, dt)
U.block_until_ready()
t_compile = time.perf_counter() - t0
print(f"  compile+run: {t_compile:.2f}s")

# ---- time 20 steps ----
N_BENCH = 20
print(f"Timing {N_BENCH} steps ...", flush=True)
t0 = time.perf_counter()
for _ in range(N_BENCH):
    U = equation_manager.advance_one_step(U, mesh, eq_manager, dt)
U.block_until_ready()
t_bench = time.perf_counter() - t0

t_per_step = t_bench / N_BENCH
n_steps_total = int(t_final / dt)
t_total_est = t_per_step * n_steps_total

print()
print(f"Results")
print(f"  {N_BENCH} steps wall time : {t_bench:.3f} s")
print(f"  time per step            : {t_per_step*1e3:.2f} ms")
print(f"  steps for t_final={t_final:.0e}: {n_steps_total:,}")
print(f"  estimated total time     : {t_total_est:.1f} s  ({t_total_est/60:.1f} min)")
