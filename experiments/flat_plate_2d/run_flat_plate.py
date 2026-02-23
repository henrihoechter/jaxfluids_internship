"""Minimal flat-plate test: subsonic flow at an incidence angle.

Domain: rectangle [0, Lx] x [0, Ly]
  tag 1 (left):   inflow at angle alpha from horizontal
  tag 2 (right):  outflow
  tag 3 (top):    outflow
  tag 4 (bottom): wall_euler (flat plate)

Expected result: near the bottom wall the velocity vectors rotate to be
parallel to the plate (zero normal component), validating the wall_euler BC.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import matplotlib.collections as mc
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from compressible_core import chemistry_utils, energy_models
from compressible_2d import (
    equation_manager,
    equation_manager_types,
    equation_manager_utils,
    mesh_gmsh,
    numerics_types,
)


# ---- flow conditions --------------------------------------------------------

alpha_deg = 15.0
T_inf = 300.0
p_inf = 101325.0
R_N2 = 8314.46 / 28.014
rho_inf = p_inf / (R_N2 * T_inf)
a_inf = np.sqrt(1.4 * R_N2 * T_inf)
U_inf = 0.3 * a_inf  # Ma ~ 0.3
alpha = np.deg2rad(alpha_deg)
u_inf = U_inf * np.cos(alpha)
v_inf = -U_inf * np.sin(alpha)  # negative: flow directed downward toward the plate

# ---- structured rectangular mesh -------------------------------------------

Lx, Ly = 0.3, 0.15
Nx, Ny = 60, 30
dx, dy = Lx / Nx, Ly / Ny

nodes = np.array(
    [[i * dx, j * dy] for j in range(Ny + 1) for i in range(Nx + 1)],
    dtype=float,
)


def nid(i, j):
    return j * (Nx + 1) + i


# CCW quad: bottom-left, bottom-right, top-right, top-left
cells = [
    [nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)]
    for j in range(Ny)
    for i in range(Nx)
]

boundary_edges = []
for i in range(Nx):  # bottom: wall_euler  tag 4
    boundary_edges.append((nid(i, 0), nid(i + 1, 0), 4))
for i in range(Nx):  # top: outflow  tag 3
    boundary_edges.append((nid(i, Ny), nid(i + 1, Ny), 3))
for j in range(Ny):  # left: inflow  tag 1
    boundary_edges.append((nid(0, j), nid(0, j + 1), 1))
for j in range(Ny):  # right: outflow  tag 2
    boundary_edges.append((nid(Nx, j), nid(Nx, j + 1), 2))

mesh = mesh_gmsh.Mesh2D.from_cells(nodes, cells, boundary_edges)

# ---- species + equation manager ---------------------------------------------

repo_root = Path(__file__).resolve().parents[2]
data_dir = repo_root / "data"

species = chemistry_utils.load_species_table(
    species_names=("N2",),
    general_data_path=str(data_dir / "species.json"),
    energy_model_config=energy_models.EnergyModelConfig(
        model="bird",
        include_electronic=False,
        data_path=str(data_dir / "air_5_bird_energy.json"),
    ),
)

boundary_config = equation_manager_types.BoundaryConditionConfig2D(
    tag_to_bc={
        1: {
            "type": "inflow",
            "rho": rho_inf,
            "u": u_inf,
            "v": v_inf,
            "T": T_inf,
            "Tv": T_inf,
            "Y": [1.0],
        },
        2: {"type": "outflow"},
        3: {"type": "outflow"},
        4: {"type": "wall_euler"},
    }
)

numerics_config = numerics_types.NumericsConfig2D(
    dt=1e-7,
    cfl=0.4,
    dt_mode="fixed",
    integrator_scheme="rk2",
    spatial_scheme="first_order",
    flux_scheme="hllc",
    axisymmetric=False,
    clipping=numerics_types.ClippingConfig2D(),
)

eq_manager = equation_manager.build_equation_manager(
    mesh,
    species=species,
    collision_integrals=None,
    reactions=None,
    numerics_config=numerics_config,
    boundary_config=boundary_config,
    transport_model=None,
    casseau_transport=None,
)

# ---- initial condition (freestream everywhere) ------------------------------

n_cells = mesh.cell_areas.shape[0]
U_init = equation_manager_utils.compute_U_from_primitives(
    Y_s=jnp.ones((n_cells, 1)),
    rho=jnp.full((n_cells,), rho_inf),
    u=jnp.full((n_cells,), u_inf),
    v=jnp.full((n_cells,), v_inf),
    T_tr=jnp.full((n_cells,), T_inf),
    T_V=jnp.full((n_cells,), T_inf),
    equation_manager=eq_manager,
)

# ---- run --------------------------------------------------------------------

t_final = 2e-4
save_interval = 100

print(
    f"Flat-plate: Ma={U_inf/a_inf:.2f}, alpha={alpha_deg}deg, "
    f"rho={rho_inf:.4f} kg/m3, U={U_inf:.1f} m/s"
)
print(f"  u_inf={u_inf:.2f} m/s, v_inf={v_inf:.2f} m/s")
print(f"  Mesh: {n_cells} cells ({Nx}x{Ny}), dt=1e-7 s, t_final={t_final:.1e} s")

U_hist, t_hist = equation_manager.run_scan(
    U_init, mesh, eq_manager, t_final, save_interval
)

# ---- plot -------------------------------------------------------------------

tri_i, tri_j, tri_k = [], [], []
for cn in mesh.cells:
    n = len(cn)
    if n == 3:
        tri_i += [cn[0]]
        tri_j += [cn[1]]
        tri_k += [cn[2]]
    elif n == 4:
        tri_i += [cn[0], cn[0]]
        tri_j += [cn[1], cn[2]]
        tri_k += [cn[2], cn[3]]

triang = mtri.Triangulation(
    mesh.nodes[:, 0],
    mesh.nodes[:, 1],
    np.column_stack([tri_i, tri_j, tri_k]),
)


def cell_to_tri(vals):
    out = []
    for ci, cn in enumerate(mesh.cells):
        out.append(vals[ci])
        if len(cn) == 4:
            out.append(vals[ci])
    return np.array(out)


_, rho, u, v, T, Tv, p = equation_manager_utils.extract_primitives_from_U(
    jnp.array(U_hist[-1]), eq_manager
)
u, v = np.array(u), np.array(v)
speed = np.sqrt(u**2 + v**2)

cx, cy = mesh.cell_centroids[:, 0], mesh.cell_centroids[:, 1]

rng = np.random.default_rng(42)
idx = rng.choice(len(cx), size=800, replace=False)
s = speed[idx] + 1e-30

fig, ax = plt.subplots(figsize=(12, 6))

tc = ax.tripcolor(triang, facecolors=cell_to_tri(speed), cmap="viridis", shading="flat")
fig.colorbar(tc, ax=ax, label="|v| [m/s]")

ax.quiver(
    cx[idx],
    cy[idx],
    u[idx] / s,
    v[idx] / s,
    speed[idx],
    cmap="autumn",
    clim=(0, speed.max()),
    angles="xy",
    scale=40,
    width=1.5e-3,
    headwidth=3,
    alpha=0.8,
)

# mark wall cells (bottom row) with larger arrows
wall_fi = np.where((mesh.face_right == -1) & (mesh.boundary_tags == 4))[0]
wall_cells = mesh.face_left[wall_fi]
wc, wv_u, wv_v = cx[wall_cells], u[wall_cells], v[wall_cells]
ws = np.sqrt(wv_u**2 + wv_v**2) + 1e-30
ax.quiver(
    wc,
    cy[wall_cells],
    wv_u / ws,
    wv_v / ws,
    color="red",
    angles="xy",
    scale=25,
    width=2.5e-3,
    headwidth=4,
    label="wall-adjacent cells",
)

boundary_segs = [
    [mesh.nodes[mesh.face_nodes[fi, 0]], mesh.nodes[mesh.face_nodes[fi, 1]]]
    for fi in np.where(mesh.face_right == -1)[0]
]
ax.add_collection(mc.LineCollection(boundary_segs, colors="black", linewidths=1.0))

ax.set_aspect("equal")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.legend(loc="upper right")
ax.set_title(
    f"Flat plate (wall_euler)  alpha={alpha_deg}deg  Ma={U_inf/a_inf:.2f}  "
    f"t = {float(t_hist[-1]):.2e} s\n"
    "Red = wall-adjacent cell velocity — should be nearly horizontal"
)

plt.tight_layout()
out_path = Path(__file__).parent / "flat_plate_velocity.png"
plt.savefig(out_path, dpi=150)
print(f"Saved to {out_path}")
plt.show()
