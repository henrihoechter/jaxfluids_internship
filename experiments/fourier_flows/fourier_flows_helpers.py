from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from compressible_core import (
    chemistry_utils,
    energy_models,
    transport_model_gnoffo_utils as transport_core,
    transport_model_casseau_utils as transport_casseau,
    transport_models_types,
    transport_models_utils,
)
from compressible_2d import mesh_gmsh
from compressible_2d import equation_manager
from compressible_2d import equation_manager_types
from compressible_2d import equation_manager_utils
from compressible_2d import numerics_types


def find_repo_root() -> Path:
    cwd = Path.cwd().resolve()
    for parent in [cwd, *cwd.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return cwd


def build_channel_mesh(nx: int, ny: int, Lx: float, H: float) -> mesh_gmsh.Mesh2D:
    """Structured quad mesh for a 2D channel."""
    x = np.linspace(0.0, Lx, nx + 1)
    y = np.linspace(0.0, H, ny + 1)
    nodes = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            nodes.append([x[i], y[j]])
    nodes = np.asarray(nodes, dtype=float)

    def node_id(i, j):
        return j * (nx + 1) + i

    cells = []
    for j in range(ny):
        for i in range(nx):
            n0 = node_id(i, j)
            n1 = node_id(i + 1, j)
            n2 = node_id(i + 1, j + 1)
            n3 = node_id(i, j + 1)
            cells.append([n0, n1, n2, n3])

    # Boundary tags
    TAG_LEFT = 1
    TAG_RIGHT = 2
    TAG_BOTTOM = 3
    TAG_TOP = 4

    boundary_edges = []
    # Left and right
    for j in range(ny):
        boundary_edges.append((node_id(0, j), node_id(0, j + 1), TAG_LEFT))
        boundary_edges.append((node_id(nx, j), node_id(nx, j + 1), TAG_RIGHT))
    # Bottom and top
    for i in range(nx):
        boundary_edges.append((node_id(i, 0), node_id(i + 1, 0), TAG_BOTTOM))
        boundary_edges.append((node_id(i, ny), node_id(i + 1, ny), TAG_TOP))

    return mesh_gmsh.Mesh2D.from_cells(nodes, cells, boundary_edges)


def load_species_table(species_names: tuple[str, ...]) -> chemistry_utils.SpeciesTable:
    repo_root = find_repo_root()
    data_dir = repo_root / "data"
    general_data = data_dir / "species.json"
    bird_data = data_dir / "air_5_bird_energy.json"

    energy_cfg = energy_models.EnergyModelConfig(
        model="bird",
        include_electronic=False,
        data_path=str(bird_data),
    )

    return chemistry_utils.load_species_table(
        species_names=species_names,
        general_data_path=str(general_data),
        energy_model_config=energy_cfg,
    )


def number_density_to_rho_Y(species, n_by_species):
    """Convert number densities (1/m^3) to rho and mass fractions."""
    m_s = species.molar_masses / 6.02214076e23  # kg per particle
    n_by_species = np.asarray(n_by_species, dtype=float)
    rho_s = n_by_species * m_s
    rho = np.sum(rho_s)
    Y = rho_s / rho
    return rho, Y


def make_equation_manager(
    species, collision_integrals, H, U, Tw_bottom, Tw_top, mesh, dt=1e-6
):
    TAG_LEFT = 1
    TAG_RIGHT = 2
    TAG_BOTTOM = 3
    TAG_TOP = 4

    repo_root = find_repo_root()
    data_dir = repo_root / "data"
    casseau_transport = transport_casseau.load_casseau_transport_table(
        data_dir / "air_5_casseau_transport.json", species.names
    )

    numerics_config = numerics_types.NumericsConfig2D(
        dt=dt,
        cfl=0.4,
        dt_mode="fixed",
        integrator_scheme="rk2",
        spatial_scheme="muscl",
        flux_scheme="hllc",
        axisymmetric=False,
        clipping=numerics_types.ClippingConfig2D(),
    )

    boundary_config = equation_manager_types.BoundaryConditionConfig2D(
        tag_to_bc={
            TAG_LEFT: {"type": "outflow"},
            TAG_RIGHT: {"type": "outflow"},
            TAG_BOTTOM: {
                "type": "wall_slip",
                "Tw": Tw_bottom,
                "u_wall": U,
                "v_wall": 0.0,
                "sigma_t": 1.0,
                "sigma_v": 1.0,
            },
            TAG_TOP: {
                "type": "wall_slip",
                "Tw": Tw_top,
                "u_wall": U,
                "v_wall": 0.0,
                "sigma_t": 1.0,
                "sigma_v": 1.0,
            },
        }
    )

    return equation_manager.build_equation_manager(
        mesh,
        species=species,
        collision_integrals=collision_integrals,
        reactions=None,
        numerics_config=numerics_config,
        boundary_config=boundary_config,
        transport_model=transport_models_utils.build_transport_model_from_config(
            transport_models_types.TransportModelConfig(
                model="casseau", include_diffusion=False
            ),
            species_table=species,
            collision_integrals=collision_integrals,
            casseau_transport=casseau_transport,
        ),
        casseau_transport=casseau_transport,
    )


def run_case(
    species_names,
    n_by_species,
    case_name,
    nx=1,
    ny=101,
    Lx=1.0,
    H=1.0,
    U=300.0,
    Tw_bottom=2000.0,
    Tw_top=3000.0,
    T_init=2500.0,
    t_final=5e-3,
    save_interval=10,
    dt=1e-6,
):
    species = load_species_table(species_names)

    repo_root = find_repo_root()
    ci_path = repo_root / "data" / "collision_integrals_tp2867.json"
    collision_integrals = transport_core.create_collision_integral_table_from_json(
        ci_path
    )

    mesh = build_channel_mesh(nx, ny, Lx, H)

    eq_manager = make_equation_manager(
        species, collision_integrals, H, U, Tw_bottom, Tw_top, mesh, dt=dt
    )

    rho, Y = number_density_to_rho_Y(species, n_by_species)
    n_cells = mesh.cell_areas.shape[0]
    Y_field = np.broadcast_to(Y[None, :], (n_cells, len(Y)))

    rho_field = np.full((n_cells,), rho)
    u_field = np.full((n_cells,), U)
    v_field = np.zeros((n_cells,))
    T_field = np.full((n_cells,), T_init)
    Tv_field = np.full((n_cells,), T_init)

    U_init = equation_manager_utils.compute_U_from_primitives(
        Y_s=jnp.asarray(Y_field),
        rho=jnp.asarray(rho_field),
        u=jnp.asarray(u_field),
        v=jnp.asarray(v_field),
        T_tr=jnp.asarray(T_field),
        T_V=jnp.asarray(Tv_field),
        equation_manager=eq_manager,
    )

    U_hist, t_hist = equation_manager.run_scan(
        U_init, mesh, eq_manager, t_final=t_final, save_interval=save_interval
    )
    return {
        "name": case_name,
        "mesh": mesh,
        "eq_manager": eq_manager,
        "U_hist": U_hist,
        "t_hist": t_hist,
        "Tw_bottom": Tw_bottom,
    }


def extract_profile(case, x_target=None):
    U = case["U_hist"][-1]
    return extract_profile_from_U(case, U, x_target=x_target)


def extract_profile_from_U(case, U, x_target=None):
    mesh = case["mesh"]
    eq_manager = case["eq_manager"]

    prim = equation_manager_utils.extract_primitives(U, eq_manager)
    x = mesh.cell_centroids[:, 0]
    y = mesh.cell_centroids[:, 1]

    if x_target is None:
        x_target = 0.5 * (x.min() + x.max())
    dx = np.min(np.diff(np.unique(x))) if len(np.unique(x)) > 1 else 1.0
    mask = np.abs(x - x_target) <= 0.25 * dx

    y_sel = y[mask]
    T_sel = np.asarray(prim.T)[mask]
    Tv_sel = np.asarray(prim.Tv)[mask]

    order = np.argsort(y_sel)
    y_sorted = y_sel[order]
    T_sorted = T_sel[order]
    Tv_sorted = Tv_sel[order]

    y_over_H = y_sorted / (y.max() - y.min())
    Tb0 = case["Tw_bottom"]
    T_norm = T_sorted / Tb0
    Tv_norm = Tv_sorted / Tb0

    return y_over_H, T_norm, Tv_norm


def extract_mole_fraction_profile_from_U(case, U, x_target=None):
    """Extract mole fraction profiles (one per species) along y for a given U."""
    mesh = case["mesh"]
    eq_manager = case["eq_manager"]

    prim = equation_manager_utils.extract_primitives(U, eq_manager)
    x = mesh.cell_centroids[:, 0]
    y = mesh.cell_centroids[:, 1]

    if x_target is None:
        x_target = 0.5 * (x.min() + x.max())
    dx = np.min(np.diff(np.unique(x))) if len(np.unique(x)) > 1 else 1.0
    mask = np.abs(x - x_target) <= 0.25 * dx

    y_sel = y[mask]
    # prim.Y_s is mole fractions (n_cells, n_species)
    X_sel = np.asarray(prim.Y_s)[mask]

    order = np.argsort(y_sel)
    y_sorted = y_sel[order]
    X_sorted = X_sel[order]

    y_over_H = y_sorted / (y.max() - y.min())
    species_names = eq_manager.species.names
    return y_over_H, X_sorted, species_names


def extract_thermal_conductivity_profile_from_U(case, U, x_target=None):
    """Extract thermal conductivity profiles (eta_tr, eta_v) along y for a given U."""
    mesh = case["mesh"]
    eq_manager = case["eq_manager"]

    prim = equation_manager_utils.extract_primitives(U, eq_manager)
    if eq_manager.transport_model is None:
        raise ValueError("Transport model is required to extract conductivity.")
    _, eta_t, eta_r, eta_v, _ = (
        eq_manager.transport_model.compute_transport_properties(
            T=prim.T,
            T_v=prim.Tv,
            p=prim.p,
            Y_s=prim.Y_s,
            rho=prim.rho,
        )
    )

    eta_tr = np.asarray(eta_t + eta_r)
    eta_v = np.asarray(eta_v)

    x = mesh.cell_centroids[:, 0]
    y = mesh.cell_centroids[:, 1]

    if x_target is None:
        x_target = 0.5 * (x.min() + x.max())
    dx = np.min(np.diff(np.unique(x))) if len(np.unique(x)) > 1 else 1.0
    mask = np.abs(x - x_target) <= 0.25 * dx

    y_sel = y[mask]
    eta_tr_sel = eta_tr[mask]
    eta_v_sel = eta_v[mask]

    order = np.argsort(y_sel)
    y_over_H = y_sel[order] / (y.max() - y.min())
    return y_over_H, eta_tr_sel[order], eta_v_sel[order]


def extract_boundary_profile_from_U(
    case,
    U,
    x_target=None,
    boundary_tags=(3, 4),
):
    mesh = case["mesh"]
    eq_manager = case["eq_manager"]

    U_L, U_R = equation_manager.compute_face_states(U, mesh, eq_manager)
    prim_R = equation_manager_utils.extract_primitives(U_R, eq_manager)

    face_right = np.asarray(mesh.face_right)
    boundary_mask = face_right < 0
    tags = np.asarray(mesh.boundary_tags)
    tag_mask = np.isin(tags, boundary_tags)
    mask = boundary_mask & tag_mask

    if not np.any(mask):
        return np.array([]), np.array([]), np.array([])

    face_centroids = np.asarray(mesh.face_centroids)
    x = face_centroids[:, 0]
    y = face_centroids[:, 1]

    if x_target is None:
        x_target = 0.5 * (x[mask].min() + x[mask].max())
    unique_x = np.unique(x[mask])
    dx = np.min(np.diff(unique_x)) if len(unique_x) > 1 else 1.0
    mask = mask & (np.abs(x - x_target) <= 0.25 * dx)

    if not np.any(mask):
        return np.array([]), np.array([]), np.array([])

    y_sel = y[mask]
    T_sel = np.asarray(prim_R.T)[mask]
    Tv_sel = np.asarray(prim_R.Tv)[mask]

    order = np.argsort(y_sel)
    y_sorted = y_sel[order]
    T_sorted = T_sel[order]
    Tv_sorted = Tv_sel[order]

    y_over_H = y_sorted / (y.max() - y.min())
    Tb0 = case["Tw_bottom"]
    T_norm = T_sorted / Tb0
    Tv_norm = Tv_sorted / Tb0

    return y_over_H, T_norm, Tv_norm


def load_casseau_tr_tb_reference(csv_path: str | Path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return []

    lines = csv_path.read_text().splitlines()
    if not lines:
        return []

    header = lines[0]
    dataset_names = [name for name in header.split(",") if name]
    data = np.genfromtxt(csv_path, delimiter=",", skip_header=2)
    data = np.atleast_2d(data)

    references = []
    n_pairs = data.shape[1] // 2
    for i in range(n_pairs):
        label = dataset_names[i] if i < len(dataset_names) else f"dataset_{i+1}"
        x = data[:, i * 2]
        y = data[:, i * 2 + 1]
        mask = np.isfinite(x) & np.isfinite(y)
        references.append(
            {
                "label": label,
                "Ttr_over_Tb0": x[mask],
                "y_over_H": y[mask],
            }
        )

    return references
