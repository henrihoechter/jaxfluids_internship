"""Profile 1D vs 2D shock tube runs to compare runtime.

Example:
  python3 experiments/profile_shock_tube.py --profile --warmup 1
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from compressible_core import chemistry_utils, constants, energy_models, transport, transport_casseau, chemistry
from compressible_1d import equation_manager as eq1d_manager
from compressible_1d import equation_manager_types as eq1d_types
from compressible_1d import equation_manager_utils as eq1d_utils
from compressible_1d import numerics_types as numerics1d_types
from compressible_2d import mesh_gmsh
from compressible_2d import equation_manager as eq2d_manager
from compressible_2d import equation_manager_types as eq2d_types
from compressible_2d import equation_manager_utils as eq2d_utils
from compressible_2d import numerics_types as numerics2d_types


jax.config.update("jax_enable_x64", False)
jax.config.update("jax_disable_jit", False)


def make_piecewise_fn(
    x0: float, left: jnp.ndarray | float, right: jnp.ndarray | float, width: float = 0.0
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    left = jnp.asarray(left)
    right = jnp.asarray(right)

    def fn(x: jnp.ndarray) -> jnp.ndarray:
        cond = x < x0
        if left.ndim > 0 or right.ndim > 0:
            cond = cond[..., None]
        if width <= 0.0:
            return jnp.where(cond, left, right)
        blend = 0.5 * (1.0 + jnp.tanh((x - x0) / width))
        if left.ndim > 0 or right.ndim > 0:
            blend = blend[..., None]
        return left * (1.0 - blend) + right * blend

    return fn


def build_grid_1d(n_cells: int, length: float) -> tuple[jnp.ndarray, float]:
    dx = length / n_cells
    x = jnp.linspace(0.5 * dx, length - 0.5 * dx, n_cells)
    return x, dx


def build_structured_mesh_2d(
    nx: int,
    ny: int,
    length_x: float,
    length_y: float,
    origin: tuple[float, float] = (0.0, 0.0),
    boundary_tags: tuple[int, int, int, int] = (1, 2, 3, 4),
):
    ox, oy = origin
    x_nodes = np.linspace(ox, ox + length_x, nx + 1)
    y_nodes = np.linspace(oy, oy + length_y, ny + 1)

    nodes = np.zeros(((nx + 1) * (ny + 1), 2))
    for j, y in enumerate(y_nodes):
        for i, x in enumerate(x_nodes):
            nodes[j * (nx + 1) + i] = [x, y]

    def node_index(i: int, j: int) -> int:
        return j * (nx + 1) + i

    cells = []
    for j in range(ny):
        for i in range(nx):
            n00 = node_index(i, j)
            n10 = node_index(i + 1, j)
            n11 = node_index(i + 1, j + 1)
            n01 = node_index(i, j + 1)
            cells.append([n00, n10, n11, n01])

    left_tag, right_tag, bottom_tag, top_tag = boundary_tags
    boundary_edges = []
    for j in range(ny):
        boundary_edges.append((node_index(0, j), node_index(0, j + 1), left_tag))
        boundary_edges.append((node_index(nx, j), node_index(nx, j + 1), right_tag))
    for i in range(nx):
        boundary_edges.append((node_index(i, 0), node_index(i + 1, 0), bottom_tag))
        boundary_edges.append((node_index(i, ny), node_index(i + 1, ny), top_tag))

    mesh = mesh_gmsh.Mesh2D.from_cells(nodes, cells, boundary_edges)
    dx = length_x / nx
    dy = length_y / ny
    x_centers = 0.5 * (x_nodes[:-1] + x_nodes[1:])
    y_centers = 0.5 * (y_nodes[:-1] + y_nodes[1:])
    return mesh, dx, dy, x_centers, y_centers


def load_species_table(
    species_names: Sequence[str],
    general_data_path: str,
    energy_data_path: str,
) -> chemistry_utils.SpeciesTable:
    energy_cfg = energy_models.EnergyModelConfig(
        model="bird",
        include_electronic=False,
        data_path=str(energy_data_path),
    )

    return chemistry_utils.load_species_table(
        species_names=species_names,
        general_data_path=str(general_data_path),
        energy_model_config=energy_cfg,
    )


def compute_temperature_from_primitives(
    p: jnp.ndarray, rho: jnp.ndarray, Y: jnp.ndarray, M_s: jnp.ndarray
) -> jnp.ndarray:
    M_mix = jnp.sum(Y * M_s[None, :], axis=1)
    return p * M_mix / (rho * constants.R_universal)


def compute_pressure_from_primitives(
    T: jnp.ndarray, rho: jnp.ndarray, Y: jnp.ndarray, M_s: jnp.ndarray
) -> jnp.ndarray:
    M_mix = jnp.sum(Y * M_s[None, :], axis=1)
    return rho * constants.R_universal * T / M_mix


def compute_density_from_primitives(
    T: jnp.ndarray, p: jnp.ndarray, Y: jnp.ndarray, M_s: jnp.ndarray
) -> jnp.ndarray:
    M_mix = jnp.sum(Y * M_s[None, :], axis=1)
    return p * M_mix / (constants.R_universal * T)


def normalize_mole_fractions(Y: jnp.ndarray) -> jnp.ndarray:
    Y = jnp.asarray(Y)
    if Y.ndim == 1:
        Y = Y[:, None]
    Y_sum = jnp.sum(Y, axis=1, keepdims=True)
    return Y / jnp.clip(Y_sum, 1e-14, None)


def build_initial_state_1d(
    x: jnp.ndarray,
    species_table,
    rho_fn: Callable[[jnp.ndarray], jnp.ndarray],
    u_fn: Callable[[jnp.ndarray], jnp.ndarray],
    Y_fn: Callable[[jnp.ndarray], jnp.ndarray],
    p_fn: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    T_fn: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    Tv_fn: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    if p_fn is None and T_fn is None:
        raise ValueError("Provide either p_fn(x) or T_fn(x).")

    rho = rho_fn(x)
    u = u_fn(x)
    Y = normalize_mole_fractions(jnp.asarray(Y_fn(x)))

    if T_fn is None:
        p = p_fn(x)
        T = compute_temperature_from_primitives(p, rho, Y, species_table.molar_masses)
    else:
        T = T_fn(x)
        p = compute_pressure_from_primitives(T, rho, Y, species_table.molar_masses)

    if Tv_fn is None:
        Tv = T
    else:
        Tv = Tv_fn(x)

    return rho, u, T, Tv, Y


def _block_until_ready(tree) -> None:
    def _block(x):
        if hasattr(x, "block_until_ready"):
            x.block_until_ready()
        return x

    jax.tree_util.tree_map(_block, tree)


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def setup_common_physics(args, repo_root: Path):
    species_names = ("N2", "N")
    species_table = load_species_table(
        species_names,
        general_data_path=str(repo_root / "data/species.json"),
        energy_data_path=str(repo_root / "data/air_5_bird_energy.json"),
    )

    reactions = None
    if args.use_chemistry:
        reactions_path = repo_root / "data/park_1990_reactions.json"
        chemistry_config = chemistry.ChemistryModelConfig(
            model="park",
            park_vibrational_source="preferential_constant",
            qp_constant=0.3,
            park_alpha=0.5,
        )
        reactions = chemistry_utils.load_reactions_from_json(
            json_path=str(reactions_path),
            species_table=species_table,
            chemistry_model_config=chemistry_config,
        )

    collision_integrals = None
    casseau_transport = None
    transport_config_1d = None
    transport_config_2d = None
    if args.use_transport:
        transport_config_1d = eq1d_types.TransportModelConfig(
            model=args.transport_model,
            include_diffusion=args.include_diffusion,
        )
        transport_config_2d = eq2d_types.TransportModelConfig(
            model=args.transport_model,
            include_diffusion=args.include_diffusion,
        )

        if args.transport_model == "gnoffo":
            collision_integrals = transport.create_collision_integral_table_from_json(
                repo_root / "data/collision_integrals_tp2867.json"
            )
        elif args.transport_model == "casseau":
            casseau_transport = transport_casseau.load_casseau_transport_table(
                repo_root / "data/air_5_casseau_transport.json",
                species_names,
            )
            if args.include_diffusion:
                collision_integrals = transport.create_collision_integral_table_from_json(
                    repo_root / "data/collision_integrals_tp2867.json"
                )

    return (
        species_table,
        reactions,
        collision_integrals,
        casseau_transport,
        transport_config_1d,
        transport_config_2d,
        species_names,
    )


def make_shock_tube_initializers(
    species_table,
    x0: float,
    transition_width: float,
    T_L: float,
    p_L: float,
    u_L: float,
    Tv_L: float,
    Y_L: jnp.ndarray,
    T_R: float,
    p_R: float,
    u_R: float,
    Tv_R: float,
    Y_R: jnp.ndarray,
):
    rho_L = compute_density_from_primitives(
        jnp.array([T_L]), jnp.array([p_L]), Y_L[None, :], species_table.molar_masses
    )[0]
    rho_R = compute_density_from_primitives(
        jnp.array([T_R]), jnp.array([p_R]), Y_R[None, :], species_table.molar_masses
    )[0]

    rho_fn = make_piecewise_fn(x0, rho_L, rho_R, width=transition_width)
    u_fn = make_piecewise_fn(x0, u_L, u_R, width=transition_width)
    Y_fn = make_piecewise_fn(x0, Y_L, Y_R, width=transition_width)
    T_fn = make_piecewise_fn(x0, T_L, T_R, width=transition_width)
    Tv_fn = make_piecewise_fn(x0, Tv_L, Tv_R, width=transition_width)
    return rho_fn, u_fn, Y_fn, T_fn, Tv_fn


def build_2d_case(args, species_table, collision_integrals, reactions, transport_config_2d, casseau_transport):
    length_x = args.length_x
    length_y = args.length_y
    x0 = 0.5 * length_x if args.x0 is None else args.x0
    left_tag, right_tag, bottom_tag, top_tag = 1, 2, 3, 4

    mesh, _, _, _, _ = build_structured_mesh_2d(
        args.nx, args.ny, length_x, length_y, boundary_tags=(left_tag, right_tag, bottom_tag, top_tag)
    )

    numerics_config_2d = numerics2d_types.NumericsConfig2D(
        dt=args.dt_2d,
        cfl=args.cfl_2d,
        dt_mode="cfl" if args.use_cfl_2d else "fixed",
        integrator_scheme="rk2",
        spatial_scheme="first_order",
        flux_scheme="hllc",
        axisymmetric=False,
        clipping=numerics2d_types.ClippingConfig2D(),
    )

    boundary_config_2d = eq2d_types.BoundaryConditionConfig2D(
        tag_to_bc={
            left_tag: {"type": "outflow"},
            right_tag: {"type": "outflow"},
            bottom_tag: {"type": "outflow"},
            top_tag: {"type": "outflow"},
        }
    )

    eq_manager_2d = eq2d_manager.build_equation_manager(
        mesh,
        species=species_table,
        collision_integrals=collision_integrals,
        reactions=reactions,
        numerics_config=numerics_config_2d,
        boundary_config=boundary_config_2d,
        transport_model=transport_config_2d or eq2d_types.TransportModelConfig(),
        casseau_transport=casseau_transport,
    )

    rho_fn, u_fn, Y_fn, T_fn, Tv_fn = make_shock_tube_initializers(
        species_table=species_table,
        x0=x0,
        transition_width=args.transition_width,
        T_L=args.T_L,
        p_L=args.p_L,
        u_L=args.u_L,
        Tv_L=args.Tv_L,
        Y_L=args.Y_L,
        T_R=args.T_R,
        p_R=args.p_R,
        u_R=args.u_R,
        Tv_R=args.Tv_R,
        Y_R=args.Y_R,
    )

    x_cells = jnp.asarray(mesh.cell_centroids[:, 0])
    rho, u, T, Tv, Y = build_initial_state_1d(
        x_cells, species_table, rho_fn, u_fn, Y_fn, p_fn=None, T_fn=T_fn, Tv_fn=Tv_fn
    )
    v = jnp.zeros_like(u)

    U_init_2d = eq2d_utils.compute_U_from_primitives(
        Y_s=Y,
        rho=rho,
        u=u,
        v=v,
        T_tr=T,
        T_V=Tv,
        equation_manager=eq_manager_2d,
    )

    return mesh, eq_manager_2d, U_init_2d


def build_1d_case(args, species_table, collision_integrals, reactions, transport_config_1d, casseau_transport):
    length_x = args.length_x
    x0 = 0.5 * length_x if args.x0 is None else args.x0

    x_1d, dx_1d = build_grid_1d(args.n_cells_1d, length_x)

    numerics_config_1d = numerics1d_types.NumericsConfig(
        dt=args.dt_1d,
        dx=dx_1d,
        integrator_scheme="rk2",
        spatial_scheme="muscl",
        flux_scheme="hllc",
        n_halo_cells=1,
        clipping=numerics1d_types.ClippingConfig(),
    )

    eq_manager_1d = eq1d_types.EquationManager(
        species=species_table,
        collision_integrals=collision_integrals,
        reactions=reactions,
        numerics_config=numerics_config_1d,
        boundary_condition=args.boundary_condition_1d,
        transport_model=transport_config_1d or eq1d_types.TransportModelConfig(),
        casseau_transport=casseau_transport,
    )

    rho_fn, u_fn, Y_fn, T_fn, Tv_fn = make_shock_tube_initializers(
        species_table=species_table,
        x0=x0,
        transition_width=args.transition_width,
        T_L=args.T_L,
        p_L=args.p_L,
        u_L=args.u_L,
        Tv_L=args.Tv_L,
        Y_L=args.Y_L,
        T_R=args.T_R,
        p_R=args.p_R,
        u_R=args.u_R,
        Tv_R=args.Tv_R,
        Y_R=args.Y_R,
    )

    rho, u, T, Tv, Y = build_initial_state_1d(
        x_1d, species_table, rho_fn, u_fn, Y_fn, p_fn=None, T_fn=T_fn, Tv_fn=Tv_fn
    )

    U_init_1d = eq1d_utils.compute_U_from_primitives(
        Y_s=Y, rho=rho, u=u, T_tr=T, T_V=Tv, equation_manager=eq_manager_1d
    )

    return eq_manager_1d, U_init_1d


def time_run(label: str, run_fn: Callable[[], tuple], warmup: int, trace_dir: Path | None):
    for _ in range(max(warmup, 0)):
        out = run_fn()
        _block_until_ready(out)

    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)
        jax.profiler.start_trace(str(trace_dir))

    start = time.perf_counter()
    out = run_fn()
    _block_until_ready(out)
    elapsed = time.perf_counter() - start

    if trace_dir is not None:
        jax.profiler.stop_trace()

    return elapsed, out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile 1D vs 2D shock tube runs")
    parser.add_argument("--run", choices=["both", "1d", "2d"], default="both")
    parser.add_argument("--profile", action="store_true", help="Write JAX trace to profiler_data/")
    parser.add_argument("--trace-dir", default="profiler_data")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs before timing (0 to include compile)")

    parser.add_argument("--use-chemistry", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-transport", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--transport-model", choices=["gnoffo", "casseau"], default="gnoffo")
    parser.add_argument("--include-diffusion", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--length-x", type=float, default=1.0)
    parser.add_argument("--length-y", type=float, default=0.1)
    parser.add_argument("--nx", type=int, default=200)
    parser.add_argument("--ny", type=int, default=20)
    parser.add_argument("--x0", type=float, default=None)

    parser.add_argument("--n-cells-1d", type=int, default=1000)
    parser.add_argument("--dt-1d", type=float, default=1.0e-8)
    parser.add_argument("--t-final-1d", type=float, default=1.6e-7)
    parser.add_argument("--save-interval-1d", type=int, default=1)
    parser.add_argument(
        "--boundary-condition-1d",
        choices=["transmissive", "reflective", "periodic"],
        default="transmissive",
    )

    parser.add_argument("--dt-2d", type=float, default=1.0e-8)
    parser.add_argument("--use-cfl-2d", action="store_true", default=False)
    parser.add_argument("--cfl-2d", type=float, default=0.4)
    parser.add_argument("--t-final-2d", type=float, default=1.6e-7)
    parser.add_argument("--save-interval-2d", type=int, default=5)

    parser.add_argument("--transition-width", type=float, default=0.0)
    parser.add_argument("--T-L", type=float, default=12000.0)
    parser.add_argument("--p-L", type=float, default=2.0e6)
    parser.add_argument("--u-L", type=float, default=0.0)
    parser.add_argument("--Tv-L", type=float, default=12000.0)
    parser.add_argument("--T-R", type=float, default=1000.0)
    parser.add_argument("--p-R", type=float, default=1.0e5)
    parser.add_argument("--u-R", type=float, default=0.0)
    parser.add_argument("--Tv-R", type=float, default=1000.0)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    args.Y_L = jnp.array([1.0, 0.0])
    args.Y_R = jnp.array([1.0, 0.0])

    repo_root = Path(__file__).resolve().parents[1]
    (
        species_table,
        reactions,
        collision_integrals,
        casseau_transport,
        transport_config_1d,
        transport_config_2d,
        _species_names,
    ) = setup_common_physics(args, repo_root)

    timings = {}

    if args.run in ("both", "2d"):
        mesh, eq_manager_2d, U_init_2d = build_2d_case(
            args,
            species_table,
            collision_integrals,
            reactions,
            transport_config_2d,
            casseau_transport,
        )

        def run_2d():
            return eq2d_manager.run_scan(
                U_init_2d,
                mesh,
                eq_manager_2d,
                t_final=args.t_final_2d,
                save_interval=args.save_interval_2d,
            )

        trace_dir = None
        if args.profile:
            trace_dir = Path(args.trace_dir) / f"shock_tube_2d_{_timestamp()}"

        elapsed_2d, out_2d = time_run("2d", run_2d, args.warmup, trace_dir)
        U_hist_2d, _ = out_2d
        n_steps_2d = (U_hist_2d.shape[0] - 1) * args.save_interval_2d
        timings["2d"] = elapsed_2d
        print(
            f"2D: {elapsed_2d:.3f}s total, steps={n_steps_2d}, cells={U_hist_2d.shape[1]}, "
            f"{elapsed_2d / max(n_steps_2d, 1):.3e}s/step"
        )

    if args.run in ("both", "1d"):
        eq_manager_1d, U_init_1d = build_1d_case(
            args,
            species_table,
            collision_integrals,
            reactions,
            transport_config_1d,
            casseau_transport,
        )

        def run_1d():
            return eq1d_manager.run(
                U_init=U_init_1d,
                equation_manager=eq_manager_1d,
                t_final=args.t_final_1d,
                save_interval=args.save_interval_1d,
            )

        trace_dir = None
        if args.profile:
            trace_dir = Path(args.trace_dir) / f"shock_tube_1d_{_timestamp()}"

        elapsed_1d, out_1d = time_run("1d", run_1d, args.warmup, trace_dir)
        U_hist_1d, _ = out_1d
        n_steps_1d = (U_hist_1d.shape[0] - 1) * args.save_interval_1d
        timings["1d"] = elapsed_1d
        print(
            f"1D: {elapsed_1d:.3f}s total, steps={n_steps_1d}, cells={U_hist_1d.shape[1]}, "
            f"{elapsed_1d / max(n_steps_1d, 1):.3e}s/step"
        )

    if len(timings) == 2:
        slower = "2D" if timings["2d"] > timings["1d"] else "1D"
        ratio = timings["2d"] / timings["1d"] if timings["1d"] else float("inf")
        if slower == "1D":
            ratio = timings["1d"] / timings["2d"] if timings["2d"] else float("inf")
        print(f"Slower overall: {slower} ({ratio:.2f}x)")


if __name__ == "__main__":
    main()
