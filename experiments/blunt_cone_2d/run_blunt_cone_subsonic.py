"""Run a 2D blunt-cone case with subsonic, thermally relaxed conditions (Ma ~ 0.3, sea-level N2)."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from compressible_core import chemistry_utils, energy_models
from compressible_core import transport_model_casseau_utils as transport_casseau
from compressible_core import transport_model_gnoffo_utils as transport_core
from compressible_core import chemistry
from compressible_core.transport_models_types import TransportModelConfig
from compressible_core.transport_models_utils import build_transport_model_from_config

from compressible_2d import mesh_gmsh
from compressible_2d import equation_manager
from compressible_2d import equation_manager_types
from compressible_2d import equation_manager_utils
from compressible_2d import numerics_types


def load_species_table(species_names: tuple[str, ...]) -> chemistry_utils.SpeciesTable:
    repo_root = Path(__file__).resolve().parents[2]
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run blunt-cone 2D subsonic relaxed case"
    )
    parser.add_argument("--mesh", required=True, help="Path to Gmsh .msh file")
    parser.add_argument("--t-final", type=float, default=1e-3)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--dt-mode", choices=["fixed", "cfl"], default="fixed")
    parser.add_argument("--dt", type=float, default=1e-7)
    parser.add_argument("--cfl", type=float, default=0.4)
    parser.add_argument("--transport", choices=["gnoffo", "casseau"], default="casseau")
    parser.add_argument("--reactions", default=None, help="Path to reactions JSON")
    parser.add_argument(
        "--collision-integrals", default=None, help="Path to collision integrals JSON"
    )
    parser.add_argument("--tag-inflow", type=int, default=1)
    parser.add_argument("--tag-outflow", type=int, default=2)
    parser.add_argument("--tag-wall", type=int, default=3)
    parser.add_argument("--tag-axis", type=int, default=4)
    parser.add_argument(
        "--plot-mesh",
        action="store_true",
        help="Plot the mesh colored by boundary tag and exit.",
    )
    parser.add_argument(
        "--debug-nan",
        action="store_true",
        help=(
            "Run a single step in eager mode with jax_debug_nans=True to locate "
            "the operation that first produces NaN. Exits after the debug step."
        ),
    )
    parser.add_argument(
        "--wedge-plane-tag",
        type=int,
        default=4,
        help="Physical tag of the 2D cross-section surface in the 3D wedge mesh (default: 4).",
    )
    parser.add_argument(
        "--axis-remap-from",
        type=int,
        default=7,
        help=(
            "Axis boundary tag as stored in the .msh file; remapped to --tag-axis "
            "(default: 7)."
        ),
    )

    args = parser.parse_args()

    # Subsonic relaxed defaults: Ma ~ 0.3, sea-level N2
    # R_N2 = 8314.46 / 28.014 = 296.8 J/(kg*K)
    # a = sqrt(1.4 * 296.8 * 300) = 352.6 m/s  =>  U_inf = 0.3 * a = 105.8 m/s
    T_inf = 300.0
    p_inf = 101325.0
    rho_inf = p_inf / (296.8 * T_inf)  # ~1.138 kg/m^3
    U_inf = 705.8  # Ma ~ 0.3
    Tw = T_inf  # isothermal wall at freestream temperature

    species_names = ("N2",)
    species = load_species_table(species_names)

    reactions = None
    if args.reactions:
        reactions = chemistry_utils.load_reactions_from_json(
            args.reactions, species, chemistry.ChemistryModelConfig()
        )

    collision_integrals = None
    if args.collision_integrals:
        collision_integrals = transport_core.create_collision_integral_table_from_json(
            args.collision_integrals
        )

    casseau_transport = None
    if args.transport == "casseau":
        repo_root = Path(__file__).resolve().parents[2]
        data_dir = repo_root / "data"
        casseau_transport = transport_casseau.load_casseau_transport_table(
            data_dir / "air_5_casseau_transport.json", species_names
        )

    numerics_config = numerics_types.NumericsConfig2D(
        dt=args.dt,
        cfl=args.cfl,
        dt_mode=args.dt_mode,
        integrator_scheme="rk2",
        spatial_scheme="first_order",
        flux_scheme="hllc",
        axisymmetric=True,
        clipping=numerics_types.ClippingConfig2D(),
    )

    boundary_config = equation_manager_types.BoundaryConditionConfig2D(
        tag_to_bc={
            args.tag_inflow: {
                "type": "inflow",
                "rho": rho_inf,
                "u": U_inf,
                "v": 0.0,
                "T": T_inf,
                "Tv": T_inf,
                "Y": [1.0],
            },
            args.tag_outflow: {"type": "outflow"},
            args.tag_wall: {"type": "wall_euler", "Tw": Tw},
            args.tag_axis: {"type": "axisymmetric"},
        }
    )

    mesh = mesh_gmsh.read_gmsh_v2_wedge_plane(
        args.mesh,
        wedge_plane_tag=args.wedge_plane_tag,
        remap_tags={args.axis_remap_from: args.tag_axis},
        axis_tag=args.tag_axis,
    )

    # --- mesh diagnostics ---
    n_faces = mesh.boundary_tags.shape[0]
    boundary_mask = mesh.face_right == -1
    n_boundary = int(np.sum(boundary_mask))
    n_untagged = int(np.sum(boundary_mask & (mesh.boundary_tags == -1)))
    print(
        f"Mesh: {mesh.cell_areas.shape[0]} cells, {n_faces} faces, {n_boundary} boundary faces"
    )
    print(
        f"  Cell area range:   [{mesh.cell_areas.min():.3e}, {mesh.cell_areas.max():.3e}]"
    )
    print(f"  cell_r (col 0):    [{mesh.cell_r.min():.3e}, {mesh.cell_r.max():.3e}]")
    print(
        f"  centroid col 1:    [{mesh.cell_centroids[:, 1].min():.3e}, {mesh.cell_centroids[:, 1].max():.3e}]"
    )
    from collections import Counter

    tag_counts = Counter(
        int(t) for t, bnd in zip(mesh.boundary_tags, boundary_mask) if bnd
    )
    print(f"  Boundary tag counts: {dict(sorted(tag_counts.items()))}")
    if n_untagged:
        print(
            f"  WARNING: {n_untagged} boundary faces have no tag (tag=-1) -- these will be ignored by BCs"
        )
    else:
        print("  All boundary faces are tagged.")
    expected_tags = {args.tag_inflow, args.tag_outflow, args.tag_wall, args.tag_axis}
    missing = expected_tags - set(tag_counts.keys())
    if missing:
        print(f"  WARNING: expected tags not found in boundary faces: {missing}")
    # --- end diagnostics ---

    if args.plot_mesh:
        import matplotlib.pyplot as plt
        import matplotlib.collections as mc

        _, ax = plt.subplots(figsize=(10, 8))

        cell_edges = []
        for cell in mesh.cells:
            pts = mesh.nodes[cell]
            for k in range(len(cell)):
                cell_edges.append([pts[k], pts[(k + 1) % len(cell)]])
        ax.add_collection(
            mc.LineCollection(cell_edges, colors="lightgrey", linewidths=0.3)
        )

        unique_tags = sorted(
            set(int(t) for t in mesh.boundary_tags[mesh.face_right == -1])
        )
        cmap = plt.get_cmap("tab10")
        for i, tag in enumerate(unique_tags):
            mask = (mesh.face_right == -1) & (mesh.boundary_tags == tag)
            segs = [
                [mesh.nodes[mesh.face_nodes[fi, 0]], mesh.nodes[mesh.face_nodes[fi, 1]]]
                for fi in np.where(mask)[0]
            ]
            color = "red" if tag == -1 else cmap(i / max(len(unique_tags), 1))
            label = f"tag {tag}" if tag != -1 else "untagged (tag=-1)"
            ax.add_collection(
                mc.LineCollection(segs, colors=[color], linewidths=1.5, label=label)
            )

        ax.autoscale()
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Mesh boundary tags")
        ax.legend(loc="best")
        plt.tight_layout()
        out_path = Path("experiments/blunt_cone_2d/mesh_plot.png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150)
        print(f"Saved mesh plot to {out_path}")
        plt.show()
        return

    eq_manager = equation_manager.build_equation_manager(
        mesh,
        species=species,
        collision_integrals=collision_integrals,
        reactions=reactions,
        numerics_config=numerics_config,
        boundary_config=boundary_config,
        transport_model=build_transport_model_from_config(
            TransportModelConfig(model=args.transport),
            species_table=species,
            collision_integrals=collision_integrals,
            casseau_transport=casseau_transport,
        ),
        casseau_transport=casseau_transport,
    )

    # Initialize freestream everywhere (thermally relaxed: Tv = T)
    n_cells = mesh.cell_areas.shape[0]
    Y = jnp.ones((n_cells, len(species_names)))
    rho = jnp.full((n_cells,), rho_inf)
    u = jnp.full((n_cells,), U_inf)
    v = jnp.zeros((n_cells,))
    T = jnp.full((n_cells,), T_inf)
    Tv = jnp.full((n_cells,), T_inf)

    U_init = equation_manager_utils.compute_U_from_primitives(
        Y_s=Y,
        rho=rho,
        u=u,
        v=v,
        T_tr=T,
        T_V=Tv,
        equation_manager=eq_manager,
    )

    if args.debug_nan:
        print("NaN debug: running one step with jax_debug_nans=True ...")
        jax.config.update("jax_debug_nans", True)
        with jax.disable_jit():
            U_debug = equation_manager.advance_one_step(
                U_init, mesh, eq_manager, args.dt
            )
        U_debug.block_until_ready()
        print("Step completed without NaN — initial condition looks clean.")
        jax.config.update("jax_debug_nans", False)
        return

    U_hist, t_hist = equation_manager.run_scan(
        U_init,
        mesh,
        eq_manager,
        args.t_final,
        args.save_interval,
    )

    out_dir = Path("experiments/blunt_cone_2d")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "solution_subsonic.npz"
    np.savez(out_path, U=U_hist, t=t_hist)
    print(f"Saved solution to {out_path}")


if __name__ == "__main__":
    main()
