"""Run a 2D axisymmetric blunt-cone case (Casseau-like defaults)."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from compressible_core import chemistry_utils, energy_models, transport_casseau
from compressible_core import transport as transport_core
from compressible_core import chemistry

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
    parser = argparse.ArgumentParser(description="Run blunt-cone 2D axisymmetric case")
    parser.add_argument("--mesh", required=True, help="Path to Gmsh .msh file")
    parser.add_argument("--t-final", type=float, default=5e-5)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--dt-mode", choices=["fixed", "cfl"], default="cfl")
    parser.add_argument("--dt", type=float, default=1e-9)
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
        "--history-device",
        choices=["device", "cpu"],
        default="device",
        help="Store history on accelerator or host memory (cpu avoids GPU OOM).",
    )

    args = parser.parse_args()

    # Casseau Mach 11.3 blunted cone defaults (non-reacting N2)
    U_inf = 2764.5
    p_inf = 21.9139
    rho_inf = 5.113e-4
    T_inf = 144.4
    Tw = 297.2

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
                "u": 0.0,
                "v": U_inf,
                "T": T_inf,
                "Tv": T_inf,
                "Y": [1.0],
            },
            args.tag_outflow: {"type": "outflow"},
            args.tag_wall: {"type": "wall", "Tw": Tw},
            args.tag_axis: {"type": "axisymmetric"},
        }
    )

    mesh = mesh_gmsh.read_gmsh(args.mesh)

    eq_manager = equation_manager.build_equation_manager(
        mesh,
        species=species,
        collision_integrals=collision_integrals,
        reactions=reactions,
        numerics_config=numerics_config,
        boundary_config=boundary_config,
        transport_model=equation_manager_types.TransportModelConfig(
            model=args.transport
        ),
        casseau_transport=casseau_transport,
    )

    # Initialize freestream everywhere
    n_cells = mesh.cell_areas.shape[0]
    Y = jnp.ones((n_cells, len(species_names)))
    rho = jnp.full((n_cells,), rho_inf)
    u = jnp.zeros((n_cells,))
    v = jnp.full((n_cells,), U_inf)
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

    U_hist, t_hist = equation_manager.run_scan(
        U_init,
        mesh,
        eq_manager,
        t_final=args.t_final,
        save_interval=args.save_interval,
    )

    out_dir = Path("experiments/blunt_cone_2d")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "solution.npz"
    np.savez(out_path, U=U_hist, t=t_hist)
    print(f"Saved solution to {out_path}")


if __name__ == "__main__":
    main()
