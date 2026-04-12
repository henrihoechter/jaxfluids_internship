"""Minimal Toro-style shock tube driver using the unified compressible package."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import compressible.chemistry_utils as chemistry_utils
import compressible.constants as constants
from compressible.boundary_conditions_utils import build_boundary_arrays_1d
from compressible.equation_manager import run_scan
from compressible.equation_manager_types import EquationManager
from compressible.energy_models_types import EnergyModelConfig
from compressible.mesh import Mesh
from compressible.numerics_types import NumericsConfig
from compressible.state import compute_U_from_primitives, extract_primitives_from_U


jax.config.update("jax_enable_x64", False)
jax.config.update("jax_disable_jit", False)


def load_species_table():
    repo_root = Path(__file__).resolve().parents[2]
    return chemistry_utils.load_species_table(
        species_names=("N2",),
        general_data_path=str(repo_root / "data" / "species.json"),
        energy_model_config=EnergyModelConfig(
            model="bird",
            include_electronic=False,
            data_path=str(repo_root / "data" / "air_5_bird_energy.json"),
        ),
    )


def compute_temperature_from_primitives(p, rho, y, molar_masses):
    m_mix = jnp.sum(y * molar_masses[None, :], axis=1)
    return p * m_mix / (rho * constants.R_universal)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a unified-package Toro shock tube.")
    parser.add_argument("--n-cells", type=int, default=400)
    parser.add_argument("--length", type=float, default=1.0)
    parser.add_argument("--t-final", type=float, default=2.5e-4)
    parser.add_argument("--dt", type=float, default=1e-7)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--flux-scheme", choices=["hllc", "exact_riemann", "lax_friedrichs"], default="hllc")
    parser.add_argument("--spatial-scheme", choices=["first_order", "muscl"], default="first_order")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/shock_tube_1d_toro/solution_unified.npz"),
    )
    args = parser.parse_args()

    species = load_species_table()

    x_nodes = jnp.linspace(0.0, args.length, args.n_cells + 1)
    x_centers = 0.5 * (x_nodes[:-1] + x_nodes[1:])
    left = x_centers < 0.5 * args.length

    rho = jnp.where(left, 1.0, 0.125)
    u = jnp.zeros((args.n_cells,))
    p = jnp.where(left, 1.0e5, 1.0e4)
    y = jnp.ones((args.n_cells, species.n_species))
    t = compute_temperature_from_primitives(p, rho, y, species.molar_masses)
    tv = t

    mesh = Mesh.from_1d_grid(x_nodes)
    boundary_arrays = build_boundary_arrays_1d(
        mesh,
        bc_left="outflow",
        bc_right="outflow",
        n_species=species.n_species,
    )
    equation_manager = EquationManager(
        species=species,
        reactions=None,
        numerics_config=NumericsConfig(
            dt=args.dt,
            dt_mode="fixed",
            integrator_scheme="rk2",
            spatial_scheme=args.spatial_scheme,
            flux_scheme=args.flux_scheme,
        ),
        boundary_arrays=boundary_arrays,
    )

    u_init = compute_U_from_primitives(
        Y_s=y,
        rho=rho,
        u=u,
        v=jnp.zeros_like(u),
        T_tr=t,
        T_V=tv,
        equation_manager=equation_manager,
    )

    u_hist, t_hist = run_scan(
        u_init,
        mesh,
        equation_manager,
        t_final=args.t_final,
        save_interval=args.save_interval,
    )
    prim_final = extract_primitives_from_U(u_hist[-1], equation_manager)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        x=np.asarray(x_centers),
        t_history=np.asarray(t_hist),
        u_history=np.asarray(u_hist),
        rho=np.asarray(prim_final.rho),
        u=np.asarray(prim_final.u),
        p=np.asarray(prim_final.p),
        t=np.asarray(prim_final.T),
        tv=np.asarray(prim_final.Tv),
    )
    print(
        f"Saved unified shock-tube solution to {args.output} "
        f"({args.n_cells} cells, flux={args.flux_scheme}, spatial={args.spatial_scheme})."
    )


if __name__ == "__main__":
    main()
