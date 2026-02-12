"""Shock tube setup with functional initial conditions.

This script builds a 1D shock tube using user-defined functions for each primitive
variable (e.g., rho(x), u(x), p(x), Y_s(x)). It prepares U_init for the
compressible_1d equation_manager and optionally runs the solver + plots.

Notes:
- Primitives here are (Y_s, rho, u, T, Tv). You may specify either p(x) or T(x).
- If Tv(x) is not provided, Tv := T.
- Chemistry and transport are disabled by default (inviscid, frozen chemistry).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable, Sequence

import jax
import jax.numpy as jnp

from compressible_1d import (
    chemistry_utils,
    constants,
    energy_models,
    equation_manager,
    equation_manager_types,
    equation_manager_utils,
    numerics_types,
    solver,
)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

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


def build_grid(n_cells: int, length: float) -> tuple[jnp.ndarray, float]:
    dx = length / n_cells
    x = jnp.linspace(0.5 * dx, length - 0.5 * dx, n_cells)
    return x, dx


def load_species_table(species_names: Sequence[str]) -> chemistry_utils.SpeciesTable:
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


def build_equation_manager(
    species_table,
    dx: float,
    dt: float,
    boundary_condition: str = "transmissive",
) -> equation_manager_types.EquationManager:
    numerics_config = numerics_types.NumericsConfig(
        dt=dt,
        dx=dx,
        integrator_scheme="rk2",
        spatial_scheme="first_order",
        flux_scheme="hllc",
        n_halo_cells=1,
        clipping=numerics_types.ClippingConfig(),
    )

    return equation_manager_types.EquationManager(
        species=species_table,
        collision_integrals=None,
        reactions=None,
        numerics_config=numerics_config,
        boundary_condition=boundary_condition,
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


def normalize_mole_fractions(Y: jnp.ndarray) -> jnp.ndarray:
    Y = jnp.asarray(Y)
    if Y.ndim == 1:
        Y = Y[:, None]
    Y_sum = jnp.sum(Y, axis=1, keepdims=True)
    return Y / jnp.clip(Y_sum, 1e-14, None)


def build_initial_state(
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


def compute_cfl_dt(
    U_init: jnp.ndarray,
    eq_manager: equation_manager_types.EquationManager,
    cfl: float = 0.4,
) -> float:
    Y, rho, T, Tv, p = equation_manager_utils.extract_primitives_from_U(
        U_init, eq_manager
    )
    n_species = eq_manager.species.n_species
    u = U_init[:, n_species] / rho
    a = solver.compute_speed_of_sound(rho, p, Y, T, Tv, eq_manager)
    return float(cfl * eq_manager.numerics_config.dx / jnp.max(jnp.abs(u) + a))


def plot_primitives_slices(
    x: jnp.ndarray,
    primitive_slices: list[dict],
    title: str = "Shock Tube Profiles",
):
    if not HAS_PLOTLY:
        print("plotly not available; skipping plots.")
        return

    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        subplot_titles=["rho", "u", "p", "T", "Tv"],
        vertical_spacing=0.06,
    )

    for entry in primitive_slices:
        label = entry["label"]
        rho, u, p, T, Tv = entry["rho"], entry["u"], entry["p"], entry["T"], entry["Tv"]
        fig.add_trace(go.Scatter(x=x, y=rho, name=f"rho {label}"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=u, name=f"u {label}"), row=2, col=1)
        fig.add_trace(go.Scatter(x=x, y=p, name=f"p {label}"), row=3, col=1)
        fig.add_trace(go.Scatter(x=x, y=T, name=f"T {label}"), row=4, col=1)
        fig.add_trace(go.Scatter(x=x, y=Tv, name=f"Tv {label}"), row=5, col=1)

    fig.update_layout(title=title, height=900, width=900, showlegend=True)
    fig.show()


def main():
    print("=" * 80)
    print("Shock Tube (function-based initialization)")
    print("=" * 80)

    # --- user input start ---
    tube_length = 1.0
    n_cells = 400
    x0 = 0.5 * tube_length
    transition_width = 0.0  # set >0 for smooth tanh transition

    species_names = ("N2",)

    # Left state
    rho_L = 1.0
    u_L = 0.0
    p_L = 1.0e5
    T_L = None  # set to value if you want T(x) instead of p(x)
    Tv_L = None  # set to value if you want Tv(x)
    Y_L = jnp.array([1.0])

    # Right state
    rho_R = 0.125
    u_R = 0.0
    p_R = 1.0e4
    T_R = None
    Tv_R = None
    Y_R = jnp.array([1.0])

    boundary_condition = "transmissive"  # transmissive, reflective, periodic
    t_final = 1e-4
    save_interval = 20
    cfl = 0.4
    run_simulation = True
    # --- user input end ---

    x, dx = build_grid(n_cells, tube_length)

    species_table = load_species_table(species_names)

    rho_fn = make_piecewise_fn(x0, rho_L, rho_R, width=transition_width)
    u_fn = make_piecewise_fn(x0, u_L, u_R, width=transition_width)
    Y_fn = make_piecewise_fn(x0, Y_L, Y_R, width=transition_width)

    if (T_L is None) ^ (T_R is None):
        raise ValueError("Provide both T_L and T_R or neither.")
    if (Tv_L is None) ^ (Tv_R is None):
        raise ValueError("Provide both Tv_L and Tv_R or neither.")
    if Y_L.shape[0] != len(species_names) or Y_R.shape[0] != len(species_names):
        raise ValueError("Y_L and Y_R must match species_names length.")

    p_fn = None if (T_L is not None) else make_piecewise_fn(
        x0, p_L, p_R, width=transition_width
    )
    T_fn = None if p_fn is not None else make_piecewise_fn(
        x0, T_L, T_R, width=transition_width
    )
    Tv_fn = None
    if Tv_L is not None or Tv_R is not None:
        Tv_fn = make_piecewise_fn(x0, Tv_L, Tv_R, width=transition_width)

    rho, u, T, Tv, Y = build_initial_state(
        x, species_table, rho_fn, u_fn, Y_fn, p_fn=p_fn, T_fn=T_fn, Tv_fn=Tv_fn
    )

    eq_manager = build_equation_manager(
        species_table=species_table,
        dx=dx,
        dt=1e-8,
        boundary_condition=boundary_condition,
    )

    U_init = equation_manager_utils.compute_U_from_primitives(
        Y_s=Y, rho=rho, u=u, T_tr=T, T_V=Tv, equation_manager=eq_manager
    )

    dt = compute_cfl_dt(U_init, eq_manager, cfl=cfl)
    eq_manager = replace(
        eq_manager, numerics_config=replace(eq_manager.numerics_config, dt=dt)
    )

    print(f"dx = {dx:.3e} m, dt = {dt:.3e} s, n_cells = {n_cells}")

    if not run_simulation:
        plot_primitives_slices(
            x,
            [
                {
                    "label": "t=0",
                    "rho": rho,
                    "u": u,
                    "p": compute_pressure_from_primitives(
                        T, rho, Y, species_table.molar_masses
                    ),
                    "T": T,
                    "Tv": Tv,
                }
            ],
            title="Initial Condition",
        )
        return

    U_hist, t_hist = equation_manager.run(
        U_init=U_init,
        equation_manager=eq_manager,
        t_final=t_final,
        save_interval=save_interval,
    )

    # Plot initial/mid/final slices
    mid_idx = max(0, U_hist.shape[0] // 2)
    indices = [0, mid_idx, U_hist.shape[0] - 1]

    primitive_slices = []
    for idx in indices:
        Y_i, rho_i, T_i, Tv_i, p_i = equation_manager_utils.extract_primitives_from_U(
            U_hist[idx], eq_manager
        )
        primitive_slices.append(
            {
                "label": f"t={float(t_hist[idx]):.2e}s",
                "rho": rho_i,
                "u": U_hist[idx][:, eq_manager.species.n_species] / rho_i,
                "p": p_i,
                "T": T_i,
                "Tv": Tv_i,
            }
        )

    plot_primitives_slices(x, primitive_slices, title="Shock Tube Evolution")


if __name__ == "__main__":
    main()
