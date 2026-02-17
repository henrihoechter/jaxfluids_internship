from __future__ import annotations

from typing import Callable, Sequence

import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from compressible_core import chemistry_utils, constants, energy_models
from compressible_2d import mesh_gmsh


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
            cells.append([n00, n10, n11, n01])  # CCW

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


def make_spacetime_heatmap_figure(
    x: np.ndarray,
    t: np.ndarray,
    primitives: dict[str, np.ndarray],
    title: str | None = None,
    color_limits: dict[str, tuple[float, float]] | None = None,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
):
    n_cols = len(primitives)
    h_spacing = 0.01
    plot_width = (1.0 - (n_cols - 1) * h_spacing) / n_cols
    cb_centers = [i * (plot_width + h_spacing) + plot_width / 2 for i in range(n_cols)]

    fig = make_subplots(
        rows=1,
        cols=n_cols,
        subplot_titles=list(primitives.keys()),
        shared_yaxes=True,
        horizontal_spacing=h_spacing,
    )

    for col, (name, z) in enumerate(primitives.items(), start=1):
        trace = go.Heatmap(
            x=np.array(x),
            y=np.array(t),
            z=np.array(z),
            colorscale="Jet",
            colorbar=dict(
                orientation="h",
                x=cb_centers[col - 1],
                xanchor="center",
                y=-0.18,
                yanchor="top",
                len=plot_width * 0.95,
                thickness=12,
                title=dict(text=name, side="bottom"),
            ),
        )
        if color_limits is not None and name in color_limits:
            vmin, vmax = color_limits[name]
            trace.update(zmin=float(vmin), zmax=float(vmax))
        fig.add_trace(trace, row=1, col=col)

    fig.update_yaxes(title_text="t [s]", col=1, matches="y")
    fig.update_xaxes(title_text="x [m]", matches="x")
    if x_range is not None:
        fig.update_xaxes(range=[float(x_range[0]), float(x_range[1])])
    if y_range is not None:
        fig.update_yaxes(range=[float(y_range[0]), float(y_range[1])])

    fig.update_layout(
        height=650,
        width=1400,
        margin=dict(b=120),
    )
    if title:
        fig.update_layout(title=title)
    return fig


def make_spacetime_heatmap_figure_2rows(
    x_top: np.ndarray,
    t_top: np.ndarray,
    primitives_top: dict[str, np.ndarray],
    x_bottom: np.ndarray,
    t_bottom: np.ndarray,
    primitives_bottom: dict[str, np.ndarray],
    title: str | None = None,
    color_limits: dict[str, tuple[float, float]] | None = None,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    row_titles: tuple[str, str] = ("2D centerline", "1D"),
):
    names = list(primitives_top.keys())
    n_cols = len(names)
    h_spacing = 0.01
    plot_width = (1.0 - (n_cols - 1) * h_spacing) / n_cols
    cb_centers = [i * (plot_width + h_spacing) + plot_width / 2 for i in range(n_cols)]

    subplot_titles = names + [""] * n_cols
    fig = make_subplots(
        rows=2,
        cols=n_cols,
        subplot_titles=subplot_titles,
        row_titles=list(row_titles),
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=h_spacing,
        vertical_spacing=0.1,
    )

    for col, name in enumerate(names, start=1):
        z_top = primitives_top[name]
        z_bottom = primitives_bottom[name]

        trace_top = go.Heatmap(
            x=np.array(x_top),
            y=np.array(t_top),
            z=np.array(z_top),
            colorscale="Jet",
            colorbar=dict(
                orientation="h",
                x=cb_centers[col - 1],
                xanchor="center",
                y=-0.12,
                yanchor="top",
                len=plot_width * 0.95,
                thickness=12,
                title=dict(text=name, side="bottom"),
            ),
        )
        if color_limits is not None and name in color_limits:
            vmin, vmax = color_limits[name]
            trace_top.update(zmin=float(vmin), zmax=float(vmax))

        trace_bottom = go.Heatmap(
            x=np.array(x_bottom),
            y=np.array(t_bottom),
            z=np.array(z_bottom),
            colorscale="Jet",
            showscale=False,
        )
        if color_limits is not None and name in color_limits:
            vmin, vmax = color_limits[name]
            trace_bottom.update(zmin=float(vmin), zmax=float(vmax))

        fig.add_trace(trace_top, row=1, col=col)
        fig.add_trace(trace_bottom, row=2, col=col)

    fig.update_yaxes(title_text="t [s]", row=1, col=1, matches="y")
    fig.update_xaxes(title_text="x [m]", row=2, col=1, matches="x")
    if x_range is not None:
        fig.update_xaxes(range=[float(x_range[0]), float(x_range[1])])
    if y_range is not None:
        fig.update_yaxes(range=[float(y_range[0]), float(y_range[1])])

    fig.update_layout(
        height=900,
        width=1400,
        margin=dict(b=140),
    )
    if title:
        fig.update_layout(title=title)
    return fig
