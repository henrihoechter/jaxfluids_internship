from jaxtyping import Float, Array
import jax.numpy as jnp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from compressible_1d import physics


def plot_U_heatmaps(
    U_field: Float[Array, "3 n_cells n_time"],
    to_primitive: bool = False,
    gamma: float | None = None,
) -> go.Figure:
    """
    Plot either conserved or primitive variables as heatmaps:
      - conserved: (rho, rho*u, rho*E)
      - primitive: (rho, u, p)

    Args:
        U_field: state field over time
        to_primitive: whether to convert from conserved → primitive
        gamma: ratio of specific heats (for p = (γ - 1)*(E - ½ρu²))
    """
    if to_primitive and gamma is None:
        raise ValueError(
            "gamma must be provided when converting to primitive variables."
        )

    U_np = jnp.asarray(U_field)
    _, n_cells, n_time = U_np.shape

    # --- Convert to primitive if requested ---
    if to_primitive:
        U_np = physics.to_primitives(U_np, gamma=gamma)
        titles = ["Density ρ", "Velocity u", "Pressure p"]
    else:
        titles = ["Density ρ", "Momentum ρu", "Energy ρE"]

    # --- Normalized x-axis ---
    x_norm = jnp.linspace(0.0, 1.0, n_cells)
    tickvals = jnp.arange(0.0, 1.01, 0.1)
    ticktext = [f"{v:.1f}" for v in tickvals]

    # --- Build figure with subplots ---
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, subplot_titles=titles, vertical_spacing=0.06
    )

    for i in range(3):
        fig.add_trace(
            go.Heatmap(
                z=U_np[i].T,  # shape (n_time, n_cells)
                x=x_norm,  # normalized x-axis
                colorscale="RdBu",
                reversescale=True,
                showscale=True,
            ),
            row=i + 1,
            col=1,
        )

    # --- Adjust colorbars vertically per subplot ---
    domains = [fig.layout[f"yaxis{i if i > 1 else ''}"].domain for i in range(1, 4)]
    for i, (y0, y1) in enumerate(domains, start=1):
        mid = 0.5 * (y0 + y1)
        span = y1 - y0
        fig.data[i - 1].update(
            colorbar=dict(
                title=titles[i - 1],
                x=1.02,
                xanchor="left",
                y=mid,
                yanchor="middle",
                len=span * 0.9,
                thickness=14,
                title_side="right",
            )
        )

    # --- Axis labels ---
    for r in range(1, 4):
        fig.update_yaxes(title_text="Time step", row=r, col=1)
        fig.update_xaxes(
            title_text="Normalized Cell Index (x/L)" if r == 3 else "",
            tickvals=tickvals,
            ticktext=ticktext,
            range=[0, 1],
            showticklabels=True,  # <-- show labels on all subplots
            row=r,
            col=1,
        )

    fig.update_layout(
        title=f"{'Primitive' if to_primitive else 'Conserved'} Variables as Heatmaps",
        height=980,
        width=900,
        margin=dict(l=70, r=90, t=70, b=50),
    )

    return fig


def plot_U_slice(
    U_field: Float[Array, "3 n_cells"],
    to_primitive: bool = False,
    gamma: float | None = None,
    fig: go.Figure | None = None,
    line_dash: str = "solid",
    legend_label: str = "Series",
) -> go.Figure:
    """
    Plot a single time slice of U_field along the spatial domain (normalized x in [0,1]),
    optionally overlaying onto an existing figure. Uses consistent colors per field and
    user-specified line dash style for all subplots. A legend label specifying the whole
    state series can be provided.
    """
    assert len(U_field.shape) == 2, "U_field must be 2D (3, n_cells)."

    # Convert variables if needed
    if to_primitive:
        assert gamma is not None, "gamma must be provided when plotting primitives."
        U_field = physics.to_primitives(U_field, gamma=gamma)

    U_np = jnp.asarray(U_field)
    n_cells = U_np.shape[1]
    x = jnp.linspace(0.0, 1.0, n_cells)  # normalized domain

    # Extract slices
    field1, field2, field3 = (
        U_np[0, :],
        U_np[1, :],
        U_np[2, :],
    )

    # Labels and colors
    if not to_primitive:
        titles = ("Density ρ", "Momentum ρu", "Energy ρE")
        ylabels = ("ρ", "ρu", "ρE")
        colors = ("blue", "orange", "red")
    else:
        titles = ("Density ρ", "Velocity u", "Pressure p")
        ylabels = ("ρ", "u", "p")
        colors = ("blue", "red", "green")

    created_fig = False
    if fig is None:
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=False,
            subplot_titles=titles,
            vertical_spacing=0.14,
        )
        # Layout
        fig.update_layout(
            title=f"{'Conservative' if not to_primitive else 'Primitive'} Variables",
            height=850,
            width=900,
            showlegend=True,
            margin=dict(l=60, r=30, t=100, b=50),
            plot_bgcolor="white",
            paper_bgcolor="white",
            legend=dict(
                orientation="h",  # horizontal
                yanchor="bottom",
                y=1.05,  # above the plots
                xanchor="center",
                x=0.5,
                tracegroupgap=10,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="black",
                borderwidth=0.5,
            ),
        )
        created_fig = True

        for r in (1, 2, 3):
            fig.update_xaxes(
                title_text="Normalized cell index",
                range=[0, 1],
                dtick=0.1,  # vertical gridlines every 0.1
                showgrid=True,
                gridcolor="black",
                zeroline=False,
                linecolor="black",
                mirror=True,
                row=r,
                col=1,
            )
        for i, ylabel in enumerate(ylabels, start=1):
            fig.update_yaxes(
                title_text=ylabel,
                showgrid=True,
                gridcolor="black",
                zeroline=False,
                linecolor="black",
                mirror=True,
                row=i,
                col=1,
            )

    # Add traces
    fields = [field1, field2, field3]
    for i, (data, color, ylabel) in enumerate(zip(fields, colors, ylabels), start=1):
        fig.add_trace(
            go.Scatter(
                x=jnp.asarray(x),
                y=jnp.asarray(data),
                mode="lines",
                line=dict(color=color, dash=line_dash),
                name=f"{ylabel} — {legend_label}",
                legendgroup=ylabel,
                showlegend=True,
                hovertemplate=f"x=%{{x:.3f}}<br>{ylabel}=%{{y:.6g}}<extra>{legend_label}</extra>",
            ),
            row=i,
            col=1,
        )
        if created_fig:
            fig.data[-1].legendgrouptitle = dict(text=ylabel)

    return fig
