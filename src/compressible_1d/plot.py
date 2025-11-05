from jaxtyping import Float, Array
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import jax.numpy as jnp
from compressible_1d import physics


def plot_U_field(U_field: Float[Array, "3 n_cells n_time_steps"]) -> go.Figure:
    """
    Plot normalized density, velocity, and pressure over time
    for each cell in three stacked subplots.
    """
    # Normalize each field for visual comparability
    rho = U_field[0] / U_field[0].max()
    u = U_field[1] / U_field[1].max()
    p = U_field[2] / U_field[2].max()

    n_cells = U_field.shape[1]
    time_steps = list(range(U_field.shape[2]))

    # Create 3 vertically stacked subplots sharing the same x-axis
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Density ρ", "Velocity u", "Pressure p"),
    )

    # Add one trace per cell in each subplot
    for cell in range(n_cells):
        fig.add_trace(
            go.Scatter(
                x=time_steps,
                y=rho[cell],
                name=f"ρ cell {cell}",
                line=dict(color="blue", width=1),
                showlegend=(cell == 0),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=time_steps,
                y=u[cell],
                name=f"u cell {cell}",
                line=dict(color="red", width=1),
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=time_steps,
                y=p[cell],
                name=f"p cell {cell}",
                line=dict(color="green", width=1),
                showlegend=False,
            ),
            row=3,
            col=1,
        )

    # Common layout tweaks
    fig.update_layout(
        height=900,
        width=900,
        title="Normalized Field Values over Time per Cell",
        xaxis3_title="Time Step",
        yaxis_title="ρ (norm.)",
        yaxis2_title="u (norm.)",
        yaxis3_title="p (norm.)",
        legend=dict(yanchor="top", y=1.02, xanchor="left", x=0),
    )

    return fig


def plot_U_heatmaps(U_field) -> go.Figure:
    """
    Plot conserved variables (rho, rho*u, rho*E) as heatmaps:
      - x-axis: spatial cells
      - y-axis: time steps
      - color: value magnitude (blue → red)
    """
    # Ensure numpy array (handles JAX or numpy)
    U_np = jnp.asarray(U_field)
    _, n_cells, n_time = U_np.shape

    titles = ["Density ρ", "Momentum ρu", "Energy ρE"]

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, subplot_titles=titles, vertical_spacing=0.06
    )

    # Add one heatmap per conserved variable
    for i in range(3):
        fig.add_trace(
            go.Heatmap(
                z=U_np[i].T,  # (n_time, n_cells)
                colorscale="RdBu",
                reversescale=True,
                showscale=True,  # we’ll position the colorbar next
            ),
            row=i + 1,
            col=1,
        )

    # Compute each subplot's vertical domain to place colorbars correctly
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

    # Axis labels (only bottom x-axis needs a title)
    for r in range(1, 4):
        fig.update_yaxes(title_text="Time step", row=r, col=1)
    fig.update_xaxes(title_text="Cell index", row=3, col=1)

    # Layout polish
    fig.update_layout(
        title="Conserved Variables as Heatmaps",
        height=980,
        width=900,
        margin=dict(l=70, r=90, t=70, b=50),
    )

    # Optional: keep upper x tick labels off for clarity
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=2, col=1)

    return fig


def plot_U_slice(
    U_field: Float[Array, "3 n_cells n_time_steps"],
    time_id: int,
    show_conservative: bool = True,
    gamma: float | None = None,
) -> go.Figure:
    """
    Plot a single time slice of U_field along the spatial domain (cell index).

    Args:
        U_field: Array of shape (3, n_cells, n_time_steps)
        time_id: Index of the time step to visualize
        conservative: If True, plot conservative vars (ρ, ρu, ρE),
                      otherwise plot primitive vars (ρ, u, p)
    """
    # Convert variables if needed
    if not show_conservative:
        assert gamma is not None
        U_field = physics.to_primitives(U_field, gamma=gamma)

    U_np = jnp.asarray(U_field)
    n_cells = U_np.shape[1]
    x = jnp.arange(n_cells)

    # Extract slices
    field1 = U_np[0, :, time_id]
    field2 = U_np[1, :, time_id]
    field3 = U_np[2, :, time_id]

    # Labeling based on mode
    if show_conservative:
        titles = ("Density ρ", "Momentum ρu", "Energy ρE")
        ylabels = ("ρ", "ρu", "ρE")
        colors = ("blue", "orange", "red")
    else:
        titles = ("Density ρ", "Velocity u", "Pressure p")
        ylabels = ("ρ", "u", "p")
        colors = ("blue", "red", "green")

    # Create subplots
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=titles,
        vertical_spacing=0.08,
    )

    # Add traces
    for i, (data, color, ylabel) in enumerate(
        zip([field1, field2, field3], colors, ylabels), start=1
    ):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=data,
                mode="lines",
                line=dict(color=color),
                name=ylabel,
            ),
            row=i,
            col=1,
        )
        fig.update_yaxes(title_text=ylabel, row=i, col=1)

    # Axis labels and layout
    fig.update_xaxes(title_text="Cell index", row=3, col=1)
    fig.update_layout(
        title=f"{'Conservative' if show_conservative else 'Primitive'} Variables at Time Step {time_id}",
        height=850,
        width=900,
        showlegend=False,
        margin=dict(l=60, r=30, t=60, b=50),
    )

    return fig
