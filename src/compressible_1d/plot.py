from jaxtyping import Float, Array
import plotly.graph_objects as go


def plot_U_field(U_field: Float[Array, "3 n_cells n_time_steps"]) -> go.Figure:
    # Create subplots
    fig = go.Figure()

    # Get normalized values for each field
    rho = U_field[0] / U_field[0].max()  # density
    u = U_field[1] / U_field[1].max()  # velocity
    p = U_field[2] / U_field[2].max()  # pressure

    # Create time steps array
    time_steps = list(range(U_field.shape[2]))

    # Add traces for each cell
    for cell in range(U_field.shape[1]):
        fig.add_trace(
            go.Scatter(
                x=time_steps,
                y=rho[cell],
                name=f"ρ cell {cell}",
                line=dict(color="blue", width=1),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=time_steps,
                y=u[cell],
                name=f"u cell {cell}",
                line=dict(color="red", width=1),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=time_steps,
                y=p[cell],
                name=f"p cell {cell}",
                line=dict(color="green", width=1),
            )
        )

    # Update layout
    fig.update_layout(
        title="Normalized Field Values over Time",
        height=800,
        yaxis_title="Normalized Value",
        xaxis_title="Time Step",
        showlegend=True,
    )

    return fig
