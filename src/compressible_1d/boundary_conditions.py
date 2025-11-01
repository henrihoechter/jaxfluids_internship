import jax.numpy as jnp
from jaxtyping import Float, Array


def apply_boundary_condition(
    U: Float[Array, "3 N"], boundary_condition_type: str, n_ghosts: int
) -> Float[Array, "3 N+2*n_ghosts"]:
    if n_ghosts < 1:
        raise ValueError("Number of ghost cells must be at least 1.")

    if n_ghosts > 1:
        raise NotImplementedError("Test")

    if boundary_condition_type == "periodic":
        U_with_ghosts = jnp.concatenate([U[:, -n_ghosts:], U, U[:, :n_ghosts]], axis=1)

    elif boundary_condition_type == "transmissive":
        U_with_ghosts = jnp.concatenate([U[:, :n_ghosts], U, U[:, -n_ghosts:]], axis=1)

    elif boundary_condition_type == "reflective":
        start = jnp.stack([1.0, -1.0, 1.0]) * U[:, 0]
        end = jnp.stack([1.0, -1.0, 1.0]) * U[:, -1]

        U_with_ghosts = jnp.concatenate(
            [start[:, jnp.newaxis], U, end[:, jnp.newaxis]], axis=1
        )

    return U_with_ghosts


def initialize_two_domains(
    rho_left: float,
    rho_right: float,
    u_left: float,
    u_right: float,
    p_left: float,
    p_right: float,
    n_cells: int,
) -> Float[Array, "3 n_cells"]:
    """Initialize a domain with two different states separated in the middle.

    Args:
        rho_left: Mass density on the left side.
        rho_right: Mass density on the right side.
        u_left: Velocity on the left side.
        u_right: Velocity on the right side.
        p_left: Pressure on the left side.
        p_right: Pressure on the right side.
        n_cells: Number of cells in the domain.

    Returns:
        U: Conserved variables array of shape (3, n_cells).
    """
    # Define left and right states
    mid = n_cells // 2
    rho = jnp.where(jnp.arange(n_cells) < mid, rho_left, rho_right)
    rhoU = jnp.where(jnp.arange(n_cells) < mid, u_left, u_right)
    rhoE = jnp.where(jnp.arange(n_cells) < mid, p_left, p_right)

    return jnp.stack([rho, rhoU, rhoE], axis=0)
