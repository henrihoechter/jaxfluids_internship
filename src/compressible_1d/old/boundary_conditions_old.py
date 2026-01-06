import jax.numpy as jnp
from jaxtyping import Float, Array


def apply_boundary_condition(
    U: Float[Array, "3 N"], boundary_condition_type: str, n_ghosts: int
) -> Float[Array, "3 N+2*n_ghosts"]:
    """Apply boundary conditions to conserved variables.

    Works for both single-species (3 variables) and multi-species systems.
    For multi-species two-temperature model, U has shape [n_conserved, N].

    Args:
        U: Conserved variables [n_conserved, N]
        boundary_condition_type: Type of boundary condition
        n_ghosts: Number of ghost cells per side

    Returns:
        U_with_ghosts: Variables with ghost cells [n_conserved, N+2*n_ghosts]
    """
    if n_ghosts < 1:
        raise ValueError("Number of ghost cells must be at least 1.")

    if n_ghosts > 1:
        raise NotImplementedError("Test")

    if boundary_condition_type == "periodic":
        U_with_ghosts = jnp.concatenate([U[:, -n_ghosts:], U, U[:, :n_ghosts]], axis=1)

    elif boundary_condition_type == "transmissive":
        U_with_ghosts = jnp.concatenate([U[:, :n_ghosts], U, U[:, -n_ghosts:]], axis=1)

    elif boundary_condition_type == "reflective":
        # For multi-species: reflect all species densities, negate momentum, keep energies
        # Build reflection mask: [1, 1, ..., 1, -1, 1, 1] where -1 is at momentum index
        n_conserved = U.shape[0]

        # For single-species (3 variables): [rho, rho*u, rho*E] -> [1, -1, 1]
        # For two-temp multi-species: [rho_1, ..., rho_ns, rho*u, rho*E, rho*E_v] -> [1, ..., 1, -1, 1, 1]
        if n_conserved == 3:
            # Single-species Euler
            reflection_mask = jnp.array([1.0, -1.0, 1.0])
        else:
            # Multi-species two-temperature: momentum is at index n_species
            reflection_mask = jnp.ones(n_conserved)
            # Momentum index is n_conserved - 3
            momentum_idx = n_conserved - 3
            reflection_mask = reflection_mask.at[momentum_idx].set(-1.0)

        start = reflection_mask[:, jnp.newaxis] * U[:, 0:1]
        end = reflection_mask[:, jnp.newaxis] * U[:, -1:]

        U_with_ghosts = jnp.concatenate([start, U, end], axis=1)

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
