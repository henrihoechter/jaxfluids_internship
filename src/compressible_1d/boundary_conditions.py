"""Boundary conditions for 1D finite volume solver.

Handles boundary conditions with shape convention (n_cells, n_variables).
"""
import jax.numpy as jnp
from jaxtyping import Float, Array


def apply_boundary_conditions(
    U: Float[Array, "n_cells n_variables"],
    boundary_condition_type: str,
    n_ghosts: int,
) -> Float[Array, "n_cells+2*n_ghosts n_variables"]:
    """Apply boundary conditions to conserved variables.

    Shape convention: (n_cells, n_variables)
    - For two-temperature multi-species: [ρ_1, ..., ρ_ns, ρu, ρE, ρE_v]
    - n_variables = n_species + 3

    Args:
        U: Conserved state [n_cells, n_variables]
        boundary_condition_type: 'periodic', 'transmissive', or 'reflective'
        n_ghosts: Number of ghost cells per side

    Returns:
        U_with_ghosts: [n_cells + 2*n_ghosts, n_variables]

    Raises:
        NotImplementedError: If n_ghosts != 1
        ValueError: If unknown boundary condition type
    """
    if n_ghosts != 1:
        raise NotImplementedError("Only n_ghosts=1 currently supported")

    n_cells, n_variables = U.shape

    if boundary_condition_type == "periodic":
        # Left ghost: copy from right boundary
        # Right ghost: copy from left boundary
        left_ghost = U[-n_ghosts:, :]  # Last n_ghosts cells
        right_ghost = U[:n_ghosts, :]  # First n_ghosts cells
        U_with_ghosts = jnp.concatenate([left_ghost, U, right_ghost], axis=0)

    elif boundary_condition_type == "transmissive":
        # Left ghost: extrapolate from left boundary (zero gradient)
        # Right ghost: extrapolate from right boundary (zero gradient)
        left_ghost = U[:n_ghosts, :]  # Copy first cell
        right_ghost = U[-n_ghosts:, :]  # Copy last cell
        U_with_ghosts = jnp.concatenate([left_ghost, U, right_ghost], axis=0)

    elif boundary_condition_type == "reflective":
        # For multi-species two-temperature: [ρ_1, ..., ρ_ns, ρu, ρE, ρE_v]
        # Momentum index is n_variables - 3
        momentum_idx = n_variables - 3

        # Left ghost: reflect and negate momentum
        left_ghost = U[:n_ghosts, :].copy()
        left_ghost = left_ghost.at[:, momentum_idx].set(-left_ghost[:, momentum_idx])

        # Right ghost: reflect and negate momentum
        right_ghost = U[-n_ghosts:, :].copy()
        right_ghost = right_ghost.at[:, momentum_idx].set(-right_ghost[:, momentum_idx])

        U_with_ghosts = jnp.concatenate([left_ghost, U, right_ghost], axis=0)

    else:
        raise ValueError(f"Unknown boundary condition: {boundary_condition_type}")

    return U_with_ghosts
