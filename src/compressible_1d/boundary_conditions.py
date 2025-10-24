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
        U_with_ghosts = jnp.concatenate([U[:, -1:], U, U[:, :1]], axis=-1)

    elif boundary_condition_type == "transmissive":
        U_with_ghosts = jnp.concatenate([U[:, :1], U, U[:, -1:]], axis=-1)

    elif boundary_condition_type == "reflective":
        start = jnp.stack([1.0, -1.0, 1.0]) * U[:, 0]
        end = jnp.stack([1.0, -1.0, 1.0]) * U[:, -1]

        U_with_ghosts = jnp.concatenate(
            [start[:, jnp.newaxis], U, end[:, jnp.newaxis]], axis=-1
        )

    return U_with_ghosts
