"""Migration utilities for the compressible solver."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float


def upgrade_state_1d_to_unified(
    U_1d: Float[Array, "n_cells n_variables_1d"],
    n_species: int,
) -> Float[Array, "n_cells n_variables"]:
    """Convert a 1D state vector (n+3) to the unified layout (n+4).

    The 1D layout is [rho_s..., rho*u, rho*E, rho*Ev] (n_species + 3 columns).
    The unified layout is [rho_s..., rho*u, rho*v, rho*E, rho*Ev] (n_species + 4).
    This inserts a zero rho*v column at position n_species+1.

    Args:
        U_1d: 1D conserved state [n_cells, n_species + 3].
        n_species: Number of species.

    Returns:
        U: Unified conserved state [n_cells, n_species + 4] with rho*v = 0.
    """
    n_cells = U_1d.shape[0]
    return jnp.concatenate(
        [
            U_1d[:, : n_species + 1],  # rho_s..., rho*u
            jnp.zeros((n_cells, 1)),  # rho*v = 0
            U_1d[:, n_species + 1 :],  # rho*E, rho*Ev
        ],
        axis=1,
    )
