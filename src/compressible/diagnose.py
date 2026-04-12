"""Runtime diagnostic helpers for solver states."""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import jaxtyping as jt
from beartype import beartype

ATOL = 1e-6


def runtime_check_array_sizes(function: Callable[..., object]) -> Callable[..., object]:
    """Wrap a function with runtime jaxtyping checks."""
    return jt.jaxtyped(typechecker=beartype)(function)


def check_conservation(
    U: jt.Float[jt.Array, "n_variables n_cells"],
    U_ref: jt.Float[jt.Array, "n_variables n_cells"],
    debug: bool = False,
    abort: bool = True,
) -> None:
    """Check that the total conserved variables match a reference state."""
    if debug:
        print(
            f"Mass: \t\tU={jnp.sum(U[0, :]):2.5e}, \tU_ref={jnp.sum(U_ref[0, :]):2.5e}, \tdiff (abs)={jnp.sum(U[0, :]) - jnp.sum(U_ref[0, :]):2.5e}"
        )
        print(
            f"Momentum: \tU={jnp.sum(U[1, :]):2.5e}, \tU_ref={jnp.sum(U_ref[1, :]):2.5e}, \tdiff (abs)={jnp.sum(U[1, :]) - jnp.sum(U_ref[1, :]):2.5e}"
        )
        print(
            f"Energy: \tU={jnp.sum(U[2, :]):2.5e}, \tU_ref={jnp.sum(U_ref[2, :]):2.5e}, \tdiff (abs)={jnp.sum(U[2, :]) - jnp.sum(U_ref[2, :]):2.5e}"
        )

    total_U = jnp.sum(U, axis=1)
    total_Uref = jnp.sum(U_ref, axis=1)

    if abort and not jnp.allclose(total_U, total_Uref, atol=ATOL):
        raise ValueError("U is not conserved.")

    return None


def check_nonnegativity(U: jt.Float[jt.Array, "n_variables n_cells"]) -> None:
    """Check that mass and energy densities stay nonnegative."""
    if jnp.any(U[0, :] < 0.0):
        raise ValueError("Mass density negative.")

    if jnp.any(U[2, :] < 0.0):
        raise ValueError("Energy density negative.")


def check_nan_inf(U: jt.Float[jt.Array, "n_variables n_cells"]) -> None:
    """Check that the state does not contain NaN or Inf values."""
    if jnp.any(jnp.isnan(U)):
        raise ValueError("NaN values present in solution.")

    if jnp.any(jnp.isinf(U)):
        raise ValueError("Inf values present in solution.")


def check_all(
    U: jt.Float[jt.Array, "n_variables n_cells"],
    U_ref: jt.Float[jt.Array, "n_variables n_cells"],
    debug: bool,
    abort: bool = True,
) -> None:
    """Run all diagnostic checks on the current state."""
    check_nan_inf(U)
    check_nonnegativity(U)
    check_conservation(U, U_ref, debug, abort)


def live_diagnostics(
    U: jt.Float[jt.Array, "n_variables n_cells"], step: int
) -> None:
    """Print a compact live diagnostic line."""
    print(
        f"step {step}, \tmass density max-min: {jnp.max(U[0, :]) - jnp.min(U[0, :]):.4e}"
    )
