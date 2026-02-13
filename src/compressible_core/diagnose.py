import jax.numpy as jnp
import jaxtyping as jt
from beartype import beartype

ATOL = 1e-6


def runtime_check_array_sizes(f):
    """Decorator to enforce jaxtyping shape annotations at runtime."""
    return jt.jaxtyped(typechecker=beartype)(f)


def check_conservation(U, U_ref, debug: bool = False, abort: bool = True) -> None:
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


def check_nonnegativity(U) -> None:
    if jnp.any(U[0, :] < 0.0):
        raise ValueError("Mass density negative.")

    if jnp.any(U[2, :] < 0.0):
        raise ValueError("Energy density negative.")


def check_nan_inf(U) -> None:
    if jnp.any(jnp.isnan(U)):
        raise ValueError("NaN values present in solution.")

    if jnp.any(jnp.isinf(U)):
        raise ValueError("Inf values present in solution.")


def check_all(U, U_ref, debug: bool, abort: bool = True) -> None:
    check_nan_inf(U)
    check_nonnegativity(U)
    check_conservation(U, U_ref, debug, abort)


def live_diagnostics(U, step):
    print(
        f"step {step}, \tmass density max-min: {jnp.max(U[0, :]) - jnp.min(U[0, :]):.4e}"
    )
