import jax.numpy as jnp
import pytest

from src.compressible_1d import diagnose


def test_check_nonnegativity_success():
    U_test = jnp.ones((3, 10))

    diagnose.check_nonnegativity(U_test)


def test_check_nonnegativity_with_wrong_mass():
    """Tests if wrong mass is caught correctly."""
    U_test = jnp.concatenate(
        [jnp.stack([-1.0, 0.0, 0.0], axis=0)[:, jnp.newaxis], jnp.ones((3, 10))],
        axis=-1,
    )
    with pytest.raises(ValueError, match="Mass"):
        diagnose.check_nonnegativity(U_test)


def test_check_nonnegativity_with_wrong_energy():
    """Tests if wrong mass is caught correctly."""
    U_test = jnp.concatenate(
        [jnp.stack([0.0, 0.0, -1.0], axis=0)[:, jnp.newaxis], jnp.ones((3, 10))],
        axis=-1,
    )
    with pytest.raises(ValueError, match="Energy"):
        diagnose.check_nonnegativity(U_test)


def test_check_nan_inf():
    U_test_nan = jnp.array([[jnp.nan, 1.0], [1.0, 1.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="NaN"):
        diagnose.check_nan_inf(U_test_nan)

    U_test_inf = jnp.array([[jnp.inf, 1.0], [1.0, 1.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="Inf"):
        diagnose.check_nan_inf(U_test_inf)
