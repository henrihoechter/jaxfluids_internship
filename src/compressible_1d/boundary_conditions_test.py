"""Unit tests for boundary_conditions.py"""

import jax
import jax.numpy as jnp
import pytest

from compressible_1d import boundary_conditions

# Configure JAX for testing
jax.config.update("jax_enable_x64", True)


def test_periodic_boundary_shape():
    """Test that periodic boundary conditions produce correct output shape."""
    n_cells = 10
    n_variables = 5
    n_ghosts = 1

    U = jnp.ones((n_cells, n_variables))
    U_with_ghosts = boundary_conditions.apply_boundary_conditions(
        U, "periodic", n_ghosts
    )

    expected_shape = (n_cells + 2 * n_ghosts, n_variables)
    assert U_with_ghosts.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {U_with_ghosts.shape}"
    )

    print("✓ Periodic boundary shape test passed")


def test_periodic_boundary_values():
    """Test that periodic boundary conditions copy correct values."""
    n_cells = 10
    n_variables = 5
    n_ghosts = 1

    # Create test data with distinct values
    U = jnp.arange(n_cells * n_variables).reshape(n_cells, n_variables)
    U_with_ghosts = boundary_conditions.apply_boundary_conditions(
        U, "periodic", n_ghosts
    )

    # Left ghost should equal rightmost interior cell
    assert jnp.allclose(U_with_ghosts[0, :], U[-1, :]), (
        "Left ghost should equal rightmost interior cell"
    )

    # Right ghost should equal leftmost interior cell
    assert jnp.allclose(U_with_ghosts[-1, :], U[0, :]), (
        "Right ghost should equal leftmost interior cell"
    )

    # Interior should be unchanged
    assert jnp.allclose(U_with_ghosts[n_ghosts:-n_ghosts, :], U), (
        "Interior cells should be unchanged"
    )

    print("✓ Periodic boundary values test passed")


def test_transmissive_boundary_shape():
    """Test that transmissive boundary conditions produce correct output shape."""
    n_cells = 10
    n_variables = 5
    n_ghosts = 1

    U = jnp.ones((n_cells, n_variables))
    U_with_ghosts = boundary_conditions.apply_boundary_conditions(
        U, "transmissive", n_ghosts
    )

    expected_shape = (n_cells + 2 * n_ghosts, n_variables)
    assert U_with_ghosts.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {U_with_ghosts.shape}"
    )

    print("✓ Transmissive boundary shape test passed")


def test_transmissive_boundary_values():
    """Test that transmissive boundary conditions extrapolate correctly."""
    n_cells = 10
    n_variables = 5
    n_ghosts = 1

    # Create test data with distinct values
    U = jnp.arange(n_cells * n_variables).reshape(n_cells, n_variables)
    U_with_ghosts = boundary_conditions.apply_boundary_conditions(
        U, "transmissive", n_ghosts
    )

    # Left ghost should equal leftmost interior cell (zero gradient)
    assert jnp.allclose(U_with_ghosts[0, :], U[0, :]), (
        "Left ghost should equal leftmost interior cell"
    )

    # Right ghost should equal rightmost interior cell (zero gradient)
    assert jnp.allclose(U_with_ghosts[-1, :], U[-1, :]), (
        "Right ghost should equal rightmost interior cell"
    )

    # Interior should be unchanged
    assert jnp.allclose(U_with_ghosts[n_ghosts:-n_ghosts, :], U), (
        "Interior cells should be unchanged"
    )

    print("✓ Transmissive boundary values test passed")


def test_reflective_boundary_shape():
    """Test that reflective boundary conditions produce correct output shape."""
    n_cells = 10
    n_variables = 8  # n_species=5, plus [ρu, ρE, ρE_v]
    n_ghosts = 1

    U = jnp.ones((n_cells, n_variables))
    U_with_ghosts = boundary_conditions.apply_boundary_conditions(
        U, "reflective", n_ghosts
    )

    expected_shape = (n_cells + 2 * n_ghosts, n_variables)
    assert U_with_ghosts.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {U_with_ghosts.shape}"
    )

    print("✓ Reflective boundary shape test passed")


def test_reflective_boundary_momentum_flip():
    """Test that reflective boundary conditions negate momentum."""
    n_cells = 10
    n_variables = 8  # n_species=5, plus [ρu, ρE, ρE_v]
    n_ghosts = 1

    # Create test data with positive momentum
    U = jnp.ones((n_cells, n_variables))
    momentum_idx = n_variables - 3  # Index of ρu
    U = U.at[:, momentum_idx].set(10.0)  # Set momentum to 10.0

    U_with_ghosts = boundary_conditions.apply_boundary_conditions(
        U, "reflective", n_ghosts
    )

    # Left ghost momentum should be negated
    assert jnp.allclose(U_with_ghosts[0, momentum_idx], -10.0), (
        f"Left ghost momentum should be -10.0, got {U_with_ghosts[0, momentum_idx]}"
    )

    # Right ghost momentum should be negated
    assert jnp.allclose(U_with_ghosts[-1, momentum_idx], -10.0), (
        f"Right ghost momentum should be -10.0, got {U_with_ghosts[-1, momentum_idx]}"
    )

    # Other variables should be unchanged in ghost cells
    for i in range(n_variables):
        if i != momentum_idx:
            assert jnp.allclose(U_with_ghosts[0, i], U[0, i]), (
                f"Left ghost variable {i} should be unchanged"
            )
            assert jnp.allclose(U_with_ghosts[-1, i], U[-1, i]), (
                f"Right ghost variable {i} should be unchanged"
            )

    # Interior should be unchanged
    assert jnp.allclose(U_with_ghosts[n_ghosts:-n_ghosts, :], U), (
        "Interior cells should be unchanged"
    )

    print("✓ Reflective boundary momentum flip test passed")


def test_invalid_boundary_condition():
    """Test that invalid boundary condition raises ValueError."""
    n_cells = 10
    n_variables = 5
    n_ghosts = 1

    U = jnp.ones((n_cells, n_variables))

    with pytest.raises(ValueError, match="Unknown boundary condition"):
        boundary_conditions.apply_boundary_conditions(U, "invalid_bc", n_ghosts)

    print("✓ Invalid boundary condition test passed")


def test_multiple_ghost_cells_raises_error():
    """Test that n_ghosts != 1 raises NotImplementedError."""
    n_cells = 10
    n_variables = 5
    n_ghosts = 2

    U = jnp.ones((n_cells, n_variables))

    with pytest.raises(NotImplementedError, match="Only n_ghosts=1 currently supported"):
        boundary_conditions.apply_boundary_conditions(U, "periodic", n_ghosts)

    print("✓ Multiple ghost cells error test passed")
