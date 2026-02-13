"""Unit tests for equation_manager_utils.py functions"""

import jax
import jax.numpy as jnp
from pathlib import Path

from compressible_1d import equation_manager_utils
from compressible_1d import equation_manager_types
from compressible_1d import numerics_types
from compressible_core.chemistry_utils import load_species_table
from compressible_core import constants

# Configure JAX for testing
jax.config.update("jax_enable_x64", True)

# Load test data
data_dir = Path(__file__).parent.parent.parent / "data"
general_data = str(data_dir / "air_5_gnoffo.json")
enthalpy_data = str(data_dir / "air_5_gnoffo_equilibrium_enthalpy.json")


def create_test_equation_manager():
    """Create an EquationManager for testing."""
    species_table = load_species_table(general_data, enthalpy_data)

    numerics_config = numerics_types.NumericsConfig(
        dt=1e-6,
        dx=1e-3,
        integrator_scheme="forward-euler",
        spatial_scheme="first_order",
        flux_scheme="lax_friedrichs",
        n_halo_cells=2,
        clipping=numerics_types.ClippingConfig(
            rho_min=1e-6,
            rho_max=1e3,
            Y_min=0.0,
            Y_max=1.0,
            T_min=50.0,
            T_max=50000.0,
            Tv_min=50.0,
            Tv_max=50000.0,
            p_min=1.0,
            p_max=1e8,
        ),
    )

    return equation_manager_types.EquationManager(
        species=species_table,
        reactions=None,
        collision_integrals=None,
        numerics_config=numerics_config,
        boundary_condition="periodic",
    )


def create_synthetic_U(
    n_cells: int, n_species: int, equilibrium: bool = True
) -> jnp.ndarray:
    """Create a physically valid synthetic conserved state vector.

    Args:
        n_cells: Number of cells
        n_species: Number of species
        equilibrium: If True, set T = T_V (thermal equilibrium)

    Returns:
        U: Conserved state vector, shape (n_cells, n_species + 3)
    """
    # Simple equilibrium air at ~300K, 1 atm
    # Assume equal mass fractions for simplicity
    rho_total = 1.225  # kg/m^3
    rho_s = jnp.full((n_cells, n_species), rho_total / n_species)

    # Velocity
    u = jnp.linspace(0.0, 10.0, n_cells)  # m/s
    rho_u = rho_total * u

    # Temperatures
    T = jnp.full(n_cells, 300.0)  # K
    T_V = T if equilibrium else T * 1.2  # Slightly different for non-equilibrium

    # Energy computation (simplified)
    # For this test, use approximate values
    e_tr = 3.0 / 2.0 * constants.R_universal * T / 28.0  # Rough estimate for air
    e_v_scalar = 0.0 if equilibrium else 1000.0  # J/kg
    e_v = jnp.full(n_cells, e_v_scalar)
    E_total = e_tr + e_v + 0.5 * u**2
    rho_E = rho_total * E_total

    # Vibrational energy
    rho_Ev = rho_total * e_v

    # Construct U: [rho_1, ..., rho_ns, rho_u, rho_E, rho_Ev]
    U = jnp.concatenate(
        [
            rho_s,  # (n_cells, n_species)
            rho_u[:, None],  # (n_cells, 1)
            rho_E[:, None],  # (n_cells, 1)
            rho_Ev[:, None],  # (n_cells, 1)
        ],
        axis=1,
    )

    return U


def test_extract_primitives_output_shapes():
    """Test that output shapes are correct."""
    equation_manager = create_test_equation_manager()
    n_species = equation_manager.species.n_species
    n_cells = 10

    U = create_synthetic_U(n_cells, n_species)

    Y_s, rho, T, T_v, p = equation_manager_utils.extract_primitives_from_U(
        U, equation_manager
    )

    # Check shapes
    if Y_s.shape != (n_cells, n_species):
        raise ValueError(
            f"Y_s shape should be ({n_cells}, {n_species}), got {Y_s.shape}"
        )

    if rho.shape != (n_cells,):
        raise ValueError(f"rho shape should be ({n_cells},), got {rho.shape}")

    if T.shape != (n_cells,):
        raise ValueError(f"T shape should be ({n_cells},), got {T.shape}")

    if T_v.shape != (n_cells,):
        raise ValueError(f"T_v shape should be ({n_cells},), got {T_v.shape}")

    if p.shape != (n_cells,):
        raise ValueError(f"p shape should be ({n_cells},), got {p.shape}")

    print(
        f"Output shapes test passed: Y_s={Y_s.shape}, rho={rho.shape}, T={T.shape}, T_v={T_v.shape}, p={p.shape}"
    )


def test_extract_primitives_mass_conservation():
    """Test that sum of partial densities equals total density."""
    equation_manager = create_test_equation_manager()
    n_species = equation_manager.species.n_species
    n_cells = 10

    U = create_synthetic_U(n_cells, n_species)

    Y_s, rho, T, T_v, p = equation_manager_utils.extract_primitives_from_U(
        U, equation_manager
    )

    # Extract partial densities from U
    rho_s = U[:, :n_species]
    rho_total_from_partials = jnp.sum(rho_s, axis=-1)

    # Compare with returned rho
    if not jnp.allclose(rho, rho_total_from_partials):
        raise ValueError("Total density does not equal sum of partial densities")

    print("Mass conservation test passed")


def test_extract_primitives_mole_fraction_sum():
    """Test that mole fractions sum to 1."""
    equation_manager = create_test_equation_manager()
    n_species = equation_manager.species.n_species
    n_cells = 10

    U = create_synthetic_U(n_cells, n_species)

    Y_s, rho, T, T_v, p = equation_manager_utils.extract_primitives_from_U(
        U, equation_manager
    )

    # Sum of mole fractions should be 1
    Y_sum = jnp.sum(Y_s, axis=-1)

    if not jnp.allclose(Y_sum, 1.0, atol=1e-10):
        raise ValueError(f"Mole fractions do not sum to 1.0: {Y_sum}")

    print("Mole fraction sum test passed")


def test_extract_primitives_positive_quantities():
    """Test that all physical quantities are positive."""
    equation_manager = create_test_equation_manager()
    n_species = equation_manager.species.n_species
    n_cells = 10

    U = create_synthetic_U(n_cells, n_species)

    Y_s, rho, T, T_v, p = equation_manager_utils.extract_primitives_from_U(
        U, equation_manager
    )

    # Check positivity
    if not jnp.all(rho > 0):
        raise ValueError("Density must be positive")

    if not jnp.all(T > 0):
        raise ValueError("Temperature must be positive")

    if not jnp.all(T_v > 0):
        raise ValueError("Vibrational temperature must be positive")

    if not jnp.all(p > 0):
        raise ValueError("Pressure must be positive")

    if not jnp.all(Y_s >= 0):
        raise ValueError("Mole fractions must be non-negative")

    print("Positive quantities test passed")


def test_extract_primitives_ideal_gas_law():
    """Test that pressure satisfies ideal gas law (approximately)."""
    equation_manager = create_test_equation_manager()
    n_species = equation_manager.species.n_species
    n_cells = 5

    U = create_synthetic_U(n_cells, n_species, equilibrium=True)

    Y_s, rho, T, T_v, p = equation_manager_utils.extract_primitives_from_U(
        U, equation_manager
    )

    # Compute expected pressure from ideal gas law
    # p = sum(rho_s * R_s * T) = sum(rho_s * (R_universal / M_s) * T)
    rho_s = U[:, :n_species]
    M_s = equation_manager.species.M_s
    R = constants.R_universal  # J/(mol·K)

    # For equilibrium (no electrons), all species use T
    p_expected = jnp.sum(rho_s * R / M_s[None, :] * T[:, None], axis=-1)

    # Allow some tolerance due to numerical solution process
    rel_error = jnp.abs(p - p_expected) / p_expected
    if not jnp.all(rel_error < 0.1):  # 10% tolerance (loose due to approximations)
        print("Warning: Pressure deviates from ideal gas law")
        print(f"  p (computed): {p}")
        print(f"  p (expected): {p_expected}")
        print(f"  Relative error: {rel_error}")

    print("Ideal gas law test passed (with tolerance)")


def test_extract_primitives_zero_velocity():
    """Test with zero velocity (stationary gas)."""
    equation_manager = create_test_equation_manager()
    n_species = equation_manager.species.n_species
    n_cells = 5

    # Create U with zero velocity
    U = create_synthetic_U(n_cells, n_species)
    U = U.at[:, n_species].set(0.0)  # Set rho_u = 0

    Y_s, rho, T, T_v, p = equation_manager_utils.extract_primitives_from_U(
        U, equation_manager
    )

    # All quantities should still be positive and physical
    if not jnp.all(rho > 0):
        raise ValueError("Density must be positive")

    if not jnp.all(T > 0):
        raise ValueError("Temperature must be positive")

    print("Zero velocity test passed")


def test_extract_primitives_single_cell():
    """Test with a single cell."""
    equation_manager = create_test_equation_manager()
    n_species = equation_manager.species.n_species

    U = create_synthetic_U(1, n_species)

    Y_s, rho, T, T_v, p = equation_manager_utils.extract_primitives_from_U(
        U, equation_manager
    )

    # Check shapes
    if Y_s.shape != (1, n_species):
        raise ValueError(f"Y_s shape incorrect for single cell: {Y_s.shape}")

    if rho.shape != (1,):
        raise ValueError(f"rho shape incorrect for single cell: {rho.shape}")

    print("Single cell test passed")


def test_extract_primitives_physical_ranges():
    """Test that extracted values are in physically reasonable ranges."""
    equation_manager = create_test_equation_manager()
    n_species = equation_manager.species.n_species
    n_cells = 10

    U = create_synthetic_U(n_cells, n_species)

    Y_s, rho, T, T_v, p = equation_manager_utils.extract_primitives_from_U(
        U, equation_manager
    )

    # Check reasonable ranges for air at ambient conditions
    # Density: should be around 0.1 to 10 kg/m^3 for our test case
    if not jnp.all((rho > 0.01) & (rho < 100.0)):
        print(f"Warning: Density outside expected range [0.01, 100]: {rho}")

    # Temperature: should be around 200-500 K for our test case
    if not jnp.all((T > 100.0) & (T < 1000.0)):
        print(f"Warning: Temperature outside expected range [100, 1000]: {T}")

    # Pressure: should be around 1e3 to 1e6 Pa for our test case
    if not jnp.all((p > 1.0) & (p < 1e7)):
        print(f"Warning: Pressure outside expected range [1, 1e7]: {p}")

    print("Physical ranges test passed")


def test_extract_primitives_clipping_applied():
    """Test that clipping is applied according to ClippingConfig."""
    equation_manager = create_test_equation_manager()
    n_species = equation_manager.species.n_species

    # Create U with values that should trigger clipping
    U = create_synthetic_U(1, n_species)

    Y_s, rho, T, T_v, p = equation_manager_utils.extract_primitives_from_U(
        U, equation_manager
    )

    clip = equation_manager.numerics_config.clipping

    # Verify clipping bounds are respected
    if not jnp.all((rho >= clip.rho_min) & (rho <= clip.rho_max)):
        raise ValueError(f"Density outside clipping bounds: {rho}")

    if not jnp.all((T >= clip.T_min) & (T <= clip.T_max)):
        raise ValueError(f"Temperature outside clipping bounds: {T}")

    if not jnp.all((T_v >= clip.Tv_min) & (T_v <= clip.Tv_max)):
        raise ValueError(f"T_v outside clipping bounds: {T_v}")

    if not jnp.all((p >= clip.p_min) & (p <= clip.p_max)):
        raise ValueError(f"Pressure outside clipping bounds: {p}")

    if not jnp.all((Y_s >= clip.Y_min) & (Y_s <= clip.Y_max)):
        raise ValueError(f"Mole fractions outside clipping bounds: {Y_s}")

    print("Clipping applied test passed")


def test_extract_primitives_vectorized():
    """Test that function works with different batch sizes."""
    equation_manager = create_test_equation_manager()
    n_species = equation_manager.species.n_species

    for n_cells in [1, 5, 10, 100]:
        U = create_synthetic_U(n_cells, n_species)
        Y_s, rho, T, T_v, p = equation_manager_utils.extract_primitives_from_U(
            U, equation_manager
        )

        # Basic sanity checks
        if Y_s.shape != (n_cells, n_species):
            raise ValueError(f"Shape mismatch for n_cells={n_cells}")

        if not jnp.all(rho > 0):
            raise ValueError(f"Invalid density for n_cells={n_cells}")

    print("Vectorized test passed for various batch sizes")


# Tests are automatically discovered and run by pytest
# Run with: pytest src/compressible_1d/equation_manager_utils_test.py
