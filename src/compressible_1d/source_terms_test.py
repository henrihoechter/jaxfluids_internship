"""Unit tests for source_terms.py"""

import jax
import jax.numpy as jnp
from pathlib import Path

from compressible_1d import source_terms
from compressible_1d import equation_manager_types, numerics_types
from compressible_1d.chemistry_utils import load_species_table_from_gnoffo

# Configure JAX for testing
jax.config.update("jax_enable_x64", True)

# Load test data
data_dir = Path(__file__).parent.parent.parent / "data"
general_data = str(data_dir / "air_5_gnoffo.json")
enthalpy_data = str(data_dir / "air_5_gnoffo_equilibrium_enthalpy.json")


def create_test_equation_manager():
    """Create test EquationManager."""
    species_table = load_species_table_from_gnoffo(general_data, enthalpy_data)

    numerics_config = numerics_types.NumericsConfig(
        dt=1e-6,
        dx=1e-3,
        integrator_scheme="forward-euler",
        spatial_scheme="first_order",
        flux_scheme="hllc",
        n_halo_cells=1,
        clipping=numerics_types.ClippingConfig(),
    )

    return equation_manager_types.EquationManager(
        species=species_table,
        collision_integrals=None,
        reactions=None,
        numerics_config=numerics_config,
        boundary_condition="periodic",
    )


def create_equilibrium_state(n_cells, n_species, T=300.0):
    """Create thermal equilibrium state (T = T_v)."""
    rho_total = 1.225  # kg/m^3
    rho_s = jnp.full((n_cells, n_species), rho_total / n_species)
    u = jnp.zeros(n_cells)
    rho_u = rho_total * u

    # Rough energy estimate for 300K air
    c_v_tr_air = 717.5  # J/(kg·K)
    e_tr = c_v_tr_air * T
    e_v = jnp.zeros(n_cells)  # Equilibrium: minimal vibrational energy
    e_total = e_tr + e_v

    rho_E = rho_total * (e_total + 0.5 * u**2)
    rho_Ev = rho_total * e_v

    U = jnp.concatenate(
        [rho_s, rho_u[:, None], rho_E[:, None], rho_Ev[:, None]], axis=1
    )

    return U


def test_compute_source_terms_shape():
    """Test that source terms have correct output shape."""
    equation_manager = create_test_equation_manager()
    n_species = equation_manager.species.n_species
    n_cells = 10
    n_variables = n_species + 3

    U = create_equilibrium_state(n_cells, n_species)

    S = source_terms.compute_source_terms(U, equation_manager)

    expected_shape = (n_cells, n_variables)
    assert S.shape == expected_shape, f"Expected shape {expected_shape}, got {S.shape}"

    print(f"✓ Source terms shape test passed: {S.shape}")


def test_frozen_chemistry_species_source_zero():
    """Test that species source terms are zero for frozen chemistry."""
    equation_manager = create_test_equation_manager()
    n_species = equation_manager.species.n_species
    n_cells = 10

    U = create_equilibrium_state(n_cells, n_species)

    S = source_terms.compute_source_terms(U, equation_manager)

    # Species source terms should be zero (frozen chemistry)
    species_source = S[:, :n_species]
    assert jnp.allclose(
        species_source, 0.0
    ), "Species source should be zero for frozen chemistry"

    print("✓ Frozen chemistry species source test passed")


def test_equilibrium_vibrational_source_zero():
    """Test that Q_v = 0 when T = T_v (thermal equilibrium)."""
    equation_manager = create_test_equation_manager()
    n_species = equation_manager.species.n_species
    n_cells = 10

    U = create_equilibrium_state(n_cells, n_species, T=300.0)

    S = source_terms.compute_source_terms(U, equation_manager)

    # Vibrational relaxation should be near zero at equilibrium
    Q_v = S[:, n_species + 2]

    # Allow some tolerance due to numerical precision
    assert jnp.allclose(
        Q_v, 0.0, atol=1e-6
    ), f"Vibrational source should be ~0 at equilibrium, got max |Q_v| = {jnp.max(jnp.abs(Q_v)):.2e}"

    print("✓ Equilibrium vibrational source test passed")


def test_chemical_source_zero():
    """Test that chemical source is zero for frozen chemistry."""
    equation_manager = create_test_equation_manager()
    n_species = equation_manager.species.n_species
    n_cells = 10

    U = create_equilibrium_state(n_cells, n_species)

    S_chem = source_terms.compute_chemical_source(U, equation_manager)

    assert jnp.allclose(
        S_chem, 0.0
    ), "Chemical source should be zero for frozen chemistry"

    print("✓ Chemical source zero test passed")


def test_relaxation_time_positive():
    """Test that relaxation time is positive."""
    equation_manager = create_test_equation_manager()
    n_species = equation_manager.species.n_species
    n_cells = 10

    U = create_equilibrium_state(n_cells, n_species)

    # Extract primitives
    from compressible_1d import equation_manager_utils

    Y_s, rho, T, T_v, p = equation_manager_utils.extract_primitives_from_U(
        U, equation_manager
    )

    tau_v = source_terms.compute_relaxation_time(Y_s, rho, T, T_v, p, equation_manager)

    assert jnp.all(tau_v > 0), "Relaxation time must be positive"

    print(f"✓ Relaxation time test passed: τ_v ≈ {jnp.mean(tau_v):.2e} s")


def test_source_terms_jit_compilation():
    """Test that source term computation can be JIT compiled."""
    equation_manager = create_test_equation_manager()
    n_species = equation_manager.species.n_species
    n_cells = 5

    U = create_equilibrium_state(n_cells, n_species)

    # JIT compile
    @jax.jit
    def compute_source_jit(U):
        return source_terms.compute_source_terms(U, equation_manager)

    # First call (compilation)
    S1 = compute_source_jit(U)

    # Second call (use compiled version)
    S2 = compute_source_jit(U)

    # Results should be identical
    assert jnp.allclose(S1, S2), "JIT compilation should not change results"

    print("✓ Source terms JIT compilation test passed")


def test_momentum_source_zero():
    """Test that momentum source is zero (inviscid)."""
    equation_manager = create_test_equation_manager()
    n_species = equation_manager.species.n_species
    n_cells = 10

    U = create_equilibrium_state(n_cells, n_species)

    S = source_terms.compute_source_terms(U, equation_manager)

    # Momentum source should be zero (inviscid)
    momentum_source = S[:, n_species]
    assert jnp.allclose(
        momentum_source, 0.0
    ), "Momentum source should be zero for inviscid"

    print("✓ Momentum source zero test passed")


def test_total_energy_source_zero():
    """Test that total energy source is zero (inviscid, frozen chemistry)."""
    equation_manager = create_test_equation_manager()
    n_species = equation_manager.species.n_species
    n_cells = 10

    U = create_equilibrium_state(n_cells, n_species)

    S = source_terms.compute_source_terms(U, equation_manager)

    # Total energy source should be zero (inviscid, no chemistry)
    energy_source = S[:, n_species + 1]
    assert jnp.allclose(energy_source, 0.0), "Total energy source should be zero"

    print("✓ Total energy source zero test passed")
