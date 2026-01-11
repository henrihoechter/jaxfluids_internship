"""Unit tests for equation_manager.py"""

import jax
import jax.numpy as jnp
from pathlib import Path

from compressible_1d import equation_manager
from compressible_1d import equation_manager_types, numerics_types
from compressible_1d.chemistry_utils import load_species_table

# Configure JAX for testing
jax.config.update("jax_enable_x64", True)

# Load test data
data_dir = Path(__file__).parent.parent.parent / "data"
general_data = str(data_dir / "air_5_gnoffo.json")
enthalpy_data = str(data_dir / "air_5_gnoffo_equilibrium_enthalpy.json")


def create_test_equation_manager(dt=1e-7, integrator="forward-euler"):
    """Create test EquationManager."""
    species_table = load_species_table(general_data, enthalpy_data)

    numerics_config = numerics_types.NumericsConfig(
        dt=dt,
        dx=1e-3,
        integrator_scheme=integrator,
        spatial_scheme="first_order",
        flux_scheme="hllc",
        n_halo_cells=1,
        clipping=numerics_types.ClippingConfig(),
    )

    return equation_manager_types.EquationManager(
        species=species_table,
        collision_integrals=None,  # Inviscid case
        reactions=None,
        numerics_config=numerics_config,
        boundary_condition="periodic",
    )


def create_test_state(n_cells, n_species):
    """Create test conserved state."""
    rho_total = 1.225  # kg/m^3
    rho_s = jnp.full((n_cells, n_species), rho_total / n_species)
    u = jnp.zeros(n_cells)
    rho_u = rho_total * u

    c_v_tr_air = 717.5  # J/(kg·K)
    T = 300.0
    e_tr = c_v_tr_air * T
    e_v = jnp.zeros(n_cells)
    e_total = e_tr + e_v

    rho_E = rho_total * (e_total + 0.5 * u**2)
    rho_Ev = rho_total * e_v

    U = jnp.concatenate(
        [rho_s, rho_u[:, None], rho_E[:, None], rho_Ev[:, None]], axis=1
    )

    return U


def test_advance_one_step_output_shape():
    """Test that advance_one_step returns correct shape."""
    eq_manager = create_test_equation_manager()
    n_species = eq_manager.species.n_species
    n_cells = 10

    U = create_test_state(n_cells, n_species)

    U_new = equation_manager.advance_one_step(U, eq_manager)

    assert U_new.shape == U.shape, f"Expected shape {U.shape}, got {U_new.shape}"

    print(f"advance_one_step output shape test passed: {U_new.shape}")


def test_advance_one_step_rk2_output_shape():
    """Test that advance_one_step with RK2 returns correct shape."""
    eq_manager = create_test_equation_manager(integrator="rk2")
    n_species = eq_manager.species.n_species
    n_cells = 10

    U = create_test_state(n_cells, n_species)

    U_new = equation_manager.advance_one_step(U, eq_manager)

    assert U_new.shape == U.shape, f"Expected shape {U.shape}, got {U_new.shape}"

    print(f"advance_one_step (RK2) output shape test passed: {U_new.shape}")


def test_uniform_flow_conservation():
    """Test that uniform flow is preserved (no flux, no source)."""
    eq_manager = create_test_equation_manager()
    n_species = eq_manager.species.n_species
    n_cells = 10

    U = create_test_state(n_cells, n_species)

    # Run a few steps
    for _ in range(5):
        U = equation_manager.advance_one_step(U, eq_manager)

    # Check that solution hasn't changed much (uniform flow should be steady)
    U_expected = create_test_state(n_cells, n_species)

    # Allow some tolerance due to numerical diffusion
    assert jnp.allclose(
        U, U_expected, rtol=1e-6
    ), "Uniform flow should be approximately preserved"

    print("Uniform flow conservation test passed")


def test_run_output_shapes():
    """Test that run() returns correct output shapes."""
    eq_manager = create_test_equation_manager()
    n_species = eq_manager.species.n_species
    n_cells = 10

    U_init = create_test_state(n_cells, n_species)
    t_final = 1e-6  # 1 microsecond
    save_interval = 5

    U_history, t_history = equation_manager.run(
        U_init, eq_manager, t_final, save_interval
    )

    # Check shapes
    n_steps = int(t_final / eq_manager.numerics_config.dt)
    n_snapshots = n_steps // save_interval + 1

    expected_U_shape = (n_snapshots, n_cells, n_species + 3)
    expected_t_shape = (n_snapshots,)

    assert (
        U_history.shape == expected_U_shape
    ), f"Expected U_history shape {expected_U_shape}, got {U_history.shape}"
    assert (
        t_history.shape == expected_t_shape
    ), f"Expected t_history shape {expected_t_shape}, got {t_history.shape}"

    print(f"Run output shapes test passed: U={U_history.shape}, t={t_history.shape}")


def test_run_time_monotonic():
    """Test that time increases monotonically."""
    eq_manager = create_test_equation_manager()
    n_species = eq_manager.species.n_species
    n_cells = 10

    U_init = create_test_state(n_cells, n_species)
    t_final = 1e-6
    save_interval = 5

    U_history, t_history = equation_manager.run(
        U_init, eq_manager, t_final, save_interval
    )

    # Check that time is monotonically increasing
    dt_history = jnp.diff(t_history)
    assert jnp.all(dt_history > 0), "Time should increase monotonically"

    # Check that final time is approximately t_final
    assert jnp.isclose(
        t_history[-1], t_final, rtol=1e-3
    ), f"Final time should be approx {t_final}, got {t_history[-1]}"

    print(f"Run time monotonic test passed: t_final = {t_history[-1]:.2e} s")


def test_advance_one_step_jit_compilation():
    """Test that advance_one_step can be JIT compiled."""
    eq_manager = create_test_equation_manager()
    n_species = eq_manager.species.n_species
    n_cells = 10

    U = create_test_state(n_cells, n_species)

    @jax.jit
    def step_jit(U):
        return equation_manager.advance_one_step(U, eq_manager)

    # First call (compilation)
    U1 = step_jit(U)

    # Second call (use compiled version)
    U2 = step_jit(U)

    # Results should be identical
    assert jnp.allclose(U1, U2), "JIT compilation should not change results"

    print("advance_one_step JIT compilation test passed")


def test_mass_conservation():
    """Test that total mass is conserved during time integration."""
    eq_manager = create_test_equation_manager()
    n_species = eq_manager.species.n_species
    n_cells = 10

    U = create_test_state(n_cells, n_species)

    # Initial total mass
    rho_s_init = U[:, :n_species]
    mass_init = jnp.sum(rho_s_init) * eq_manager.numerics_config.dx

    # Run simulation
    for _ in range(10):
        U = equation_manager.advance_one_step(U, eq_manager)

    # Final total mass
    rho_s_final = U[:, :n_species]
    mass_final = jnp.sum(rho_s_final) * eq_manager.numerics_config.dx

    # Mass should be conserved
    rel_error = jnp.abs(mass_final - mass_init) / mass_init
    assert rel_error < 1e-10, f"Mass conservation error: {rel_error:.2e}"

    print(f"Mass conservation test passed: rel_error = {rel_error:.2e}")


def test_periodic_bc_consistency():
    """Test that periodic boundary conditions maintain consistency."""
    eq_manager = create_test_equation_manager()
    n_species = eq_manager.species.n_species
    n_cells = 10

    # Create state with a pattern
    U = create_test_state(n_cells, n_species)
    # Add a spatial variation
    x = jnp.linspace(0, 1, n_cells)
    U = U.at[:, n_species].add(0.1 * jnp.sin(2 * jnp.pi * x))  # Add velocity variation

    # Run simulation with periodic BCs
    for _ in range(10):
        U = equation_manager.advance_one_step(U, eq_manager)

    # With periodic BCs, solution should maintain some structure
    # Just check that it didn't blow up
    assert jnp.all(jnp.isfinite(U)), "Solution should remain finite with periodic BCs"

    print("Periodic BC consistency test passed")


# Tests are automatically discovered and run by pytest
# Run with: pytest src/compressible_1d/equation_manager_test.py
