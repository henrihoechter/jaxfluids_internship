"""Integration tests for viscous_flux.py - viscous flux computation.

Tests gradient computation, flux assembly, and integration with equation manager.
"""

import jax
import jax.numpy as jnp
import pytest
from pathlib import Path

from compressible_1d import viscous_flux
from compressible_1d import transport
from compressible_1d import equation_manager
from compressible_1d import equation_manager_types
from compressible_1d import numerics_types
from compressible_1d.chemistry_utils import load_species_table

# Configure JAX for testing
jax.config.update("jax_enable_x64", True)

# Test data paths
DATA_DIR = Path(__file__).parent.parent.parent / "data"
GENERAL_DATA = str(DATA_DIR / "air_5_gnoffo.json")
ENTHALPY_DATA = str(DATA_DIR / "air_5_gnoffo_equilibrium_enthalpy.json")
COLLISION_INTEGRALS_FILE = str(DATA_DIR / "collision_integrals_tp2867.json")


@pytest.fixture
def species_table():
    """Load species table for testing."""
    return load_species_table(GENERAL_DATA, ENTHALPY_DATA)


@pytest.fixture
def collision_integrals():
    """Load collision integrals for testing."""
    return transport.create_collision_integral_table_from_json(COLLISION_INTEGRALS_FILE)


@pytest.fixture
def numerics_config():
    """Create numerics config for testing."""
    return numerics_types.NumericsConfig(
        dt=1e-8,
        dx=1e-3,
        integrator_scheme="forward-euler",
        spatial_scheme="first_order",
        flux_scheme="hllc",
        n_halo_cells=1,
        clipping=numerics_types.ClippingConfig(),
    )


@pytest.fixture
def equation_manager_inviscid(species_table, numerics_config):
    """Create inviscid equation manager (no collision integrals)."""
    return equation_manager_types.EquationManager(
        species=species_table,
        collision_integrals=None,
        reactions=None,
        numerics_config=numerics_config,
        boundary_condition="periodic",
    )


@pytest.fixture
def equation_manager_viscous(species_table, collision_integrals, numerics_config):
    """Create viscous equation manager (with collision integrals)."""
    return equation_manager_types.EquationManager(
        species=species_table,
        collision_integrals=collision_integrals,
        reactions=None,
        numerics_config=numerics_config,
        boundary_condition="periodic",
    )


def create_test_state(n_cells, n_species, rho_total=1.225, T=3000.0, u=0.0):
    """Create test conserved state.

    Args:
        n_cells: Number of cells
        n_species: Number of species
        rho_total: Total density [kg/m³]
        T: Temperature [K]
        u: Velocity [m/s]

    Returns:
        U: Conserved state [n_cells, n_variables]
    """
    rho_s = jnp.full((n_cells, n_species), rho_total / n_species)
    rho_u = jnp.full(n_cells, rho_total * u)

    # Approximate internal energy for high-temperature air
    c_v_tr_air = 717.5  # J/(kg·K)
    e_tr = c_v_tr_air * T
    e_v = 0.1 * e_tr  # Small vibrational energy
    e_total = e_tr + e_v

    rho_E = jnp.full(n_cells, rho_total * (e_total + 0.5 * u**2))
    rho_Ev = jnp.full(n_cells, rho_total * e_v)

    U = jnp.concatenate(
        [rho_s, rho_u[:, None], rho_E[:, None], rho_Ev[:, None]], axis=1
    )

    return U


class TestGradientComputation:
    """Test gradient computation at interfaces."""

    def test_gradient_constant_field(self):
        """Gradient of constant field should be zero."""
        phi = jnp.ones(10) * 5.0
        dx = 0.1

        grad = viscous_flux.compute_gradients_at_interfaces(phi, dx)

        assert grad.shape == (9,)
        assert jnp.allclose(grad, 0.0)

    def test_gradient_linear_field(self):
        """Gradient of linear field should be constant."""
        x = jnp.linspace(0, 1, 11)
        phi = 2.0 * x + 3.0  # Linear: phi = 2x + 3
        dx = 0.1

        grad = viscous_flux.compute_gradients_at_interfaces(phi, dx)

        assert grad.shape == (10,)
        assert jnp.allclose(grad, 2.0, rtol=1e-10)

    def test_gradient_multispecies(self):
        """Test gradient computation for multi-species field."""
        n_cells = 10
        n_species = 5
        phi = jnp.ones((n_cells, n_species))
        # Add linear variation to first species
        phi = phi.at[:, 0].set(jnp.linspace(0, 1, n_cells))
        dx = 1.0 / (n_cells - 1)

        grad = viscous_flux.compute_gradients_at_interfaces_multispecies(phi, dx)

        assert grad.shape == (n_cells - 1, n_species)
        # First species has unit gradient
        assert jnp.allclose(grad[:, 0], 1.0, rtol=0.1)
        # Other species have zero gradient
        assert jnp.allclose(grad[:, 1:], 0.0)


class TestInterfaceValues:
    """Test interface value averaging."""

    def test_interface_averaging(self):
        """Test simple averaging at interfaces."""
        phi = jnp.array([1.0, 3.0, 5.0, 7.0])

        phi_face = viscous_flux.compute_interface_values(phi)

        expected = jnp.array([2.0, 4.0, 6.0])
        assert jnp.allclose(phi_face, expected)

    def test_interface_multispecies(self):
        """Test interface averaging for multi-species."""
        phi = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        phi_face = viscous_flux.compute_interface_values_multispecies(phi)

        expected = jnp.array([[2.0, 3.0], [4.0, 5.0]])
        assert jnp.allclose(phi_face, expected)


class TestViscousStress:
    """Test viscous stress computation."""

    def test_stress_zero_gradient(self):
        """Zero velocity gradient gives zero stress."""
        du_dx = jnp.zeros(5)
        mu = jnp.ones(5) * 1e-5

        tau = viscous_flux.compute_viscous_stress_1d(du_dx, mu)

        assert jnp.allclose(tau, 0.0)

    def test_stress_positive(self):
        """Positive velocity gradient gives positive stress."""
        du_dx = jnp.ones(5) * 100.0  # 100 s^-1
        mu = jnp.ones(5) * 1e-5  # 1e-5 Pa·s

        tau = viscous_flux.compute_viscous_stress_1d(du_dx, mu)

        # τ = (4/3) μ ∂u/∂x = (4/3) * 1e-5 * 100 = 1.33e-3 Pa
        expected = (4.0 / 3.0) * 1e-5 * 100.0
        assert jnp.allclose(tau, expected)


class TestSpeciesDiffusionFlux:
    """Test species diffusion flux computation."""

    def test_diffusion_zero_gradient(self):
        """Zero concentration gradient gives zero diffusion flux."""
        n_interfaces = 5
        n_species = 3
        rho = jnp.ones(n_interfaces) * 1.0
        D_s = jnp.ones((n_interfaces, n_species)) * 1e-4
        dc_s_dx = jnp.zeros((n_interfaces, n_species))

        j_s = viscous_flux.compute_species_diffusion_flux(rho, D_s, dc_s_dx)

        assert j_s.shape == (n_interfaces, n_species)
        assert jnp.allclose(j_s, 0.0)

    def test_diffusion_direction(self):
        """Diffusion flux is opposite to concentration gradient."""
        n_interfaces = 3
        n_species = 2
        rho = jnp.ones(n_interfaces) * 1.0
        D_s = jnp.ones((n_interfaces, n_species)) * 1e-4
        # Positive gradient for species 0
        dc_s_dx = jnp.zeros((n_interfaces, n_species))
        dc_s_dx = dc_s_dx.at[:, 0].set(1.0)

        j_s = viscous_flux.compute_species_diffusion_flux(rho, D_s, dc_s_dx)

        # Flux should be negative (diffusion down the gradient)
        assert jnp.all(j_s[:, 0] < 0)


class TestHeatFlux:
    """Test heat flux computation."""

    def test_heat_flux_zero_gradient(self):
        """Zero temperature gradient gives zero heat flux."""
        n_interfaces = 5
        eta_t = jnp.ones(n_interfaces) * 0.1
        eta_r = jnp.ones(n_interfaces) * 0.05
        eta_v = jnp.ones(n_interfaces) * 0.02
        dT_dx = jnp.zeros(n_interfaces)
        dTv_dx = jnp.zeros(n_interfaces)

        q_tr, q_v = viscous_flux.compute_heat_flux(eta_t, eta_r, eta_v, dT_dx, dTv_dx)

        assert jnp.allclose(q_tr, 0.0)
        assert jnp.allclose(q_v, 0.0)

    def test_heat_flux_direction(self):
        """Heat flux is opposite to temperature gradient."""
        n_interfaces = 3
        eta_t = jnp.ones(n_interfaces) * 0.1
        eta_r = jnp.ones(n_interfaces) * 0.05
        eta_v = jnp.ones(n_interfaces) * 0.02
        dT_dx = jnp.ones(n_interfaces) * 100.0  # 100 K/m
        dTv_dx = jnp.ones(n_interfaces) * 50.0  # 50 K/m

        q_tr, q_v = viscous_flux.compute_heat_flux(eta_t, eta_r, eta_v, dT_dx, dTv_dx)

        # Heat flows from hot to cold (opposite to gradient)
        assert jnp.all(q_tr < 0)
        assert jnp.all(q_v < 0)


class TestViscousFluxComputation:
    """Test full viscous flux computation."""

    def test_inviscid_returns_zeros(self, equation_manager_inviscid):
        """Inviscid case (no collision integrals) should return zero flux."""
        n_species = equation_manager_inviscid.species.n_species
        n_cells = 10
        n_halo = equation_manager_inviscid.numerics_config.n_halo_cells
        n_cells_with_halo = n_cells + 2 * n_halo

        U = create_test_state(n_cells_with_halo, n_species)

        F_v = viscous_flux.compute_viscous_flux(U, equation_manager_inviscid)

        n_interfaces = n_cells_with_halo - 1
        n_variables = n_species + 3
        assert F_v.shape == (n_interfaces, n_variables)
        assert jnp.allclose(F_v, 0.0)

    def test_viscous_flux_shape(self, equation_manager_viscous):
        """Test viscous flux has correct shape."""
        n_species = equation_manager_viscous.species.n_species
        n_cells = 10
        n_halo = equation_manager_viscous.numerics_config.n_halo_cells
        n_cells_with_halo = n_cells + 2 * n_halo

        U = create_test_state(n_cells_with_halo, n_species)

        F_v = viscous_flux.compute_viscous_flux(U, equation_manager_viscous)

        n_interfaces = n_cells_with_halo - 1
        n_variables = n_species + 3
        assert F_v.shape == (n_interfaces, n_variables)

    def test_uniform_state_zero_flux(self, equation_manager_viscous):
        """Uniform state should have approximately zero viscous flux."""
        n_species = equation_manager_viscous.species.n_species
        n_cells = 10
        n_halo = equation_manager_viscous.numerics_config.n_halo_cells
        n_cells_with_halo = n_cells + 2 * n_halo

        # Uniform state
        U = create_test_state(n_cells_with_halo, n_species, u=0.0)

        F_v = viscous_flux.compute_viscous_flux(U, equation_manager_viscous)

        # For uniform state, all gradients are zero, so flux should be near zero
        # (allowing for numerical precision)
        assert jnp.allclose(F_v, 0.0, atol=1e-10)


class TestDiffusiveCFL:
    """Test diffusive CFL condition."""

    def test_cfl_positive(self):
        """Diffusive CFL timestep must be positive."""
        n_cells = 5
        n_species = 3
        rho = jnp.ones(n_cells) * 1.0
        mu = jnp.ones(n_cells) * 1e-5
        eta = jnp.ones(n_cells) * 0.1
        D_s = jnp.ones((n_cells, n_species)) * 1e-4
        cv = jnp.ones(n_cells) * 1000.0
        dx = 1e-3

        dt_diff = viscous_flux.compute_diffusive_cfl(rho, mu, eta, D_s, cv, dx)

        assert float(dt_diff) > 0

    def test_cfl_scales_with_dx_squared(self):
        """Diffusive CFL should scale with dx²."""
        n_cells = 5
        n_species = 3
        rho = jnp.ones(n_cells) * 1.0
        mu = jnp.ones(n_cells) * 1e-5
        eta = jnp.ones(n_cells) * 0.1
        D_s = jnp.ones((n_cells, n_species)) * 1e-4
        cv = jnp.ones(n_cells) * 1000.0

        dx1 = 1e-3
        dx2 = 2e-3

        dt1 = viscous_flux.compute_diffusive_cfl(rho, mu, eta, D_s, cv, dx1)
        dt2 = viscous_flux.compute_diffusive_cfl(rho, mu, eta, D_s, cv, dx2)

        # dt ~ dx², so dt2/dt1 ~ (dx2/dx1)² = 4
        ratio = float(dt2) / float(dt1)
        assert jnp.isclose(ratio, 4.0, rtol=0.1)


class TestEquationManagerIntegration:
    """Test integration with equation manager."""

    def test_compute_dU_dt_diffusive_inviscid(self, equation_manager_inviscid):
        """Inviscid dU/dt should be zero."""
        n_species = equation_manager_inviscid.species.n_species
        n_cells = 10
        n_halo = equation_manager_inviscid.numerics_config.n_halo_cells
        n_cells_with_halo = n_cells + 2 * n_halo

        U = create_test_state(n_cells_with_halo, n_species)

        dU_dt = equation_manager.compute_dU_dt_diffusive(U, equation_manager_inviscid)

        assert dU_dt.shape == (n_cells, n_species + 3)
        assert jnp.allclose(dU_dt, 0.0)

    def test_compute_dU_dt_diffusive_viscous_shape(self, equation_manager_viscous):
        """Viscous dU/dt should have correct shape."""
        n_species = equation_manager_viscous.species.n_species
        n_cells = 10
        n_halo = equation_manager_viscous.numerics_config.n_halo_cells
        n_cells_with_halo = n_cells + 2 * n_halo

        U = create_test_state(n_cells_with_halo, n_species)

        dU_dt = equation_manager.compute_dU_dt_diffusive(U, equation_manager_viscous)

        n_variables = n_species + 3
        assert dU_dt.shape == (n_cells, n_variables)

    def test_check_diffusive_cfl_inviscid(self, equation_manager_inviscid):
        """CFL check should return None for inviscid case."""
        n_species = equation_manager_inviscid.species.n_species
        n_cells = 10
        U = create_test_state(n_cells, n_species)

        result = equation_manager.check_diffusive_cfl(U, equation_manager_inviscid)

        assert result is None

    def test_check_diffusive_cfl_viscous(self, equation_manager_viscous):
        """CFL check should return positive value for viscous case."""
        n_species = equation_manager_viscous.species.n_species
        n_cells = 10
        U = create_test_state(n_cells, n_species)

        result = equation_manager.check_diffusive_cfl(U, equation_manager_viscous)

        assert result is not None
        assert result > 0


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
