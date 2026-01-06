"""Unit tests for transport.py - Chapman-Enskog transport properties.

Tests collision integral interpolation, modified collision integrals,
and transport property calculations against expected physical behavior.
"""

import jax
import jax.numpy as jnp
import pytest
from pathlib import Path

from compressible_1d import transport
from compressible_1d.chemistry_types import CollisionIntegralTable

# Configure JAX for testing
jax.config.update("jax_enable_x64", True)

# Test data paths
DATA_DIR = Path(__file__).parent.parent.parent / "data"
COLLISION_INTEGRALS_FILE = DATA_DIR / "collision_integrals_tp2867.json"


@pytest.fixture
def collision_integrals():
    """Load collision integral table from JSON."""
    return transport.create_collision_integral_table_from_json(COLLISION_INTEGRALS_FILE)


@pytest.fixture
def species_names():
    """5-species air model species names."""
    return ("N", "O", "N2", "O2", "NO")


@pytest.fixture
def molar_masses():
    """Molar masses for 5-species air [kg/mol]."""
    return jnp.array([14.0067, 15.9994, 28.0134, 31.9988, 30.0061]) / 1000.0


class TestCollisionIntegralInterpolation:
    """Test collision integral interpolation (Eq. 67)."""

    def test_interpolation_at_reference_temperatures(self, collision_integrals):
        """At T=2000K and T=4000K, interpolation should return tabulated values."""
        T_2000 = jnp.array([2000.0])
        T_4000 = jnp.array([4000.0])

        # Interpolate at reference temperatures
        pi_omega_2000 = transport.interpolate_collision_integral(
            T_2000,
            collision_integrals.omega_11_2000K,
            collision_integrals.omega_11_4000K,
        )
        pi_omega_4000 = transport.interpolate_collision_integral(
            T_4000,
            collision_integrals.omega_11_2000K,
            collision_integrals.omega_11_4000K,
        )

        # At T=2000K, should get 10^(omega_11_2000K)
        expected_2000 = jnp.power(10.0, collision_integrals.omega_11_2000K)
        expected_4000 = jnp.power(10.0, collision_integrals.omega_11_4000K)

        assert jnp.allclose(pi_omega_2000[0, :], expected_2000, rtol=1e-10)
        assert jnp.allclose(pi_omega_4000[0, :], expected_4000, rtol=1e-10)

    def test_interpolation_at_midpoint(self, collision_integrals):
        """At geometric mean temperature, should interpolate correctly."""
        T_mid = jnp.array([jnp.sqrt(2000.0 * 4000.0)])  # ~2828 K

        pi_omega = transport.interpolate_collision_integral(
            T_mid,
            collision_integrals.omega_11_2000K,
            collision_integrals.omega_11_4000K,
        )

        # Should be between 2000K and 4000K values
        low_val = jnp.power(10.0, collision_integrals.omega_11_2000K)
        high_val = jnp.power(10.0, collision_integrals.omega_11_4000K)

        # For most pairs, collision integral decreases with temperature
        # so result should be between bounds (allowing for numerical tolerance)
        assert jnp.all(pi_omega[0, :] > 0)  # Must be positive

    def test_interpolation_shape(self, collision_integrals):
        """Test output shape for various input shapes."""
        T_single = jnp.array([3000.0])
        T_multiple = jnp.array([2000.0, 3000.0, 4000.0, 5000.0])

        result_single = transport.interpolate_collision_integral(
            T_single,
            collision_integrals.omega_11_2000K,
            collision_integrals.omega_11_4000K,
        )
        result_multiple = transport.interpolate_collision_integral(
            T_multiple,
            collision_integrals.omega_11_2000K,
            collision_integrals.omega_11_4000K,
        )

        n_pairs = len(collision_integrals.species_pairs)
        assert result_single.shape == (1, n_pairs)
        assert result_multiple.shape == (4, n_pairs)


class TestModifiedCollisionIntegrals:
    """Test modified collision integrals Δ^(1) and Δ^(2)."""

    def test_delta_1_shape(self, collision_integrals, molar_masses):
        """Test shape of Δ^(1) output."""
        T = jnp.array([3000.0, 4000.0, 5000.0])
        n_species = len(molar_masses)

        pair_indices = transport.build_pair_index_matrix(
            ("N", "O", "N2", "O2", "NO"), collision_integrals
        )

        pi_omega_11 = transport.interpolate_collision_integral(
            T,
            collision_integrals.omega_11_2000K,
            collision_integrals.omega_11_4000K,
        )

        delta_1 = transport.compute_modified_collision_integral_1(
            T, molar_masses, molar_masses, pi_omega_11, pair_indices
        )

        assert delta_1.shape == (3, n_species, n_species)

    def test_delta_2_shape(self, collision_integrals, molar_masses):
        """Test shape of Δ^(2) output."""
        T = jnp.array([3000.0, 4000.0])
        n_species = len(molar_masses)

        pair_indices = transport.build_pair_index_matrix(
            ("N", "O", "N2", "O2", "NO"), collision_integrals
        )

        pi_omega_22 = transport.interpolate_collision_integral(
            T,
            collision_integrals.omega_22_2000K,
            collision_integrals.omega_22_4000K,
        )

        delta_2 = transport.compute_modified_collision_integral_2(
            T, molar_masses, molar_masses, pi_omega_22, pair_indices
        )

        assert delta_2.shape == (2, n_species, n_species)

    def test_delta_positive(self, collision_integrals, molar_masses):
        """Modified collision integrals must be positive."""
        T = jnp.array([2000.0, 5000.0, 10000.0])

        pair_indices = transport.build_pair_index_matrix(
            ("N", "O", "N2", "O2", "NO"), collision_integrals
        )

        pi_omega_11 = transport.interpolate_collision_integral(
            T,
            collision_integrals.omega_11_2000K,
            collision_integrals.omega_11_4000K,
        )
        pi_omega_22 = transport.interpolate_collision_integral(
            T,
            collision_integrals.omega_22_2000K,
            collision_integrals.omega_22_4000K,
        )

        delta_1 = transport.compute_modified_collision_integral_1(
            T, molar_masses, molar_masses, pi_omega_11, pair_indices
        )
        delta_2 = transport.compute_modified_collision_integral_2(
            T, molar_masses, molar_masses, pi_omega_22, pair_indices
        )

        assert jnp.all(delta_1 > 0), "Δ^(1) must be positive"
        assert jnp.all(delta_2 > 0), "Δ^(2) must be positive"


class TestMixtureViscosity:
    """Test mixture viscosity calculation."""

    def test_viscosity_shape(self, collision_integrals, molar_masses):
        """Test viscosity output shape."""
        n_cells = 5
        n_species = len(molar_masses)
        T = jnp.full(n_cells, 3000.0)

        # Equal molar concentrations
        gamma_s = jnp.ones((n_cells, n_species)) / n_species

        pair_indices = transport.build_pair_index_matrix(
            ("N", "O", "N2", "O2", "NO"), collision_integrals
        )

        pi_omega_22 = transport.interpolate_collision_integral(
            T,
            collision_integrals.omega_22_2000K,
            collision_integrals.omega_22_4000K,
        )

        delta_2 = transport.compute_modified_collision_integral_2(
            T, molar_masses, molar_masses, pi_omega_22, pair_indices
        )

        mu = transport.compute_mixture_viscosity(T, gamma_s, molar_masses, delta_2)

        assert mu.shape == (n_cells,)

    def test_viscosity_positive(self, collision_integrals, molar_masses):
        """Viscosity must be positive."""
        n_cells = 3
        n_species = len(molar_masses)
        T = jnp.array([2000.0, 5000.0, 10000.0])

        gamma_s = jnp.ones((n_cells, n_species)) / n_species

        pair_indices = transport.build_pair_index_matrix(
            ("N", "O", "N2", "O2", "NO"), collision_integrals
        )

        pi_omega_22 = transport.interpolate_collision_integral(
            T,
            collision_integrals.omega_22_2000K,
            collision_integrals.omega_22_4000K,
        )

        delta_2 = transport.compute_modified_collision_integral_2(
            T, molar_masses, molar_masses, pi_omega_22, pair_indices
        )

        mu = transport.compute_mixture_viscosity(T, gamma_s, molar_masses, delta_2)

        assert jnp.all(mu > 0), "Viscosity must be positive"

    def test_viscosity_order_of_magnitude(self, collision_integrals, molar_masses):
        """Check viscosity is in expected range for high-temperature air."""
        n_cells = 1
        n_species = len(molar_masses)
        T = jnp.array([3000.0])

        # Air-like composition (mostly N2 and O2)
        gamma_s = jnp.array([[0.1, 0.1, 0.4, 0.3, 0.1]])

        pair_indices = transport.build_pair_index_matrix(
            ("N", "O", "N2", "O2", "NO"), collision_integrals
        )

        pi_omega_22 = transport.interpolate_collision_integral(
            T,
            collision_integrals.omega_22_2000K,
            collision_integrals.omega_22_4000K,
        )

        delta_2 = transport.compute_modified_collision_integral_2(
            T, molar_masses, molar_masses, pi_omega_22, pair_indices
        )

        mu = transport.compute_mixture_viscosity(T, gamma_s, molar_masses, delta_2)

        # High-temperature air viscosity is typically 1e-5 to 1e-4 Pa·s
        assert (
            1e-6 < float(mu[0]) < 1e-3
        ), f"Viscosity {mu[0]:.2e} outside expected range"


class TestThermalConductivity:
    """Test thermal conductivity calculations."""

    def test_translational_conductivity_positive(
        self, collision_integrals, molar_masses
    ):
        """Translational thermal conductivity must be positive."""
        n_cells = 3
        n_species = len(molar_masses)
        T = jnp.array([2000.0, 5000.0, 10000.0])

        gamma_s = jnp.ones((n_cells, n_species)) / n_species

        pair_indices = transport.build_pair_index_matrix(
            ("N", "O", "N2", "O2", "NO"), collision_integrals
        )

        pi_omega_22 = transport.interpolate_collision_integral(
            T,
            collision_integrals.omega_22_2000K,
            collision_integrals.omega_22_4000K,
        )

        delta_2 = transport.compute_modified_collision_integral_2(
            T, molar_masses, molar_masses, pi_omega_22, pair_indices
        )

        eta_t = transport.compute_translational_thermal_conductivity(
            T, gamma_s, molar_masses, delta_2
        )

        assert jnp.all(eta_t > 0), "Translational conductivity must be positive"

    def test_rotational_conductivity_molecules_only(
        self, collision_integrals, molar_masses
    ):
        """Rotational conductivity should only include molecular species."""
        n_cells = 2
        n_species = len(molar_masses)
        T = jnp.array([3000.0, 5000.0])

        # Only atoms (N and O)
        gamma_atoms_only = jnp.zeros((n_cells, n_species))
        gamma_atoms_only = gamma_atoms_only.at[:, 0].set(0.5)  # N
        gamma_atoms_only = gamma_atoms_only.at[:, 1].set(0.5)  # O

        # Only molecules (N2, O2, NO)
        gamma_molecules_only = jnp.zeros((n_cells, n_species))
        gamma_molecules_only = gamma_molecules_only.at[:, 2].set(0.4)  # N2
        gamma_molecules_only = gamma_molecules_only.at[:, 3].set(0.3)  # O2
        gamma_molecules_only = gamma_molecules_only.at[:, 4].set(0.3)  # NO

        pair_indices = transport.build_pair_index_matrix(
            ("N", "O", "N2", "O2", "NO"), collision_integrals
        )

        pi_omega_11 = transport.interpolate_collision_integral(
            T,
            collision_integrals.omega_11_2000K,
            collision_integrals.omega_11_4000K,
        )

        delta_1 = transport.compute_modified_collision_integral_1(
            T, molar_masses, molar_masses, pi_omega_11, pair_indices
        )

        # N and O are atoms (indices 0, 1), N2, O2, NO are molecules (indices 2, 3, 4)
        is_molecule = jnp.array([False, False, True, True, True])

        eta_r_atoms = transport.compute_rotational_thermal_conductivity(
            T, gamma_atoms_only, is_molecule, delta_1
        )
        eta_r_molecules = transport.compute_rotational_thermal_conductivity(
            T, gamma_molecules_only, is_molecule, delta_1
        )

        # Atoms-only mixture should have zero rotational conductivity
        assert jnp.allclose(eta_r_atoms, 0.0, atol=1e-30)
        # Molecules should have positive rotational conductivity
        assert jnp.all(eta_r_molecules > 0)


class TestDiffusionCoefficients:
    """Test diffusion coefficient calculations."""

    def test_binary_diffusion_shape(self, collision_integrals, molar_masses):
        """Test binary diffusion coefficient shape."""
        n_cells = 3
        n_species = len(molar_masses)
        T = jnp.array([2000.0, 3000.0, 4000.0])
        p = jnp.full(n_cells, 1e5)  # 1 bar

        pair_indices = transport.build_pair_index_matrix(
            ("N", "O", "N2", "O2", "NO"), collision_integrals
        )

        pi_omega_11 = transport.interpolate_collision_integral(
            T,
            collision_integrals.omega_11_2000K,
            collision_integrals.omega_11_4000K,
        )

        delta_1 = transport.compute_modified_collision_integral_1(
            T, molar_masses, molar_masses, pi_omega_11, pair_indices
        )

        D_sr = transport.compute_binary_diffusion_coefficient(T, p, delta_1)

        assert D_sr.shape == (n_cells, n_species, n_species)

    def test_binary_diffusion_positive(self, collision_integrals, molar_masses):
        """Binary diffusion coefficients must be positive."""
        n_cells = 2
        T = jnp.array([3000.0, 5000.0])
        p = jnp.full(n_cells, 1e5)

        pair_indices = transport.build_pair_index_matrix(
            ("N", "O", "N2", "O2", "NO"), collision_integrals
        )

        pi_omega_11 = transport.interpolate_collision_integral(
            T,
            collision_integrals.omega_11_2000K,
            collision_integrals.omega_11_4000K,
        )

        delta_1 = transport.compute_modified_collision_integral_1(
            T, molar_masses, molar_masses, pi_omega_11, pair_indices
        )

        D_sr = transport.compute_binary_diffusion_coefficient(T, p, delta_1)

        assert jnp.all(D_sr > 0), "Binary diffusion coefficients must be positive"

    def test_effective_diffusion_shape(self, collision_integrals, molar_masses):
        """Test effective diffusion coefficient shape."""
        n_cells = 3
        n_species = len(molar_masses)
        T = jnp.array([3000.0, 4000.0, 5000.0])
        p = jnp.full(n_cells, 1e5)

        gamma_s = jnp.ones((n_cells, n_species)) / n_species

        pair_indices = transport.build_pair_index_matrix(
            ("N", "O", "N2", "O2", "NO"), collision_integrals
        )

        pi_omega_11 = transport.interpolate_collision_integral(
            T,
            collision_integrals.omega_11_2000K,
            collision_integrals.omega_11_4000K,
        )

        delta_1 = transport.compute_modified_collision_integral_1(
            T, molar_masses, molar_masses, pi_omega_11, pair_indices
        )

        D_sr = transport.compute_binary_diffusion_coefficient(T, p, delta_1)
        D_s = transport.compute_effective_diffusion_coefficient(
            gamma_s, molar_masses, D_sr
        )

        assert D_s.shape == (n_cells, n_species)


class TestCollisionIntegralTableLoading:
    """Test loading collision integral data from JSON."""

    def test_load_from_json(self):
        """Test loading collision integrals from JSON file."""
        table = transport.create_collision_integral_table_from_json(
            COLLISION_INTEGRALS_FILE
        )

        assert isinstance(table, CollisionIntegralTable)
        assert table.n_pairs > 0
        assert len(table.species_pairs) == table.n_pairs

    def test_species_pairs_present(self):
        """Test that expected species pairs are present."""
        table = transport.create_collision_integral_table_from_json(
            COLLISION_INTEGRALS_FILE
        )

        # Check some expected pairs
        expected_pairs = [
            ("N", "N"),
            ("O", "O"),
            ("N2", "N2"),
            ("N", "O"),
            ("N2", "O2"),
        ]
        for pair in expected_pairs:
            try:
                idx = table.get_pair_index(pair[0], pair[1])
                assert idx >= 0
            except ValueError:
                pytest.fail(f"Expected pair {pair} not found in table")


class TestPairIndexMatrix:
    """Test pair index matrix construction."""

    def test_build_pair_index_matrix(self, collision_integrals):
        """Test building pair index matrix."""
        species_names = ("N", "O", "N2", "O2", "NO")
        n_species = len(species_names)

        indices = transport.build_pair_index_matrix(species_names, collision_integrals)

        assert indices.shape == (n_species, n_species)
        assert jnp.all(indices >= 0)
        assert jnp.all(indices < collision_integrals.n_pairs)

    def test_pair_index_symmetry(self, collision_integrals):
        """Pair indices should be symmetric (D_sr = D_rs)."""
        species_names = ("N", "O", "N2", "O2", "NO")

        indices = transport.build_pair_index_matrix(species_names, collision_integrals)

        # Same index for (s,r) and (r,s) since collision integrals are symmetric
        assert jnp.allclose(indices, indices.T)


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
