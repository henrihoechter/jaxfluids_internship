"""Tests for reaction rate calculations."""

import jax.numpy as jnp
import pytest

from compressible_1d import chemistry_utils, reaction_rates, energy_models


@pytest.fixture
def species_table():
    """Load 5-species air model."""
    species_names = ["N", "O", "N2", "O2", "NO"]
    return chemistry_utils.load_species_table(
        species_names=species_names,
        general_data_path="data/air_5_gnoffo.json",
        energy_model_config=energy_models.EnergyModelConfig(
            model="gnoffo",
            data_path="data/air_5_gnoffo_equilibrium_enthalpy.json",
        ),
    )


@pytest.fixture
def reaction_table():
    """Load Park neutral reactions."""
    species_names = ["N", "O", "N2", "O2", "NO"]
    return chemistry_utils.load_reactions_from_json(
        json_path="data/park_reactions_neutral.json",
        species_names=species_names,
        preferential_factor=1.0,
    )


class TestReactionTableLoading:
    """Tests for loading ReactionTable from JSON."""

    def test_load_reactions_count(self, reaction_table):
        """Should load 17 neutral reactions (expanded from 9 with M reactions)."""
        assert reaction_table.n_reactions == 17

    def test_load_reactions_species(self, reaction_table):
        """Should have correct species order."""
        assert reaction_table.species_names == ("N", "O", "N2", "O2", "NO")
        assert reaction_table.n_species == 5

    def test_stoichiometry_shapes(self, reaction_table):
        """Stoichiometry arrays should have correct shapes."""
        assert reaction_table.reactant_stoich.shape == (17, 5)
        assert reaction_table.product_stoich.shape == (17, 5)

    def test_arrhenius_shapes(self, reaction_table):
        """Arrhenius parameters should have correct shapes."""
        assert reaction_table.C_f.shape == (17,)
        assert reaction_table.n_f.shape == (17,)
        assert reaction_table.E_f_over_k.shape == (17,)

    def test_equilibrium_coeffs_shape(self, reaction_table):
        """Equilibrium coefficients should have shape [n_reactions, 5]."""
        assert reaction_table.equilibrium_coeffs.shape == (17, 5)

    def test_n2_dissociation_stoichiometry(self, reaction_table):
        """N2 + N -> 2N + N should have correct stoichiometry."""
        # Reaction at index 5: N2 + N -> 3N (table_index 3)
        # Reactants: N2=1, N=1
        # Products: N=3
        r = 5  # Reaction index for "N2 + N -> 2N + N"
        n2_idx = reaction_table.get_species_index("N2")
        n_idx = reaction_table.get_species_index("N")

        # Net stoich for N2: -1 (consumed)
        assert reaction_table.net_stoich[r, n2_idx] == -1.0
        # Net stoich for N: +2 (3 produced - 1 consumed as collision partner)
        assert reaction_table.net_stoich[r, n_idx] == 2.0


class TestRateControllingTemperature:
    """Tests for rate-controlling temperature calculation."""

    def test_dissociation_uses_geometric_mean(self, reaction_table):
        """Dissociation reactions should use T_d = sqrt(T * T_v)."""
        T = jnp.array([10000.0])
        T_v = jnp.array([5000.0])

        T_q = reaction_rates.compute_rate_controlling_temperature(
            T, T_v, reaction_table.is_dissociation, reaction_table.is_electron_impact
        )

        # First reaction is dissociation
        expected_T_d = jnp.sqrt(10000.0 * 5000.0)  # ~7071 K
        assert jnp.isclose(T_q[0, 0], expected_T_d, rtol=1e-5)

    def test_exchange_uses_translational(self, reaction_table):
        """Exchange reactions should use T (translational)."""
        T = jnp.array([10000.0])
        T_v = jnp.array([5000.0])

        T_q = reaction_rates.compute_rate_controlling_temperature(
            T, T_v, reaction_table.is_dissociation, reaction_table.is_electron_impact
        )

        # Exchange reactions are at indices 15 and 16 (last two reactions)
        assert jnp.isclose(T_q[15, 0], 10000.0, rtol=1e-5)
        assert jnp.isclose(T_q[16, 0], 10000.0, rtol=1e-5)


class TestForwardRateCoefficient:
    """Tests for forward rate coefficient calculation."""

    def test_arrhenius_form(self):
        """k_f should follow Arrhenius form: C * T^n * exp(-E/kT)."""
        T_q = jnp.array([[1000.0], [2000.0]])  # 2 reactions, 1 cell
        C_f = jnp.array([1e10, 1e12])
        n_f = jnp.array([0.0, -1.0])
        E_f_over_k = jnp.array([10000.0, 20000.0])

        k_f = reaction_rates.compute_forward_rate_coefficient(T_q, C_f, n_f, E_f_over_k)

        # Manual calculation for first reaction at T=1000K
        expected_k_f_0 = 1e10 * jnp.exp(-10000.0 / 1000.0)
        assert jnp.isclose(k_f[0, 0], expected_k_f_0, rtol=1e-5)

        # Manual calculation for second reaction at T=2000K
        expected_k_f_1 = 1e12 * (2000.0**-1.0) * jnp.exp(-20000.0 / 2000.0)
        assert jnp.isclose(k_f[1, 0], expected_k_f_1, rtol=1e-5)


class TestEquilibriumConstant:
    """Tests for equilibrium constant calculation."""

    def test_polynomial_form(self, reaction_table):
        """K_c should follow polynomial form from NASA TP-2867."""
        T = jnp.array([5000.0])

        K_c = reaction_rates.compute_equilibrium_constant(
            T, reaction_table.equilibrium_coeffs
        )

        # All K_c values should be positive
        assert jnp.all(K_c > 0)

        # At moderate temperatures, K_c for dissociation should be < 1
        # (molecules are more stable than atoms)
        assert K_c[0, 0] < 1.0  # O2 dissociation


class TestSpeciesProductionRates:
    """Tests for species production rate calculation."""

    def test_mass_conservation(self, species_table, reaction_table):
        """Sum of omega_dot should be zero (mass conservation)."""
        T = jnp.array([10000.0])
        T_v = jnp.array([5000.0])
        rho_s = jnp.array([[0.0, 0.0, 0.1, 0.0, 0.0]])  # Pure N2

        omega_dot, _, _ = reaction_rates.compute_all_chemical_sources(
            rho_s, T, T_v, species_table, reaction_table
        )

        mass_sum = jnp.sum(omega_dot)
        assert jnp.isclose(mass_sum, 0.0, atol=1e-10)

    def test_n2_dissociation_produces_n(self, species_table, reaction_table):
        """N2 dissociation should produce N and consume N2."""
        T = jnp.array([15000.0])  # High T favors dissociation
        T_v = jnp.array([10000.0])
        rho_s = jnp.array([[0.0, 0.0, 0.1, 0.0, 0.0]])  # Pure N2

        omega_dot, _, _ = reaction_rates.compute_all_chemical_sources(
            rho_s, T, T_v, species_table, reaction_table
        )

        n_idx = reaction_table.get_species_index("N")
        n2_idx = reaction_table.get_species_index("N2")

        # N should be produced (positive)
        assert omega_dot[0, n_idx] > 0
        # N2 should be consumed (negative)
        assert omega_dot[0, n2_idx] < 0

    def test_chemical_energy_release(self, species_table, reaction_table):
        """Dissociation should absorb energy (endothermic)."""
        T = jnp.array([15000.0])
        T_v = jnp.array([10000.0])
        rho_s = jnp.array([[0.0, 0.0, 0.1, 0.0, 0.0]])  # Pure N2

        _, Q_chem, _ = reaction_rates.compute_all_chemical_sources(
            rho_s, T, T_v, species_table, reaction_table
        )

        # Dissociation is endothermic: Q_chem < 0 (energy absorbed)
        # Actually Q_chem = -sum(omega_dot * h_s0)
        # For N2 -> 2N: omega_N2 < 0, h_N2 = 0, omega_N > 0, h_N > 0
        # Q_chem = -(-|omega_N2|*0 + |omega_N|*h_N) = -|omega_N|*h_N < 0
        assert Q_chem[0] < 0


class TestVibrationalReactiveSource:
    """Tests for vibrational energy reactive source."""

    def test_dissociation_removes_vibrational_energy(
        self, species_table, reaction_table
    ):
        """Dissociation should remove vibrational energy from the system."""
        T = jnp.array([15000.0])
        T_v = jnp.array([10000.0])
        rho_s = jnp.array([[0.0, 0.0, 0.1, 0.0, 0.0]])  # Pure N2

        _, _, Q_vib_chem = reaction_rates.compute_all_chemical_sources(
            rho_s, T, T_v, species_table, reaction_table
        )

        # When N2 dissociates (omega_N2 < 0), molecules carrying vibrational
        # energy are removed, so Q_vib_chem = omega_N2 * e_v_N2 < 0
        assert Q_vib_chem[0] < 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
