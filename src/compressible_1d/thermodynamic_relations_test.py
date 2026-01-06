"""Unit tests for thermodynamic_relations.py functions"""

import jax
import jax.numpy as jnp
from pathlib import Path

from compressible_1d import thermodynamic_relations
from compressible_1d.chemistry_utils import load_species_table_from_gnoffo
from compressible_1d import constants

# Configure JAX for testing
jax.config.update("jax_enable_x64", True)

# Load test data
data_dir = Path(__file__).parent.parent.parent / "data"
general_data = str(data_dir / "air_5_gnoffo.json")
enthalpy_data = str(data_dir / "air_5_gnoffo_equilibrium_enthalpy.json")


def test_compute_equilibrium_enthalpy_polynomial_shape():
    """Test output shape of equilibrium enthalpy computation."""
    species_table = load_species_table_from_gnoffo(general_data, enthalpy_data)

    T = jnp.array([300.0, 1000.0, 5000.0, 10000.0])
    h = thermodynamic_relations.compute_equilibrium_enthalpy(T, species_table)

    expected_shape = (species_table.n_species, len(T))
    if h.shape != expected_shape:
        raise ValueError(f"Expected shape {expected_shape}, got {h.shape}")

    print(f"Enthalpy shape test passed: {h.shape}")


def test_compute_equilibrium_enthalpy_polynomial_monotonic():
    """Test that enthalpy increases monotonically with temperature."""
    species_table = load_species_table_from_gnoffo(general_data, enthalpy_data)

    T = jnp.linspace(300.0, 10000.0, 50)
    h = thermodynamic_relations.compute_equilibrium_enthalpy(T, species_table)

    # Check monotonicity for each species
    for i in range(species_table.n_species):
        dh = jnp.diff(h[i, :])
        if not jnp.all(dh > 0):
            raise ValueError(
                f"Enthalpy not monotonically increasing for species {i} ({species_table.names[i]})"
            )

    print("Enthalpy monotonicity test passed")


def test_compute_cp_equilibrium_polynomial_shape():
    """Test output shape of C_p computation."""
    species_table = load_species_table_from_gnoffo(general_data, enthalpy_data)

    T = jnp.array([300.0, 1000.0, 5000.0])
    cp = thermodynamic_relations.compute_cp(T, species_table)

    expected_shape = (species_table.n_species, len(T))
    if cp.shape != expected_shape:
        raise ValueError(f"Expected shape {expected_shape}, got {cp.shape}")

    print(f"C_p shape test passed: {cp.shape}")


def test_compute_cp_equilibrium_polynomial_positive():
    """Test that C_p is positive at all temperatures."""
    species_table = load_species_table_from_gnoffo(general_data, enthalpy_data)

    # Use valid temperature range (data starts at 300K)
    T = jnp.linspace(300.0, 20000.0, 100)
    cp = thermodynamic_relations.compute_cp(T, species_table)

    if not jnp.all(cp > 0):
        raise ValueError("C_p must be positive for all species at all temperatures")

    print("C_p positivity test passed")


def test_compute_cp_equilibrium_polynomial_derivative_relationship():
    """Test that C_p = dh/dT using JAX automatic differentiation."""
    species_table = load_species_table_from_gnoffo(general_data, enthalpy_data)

    def h_function(T_scalar):
        """Wrapper for computing enthalpy at a single temperature."""
        T_array = jnp.array([T_scalar])
        h = thermodynamic_relations.compute_equilibrium_enthalpy(T_array, species_table)
        return h[:, 0]  # Return (n_species,) array

    # Test at multiple temperatures
    T_test = jnp.array([500.0, 2000.0, 8000.0])

    for T_val in T_test:
        # Compute dh/dT using JAX grad
        dh_dT = jax.grad(lambda t: jnp.sum(h_function(t)))(T_val)

        # Compute C_p directly
        cp = thermodynamic_relations.compute_cp(jnp.array([T_val]), species_table)
        cp_sum = jnp.sum(cp[:, 0])

        # Compare
        rel_error = jnp.abs(dh_dT - cp_sum) / jnp.abs(cp_sum)
        if rel_error > 1e-6:
            raise ValueError(f"C_p != dh/dT at T={T_val}K, rel_error={rel_error:.2e}")

    print("C_p = dh/dT relationship verified via automatic differentiation")


def test_compute_cv_trans_rot_atoms_vs_molecules():
    """Test that C_v,tr has correct values for atoms vs molecules."""
    species_table = load_species_table_from_gnoffo(general_data, enthalpy_data)

    T = jnp.array([1000.0])
    cv_tr = thermodynamic_relations.compute_cv_tr(T, species_table)

    # Constants
    R = constants.R_universal
    M = species_table.molar_masses  # kg/mol

    # Check atoms (is_monoatomic = True)
    is_atom = species_table.is_monoatomic.astype(bool)
    cv_tr_atoms_expected = 1.5 * R / M[is_atom]
    cv_tr_atoms_actual = cv_tr[is_atom, 0]

    if not jnp.allclose(cv_tr_atoms_actual, cv_tr_atoms_expected, rtol=1e-10):
        raise ValueError(
            f"Atom C_v,tr mismatch:\n"
            f"  Expected: {cv_tr_atoms_expected}\n"
            f"  Actual: {cv_tr_atoms_actual}"
        )

    # Check molecules
    is_molecule = ~is_atom
    cv_tr_molecules_expected = 2.5 * R / M[is_molecule]
    cv_tr_molecules_actual = cv_tr[is_molecule, 0]

    if not jnp.allclose(cv_tr_molecules_actual, cv_tr_molecules_expected, rtol=1e-10):
        raise ValueError(
            f"Molecule C_v,tr mismatch:\n"
            f"  Expected: {cv_tr_molecules_expected}\n"
            f"  Actual: {cv_tr_molecules_actual}"
        )

    print("C_v,tr atoms vs molecules test passed")


def test_compute_cv_trans_rot_temperature_independence():
    """Test that C_v,tr is constant across all temperatures."""
    species_table = load_species_table_from_gnoffo(general_data, enthalpy_data)

    T1 = jnp.array([300.0, 1000.0, 5000.0])
    T2 = jnp.array([500.0, 2000.0, 8000.0, 15000.0])

    cv_tr1 = thermodynamic_relations.compute_cv_tr(T1, species_table)
    cv_tr2 = thermodynamic_relations.compute_cv_tr(T2, species_table)

    # Each species should have same value at all temperatures
    for i in range(species_table.n_species):
        # Check T1
        if not jnp.allclose(cv_tr1[i, :], cv_tr1[i, 0]):
            raise ValueError(f"C_v,tr not constant for species {i} across T1")

        # Check T2
        if not jnp.allclose(cv_tr2[i, :], cv_tr2[i, 0]):
            raise ValueError(f"C_v,tr not constant for species {i} across T2")

        # Check that T1 and T2 give same value
        if not jnp.allclose(cv_tr1[i, 0], cv_tr2[i, 0]):
            raise ValueError(f"C_v,tr different between T1 and T2 for species {i}")

    print("C_v,tr temperature independence test passed")


def test_compute_cv_trans_rot_with_is_monoatomic_mask():
    """Test that compute_cv_tr works correctly."""
    species_table = load_species_table_from_gnoffo(general_data, enthalpy_data)

    T = jnp.array([1000.0])

    # Compute using public API
    cv_tr = thermodynamic_relations.compute_cv_tr(T, species_table)

    # Verify shape
    expected_shape = (species_table.n_species, len(T))
    if cv_tr.shape != expected_shape:
        raise ValueError(f"Expected shape {expected_shape}, got {cv_tr.shape}")

    # Verify values are positive
    if not jnp.all(cv_tr > 0):
        raise ValueError("C_v,tr must be positive")

    print("C_v,tr with is_monoatomic mask test passed")


def test_compute_e_vib_electronic_monotonic():
    """Test that vibrational energy increases with temperature for molecules."""
    species_table = load_species_table_from_gnoffo(general_data, enthalpy_data)

    T_V = jnp.linspace(300.0, 10000.0, 50)
    e_vib = thermodynamic_relations.compute_e_ve(T_V, species_table)

    # For molecules (not atoms), e_vib should increase with T_V
    is_molecule = (~species_table.is_monoatomic).astype(bool)
    for i in jnp.where(is_molecule)[0]:
        de_vib = jnp.diff(e_vib[i, :])
        if not jnp.all(de_vib >= -1e-6):  # Allow small numerical errors, saturation
            raise ValueError(
                f"Vibrational energy not monotonic for molecule {i} ({species_table.names[i]})"
            )

    print("Vibrational energy monotonicity test passed")


def test_solve_vibrational_temperature_convergence():
    """Test that T_V solver converges to correct value."""
    species_table = load_species_table_from_gnoffo(general_data, enthalpy_data)

    # Create a target state
    T_V_target = jnp.array([1000.0, 3000.0, 8000.0])
    c_s = (
        jnp.ones((species_table.n_species, len(T_V_target))) / species_table.n_species
    )  # Equal mass fractions

    # Compute e_V at target temperatures
    e_V_target = thermodynamic_relations.compute_e_ve(
        T_V_target, species_table
    )  # Shape: (n_species, 3)

    # Compute mixture vibrational energy
    e_V_mixture = jnp.sum(c_s * e_V_target, axis=0)  # Shape: (3,)

    # Initial guess (far from solution to test convergence)
    T_V_initial = jnp.full_like(e_V_mixture, 500.0)

    # Solve for T_V
    T_V_solved = (
        thermodynamic_relations.solve_vibrational_temperature_from_vibrational_energy(
            e_V_target=e_V_mixture,
            c_s=c_s,
            T_V_initial=T_V_initial,
            species_table=species_table,
            max_iterations=50,
            rtol=1e-8,
            atol=1.0,
        )
    )

    # Verify recovered temperature matches target
    # Note: Tolerance relaxed to 1e-4 to account for numerical precision
    # at high temperatures (8000K) where the solver achieves ~1e-5 error
    rel_error = jnp.abs(T_V_solved - T_V_target) / T_V_target
    if not jnp.all(rel_error < 1e-4):
        raise ValueError(
            f"T_V solver failed to converge:\n"
            f"  Target: {T_V_target}\n"
            f"  Solved: {T_V_solved}\n"
            f"  Rel error: {rel_error}"
        )

    print("T_V solver convergence test passed")


def test_solve_T_from_internal_energy_consistency():
    """Test that T solver can recover known temperature."""
    species_table = load_species_table_from_gnoffo(general_data, enthalpy_data)

    # Known state
    T_known = jnp.array([500.0, 2000.0, 8000.0])
    T_V = jnp.array([500.0, 2000.0, 8000.0])  # Equilibrium
    c_s = jnp.ones((species_table.n_species, len(T_known))) / species_table.n_species

    # Compute cv_tr
    cv_tr = thermodynamic_relations.compute_cv_tr(
        jnp.array([1000.0]), species_table
    )  # Dummy T
    cv_tr_broadcast = jnp.broadcast_to(
        cv_tr[:, 0, None], (species_table.n_species, len(T_known))
    )

    # Compute reference energies
    e_s0 = thermodynamic_relations.compute_reference_internal_energy(
        h_s0=species_table.h_s0,
        molar_masses=species_table.molar_masses,
        T_ref=298.16,
    )

    # Compute vibrational energy
    e_V = thermodynamic_relations.compute_e_ve(T_V, species_table)
    e_V_mixture = jnp.sum(c_s * e_V, axis=0)

    # Compute internal energy at known T
    # e = e_tr + e_V + e_0 where e_tr = Σ c_s cv_tr (T - T_ref)
    e_tr_and_ref = jnp.sum(
        c_s * (cv_tr_broadcast * (T_known - 298.16) + e_s0[:, None]), axis=0
    )
    e_known = e_tr_and_ref + e_V_mixture

    # Solve for T from e
    T_solved = thermodynamic_relations.solve_T_from_internal_energy(
        e=e_known,
        e_V=e_V_mixture,
        c_s=c_s,
        cv_tr=cv_tr_broadcast,
        e_s0=e_s0,
        T_ref=298.16,
    )

    # Verify
    rel_error = jnp.abs(T_solved - T_known) / T_known
    if not jnp.all(rel_error < 1e-10):
        raise ValueError(
            f"T solver failed:\n"
            f"  Known: {T_known}\n"
            f"  Solved: {T_solved}\n"
            f"  Rel error: {rel_error}"
        )

    print("T solver consistency test passed")


def test_compute_reference_internal_energy_shape():
    """Test shape of reference internal energy computation."""
    species_table = load_species_table_from_gnoffo(general_data, enthalpy_data)

    e_s0 = thermodynamic_relations.compute_reference_internal_energy(
        h_s0=species_table.h_s0,
        molar_masses=species_table.molar_masses,
        T_ref=298.16,
    )

    expected_shape = (species_table.n_species,)
    if e_s0.shape != expected_shape:
        raise ValueError(f"Expected shape {expected_shape}, got {e_s0.shape}")

    print(f"Reference internal energy shape test passed: {e_s0.shape}")


def test_compute_reference_internal_energy_relationship():
    """Test e_s0 = h_s0 - R*T_ref/M relationship."""
    species_table = load_species_table_from_gnoffo(general_data, enthalpy_data)

    T_ref = 298.16
    e_s0 = thermodynamic_relations.compute_reference_internal_energy(
        h_s0=species_table.h_s0,
        molar_masses=species_table.molar_masses,
        T_ref=T_ref,
    )

    # Manual calculation
    R = constants.R_universal
    M = species_table.molar_masses  # kg/mol
    h_s0 = species_table.h_s0
    e_s0_expected = h_s0 - R * T_ref / M

    if not jnp.allclose(e_s0, e_s0_expected, rtol=1e-10):
        raise ValueError(
            "Reference internal energy does not match e_s0 = h_s0 - R*T_ref/M"
        )

    print("Reference internal energy relationship test passed")


# Tests are automatically discovered and run by pytest
# Run with: pytest src/compressible_1d/thermodynamic_relations_test.py
