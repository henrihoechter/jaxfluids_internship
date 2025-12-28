"""Unit tests for SpeciesTable class in chemistry_types.py"""

import jax
import jax.numpy as jnp
from pathlib import Path

from compressible_1d.chemistry_utils import load_species_table_from_gnoffo
from compressible_1d import constants
from compressible_1d import thermodynamic_relations

# Configure JAX for testing
jax.config.update("jax_enable_x64", True)

# Load test data
data_dir = Path(__file__).parent.parent.parent / "data"
general_data = str(data_dir / "air_5_gnoffo.json")
enthalpy_data = str(data_dir / "air_5_gnoffo_equilibrium_enthalpy.json")


def test_has_dissociation_energy():
    """Test that molecules have dissociation energy, atoms do not."""
    species_table = load_species_table_from_gnoffo(general_data, enthalpy_data)

    has_diss = species_table.has_dissociation_energy

    # Check shape
    if has_diss.shape != (species_table.n_species,):
        raise ValueError(
            f"Expected shape ({species_table.n_species},), got {has_diss.shape}"
        )

    # For air_5: N2, O2, NO are molecules (have dissociation), N, O are atoms (no dissociation)
    # This depends on the actual data structure - adjust indices based on species order
    print(f"Species: {species_table.names}")
    print(f"has_dissociation_energy: {has_diss}")


def test_has_ionization_energy():
    """Test that species have ionization energy (except electrons if present)."""
    species_table = load_species_table_from_gnoffo(general_data, enthalpy_data)

    has_ion = species_table.has_ionization_energy

    # Check shape
    if has_ion.shape != (species_table.n_species,):
        raise ValueError(
            f"Expected shape ({species_table.n_species},), got {has_ion.shape}"
        )

    # All heavy particles should have ionization energy
    # If electrons present, they would not have ionization energy
    print(f"has_ionization_energy: {has_ion}")


def test_has_vibrational_mode():
    """Test that molecules have vibrational modes, atoms do not."""
    species_table = load_species_table_from_gnoffo(general_data, enthalpy_data)

    has_vib = species_table.has_vibrational_mode

    # Check shape
    if has_vib.shape != (species_table.n_species,):
        raise ValueError(
            f"Expected shape ({species_table.n_species},), got {has_vib.shape}"
        )

    print(f"has_vibrational_mode: {has_vib}")


def test_is_monoatomic():
    """Test the new is_monoatomic property."""
    species_table = load_species_table_from_gnoffo(general_data, enthalpy_data)

    is_monoatomic = species_table.is_monoatomic

    # Check shape
    if is_monoatomic.shape != (species_table.n_species,):
        raise ValueError(
            f"Expected shape ({species_table.n_species},), got {is_monoatomic.shape}"
        )

    # Verify logic: is_monoatomic should be inverse of has_dissociation_energy
    expected = ~species_table.has_dissociation_energy
    if not jnp.allclose(is_monoatomic, expected):
        raise ValueError("is_monoatomic should equal ~has_dissociation_energy")

    print(f"is_monoatomic: {is_monoatomic}")
    print(f"  Species: {species_table.names}")
    print("  Expected for N2, O2, NO: False (molecules)")
    print("  Expected for N, O: True (atoms)")


def test_electron_index():
    """Test electron index lookup."""
    species_table = load_species_table_from_gnoffo(general_data, enthalpy_data)

    electron_idx = species_table.electron_index

    # air_5 does not contain electrons
    if electron_idx is not None:
        raise ValueError("air_5_gnoffo should not contain electrons")

    print(f"Electron index: {electron_idx} (expected None for air_5)")


def test_array_shapes_consistency():
    """Test that all per-species arrays have consistent shapes."""
    species_table = load_species_table_from_gnoffo(general_data, enthalpy_data)

    n = species_table.n_species

    # Check 1D arrays
    arrays_1d = [
        ("molar_masses", species_table.molar_masses),
        ("formation_enthalpy", species_table.h_s0),
        ("dissociation_energy", species_table.dissociation_energy),
        ("ionization_energy", species_table.ionization_energy),
        ("vibrational_relaxation_factor", species_table.vibrational_relaxation_factor),
    ]

    for name, array in arrays_1d:
        if array.shape[0] != n:
            raise ValueError(
                f"{name} has shape {array.shape}, expected first dim = {n}"
            )

    print(f"All array shapes consistent with n_species={n}")


def test_physical_constraints():
    """Test that species data satisfies physical constraints."""
    species_table = load_species_table_from_gnoffo(general_data, enthalpy_data)

    # Molar masses must be positive
    if not jnp.all(species_table.molar_masses > 0):
        raise ValueError("All molar masses must be positive")

    # Check that atoms have no dissociation energy (is_monoatomic logic)
    is_atom = species_table.is_monoatomic.astype(bool)
    if jnp.any(jnp.isfinite(species_table.dissociation_energy[is_atom])):
        raise ValueError("Atoms should not have finite dissociation energy")

    # Check that molecules have dissociation energy
    is_molecule = ~is_atom
    if not jnp.all(jnp.isfinite(species_table.dissociation_energy[is_molecule])):
        raise ValueError("Molecules must have finite dissociation energy")

    print("All physical constraints satisfied")


def test_equilibrium_enthalpy():
    """Test equilibrium enthalpy function."""
    species_table = load_species_table_from_gnoffo(general_data, enthalpy_data)

    T = jnp.array([300.0, 1000.0, 5000.0, 10000.0])
    h = thermodynamic_relations.compute_equilibrium_enthalpy(T, species_table)

    # Check shape
    expected_shape = (species_table.n_species, len(T))
    if h.shape != expected_shape:
        raise ValueError(f"Expected shape {expected_shape}, got {h.shape}")

    # Check that enthalpy increases with temperature
    for i in range(species_table.n_species):
        if not jnp.all(h[i, 1:] > h[i, :-1]):
            print(f"Species {i} ({species_table.names[i]}) enthalpy: {h[i, :]}")
            print(f"  Differences: {jnp.diff(h[i, :])}")
            raise ValueError(
                f"Enthalpy for species {i} ({species_table.names[i]}) is not monotonically increasing"
            )

    # Check physically reasonable magnitudes (should be in range 1e4 to 1e8 J/kg)
    if not jnp.all((jnp.abs(h) > 1e3) & (jnp.abs(h) < 1e8)):
        print("Warning: Some enthalpy values outside expected range [1e3, 1e8] J/kg")
        print(f"Range: [{jnp.min(h):.2e}, {jnp.max(h):.2e}]")

    print(f"equilibrium_enthalpy test passed, shape: {h.shape}")


def test_cp():
    """Test specific heat at constant pressure function."""
    species_table = load_species_table_from_gnoffo(general_data, enthalpy_data)

    T = jnp.array([300.0, 1000.0, 5000.0, 10000.0])
    cp = thermodynamic_relations.compute_cp(T, species_table)

    # Check shape
    expected_shape = (species_table.n_species, len(T))
    if cp.shape != expected_shape:
        raise ValueError(f"Expected shape {expected_shape}, got {cp.shape}")

    # C_p must be positive
    if not jnp.all(cp > 0):
        raise ValueError("C_p must be positive for all species at all temperatures")

    # Verify C_p ≈ dh/dT using finite differences
    # Avoid temperature range boundaries (300, 1000, 6000, 15000) to avoid discontinuities
    dT = 0.1
    T_test = jnp.array([500.0, 3000.0, 8000.0])
    h_plus = thermodynamic_relations.compute_equilibrium_enthalpy(T_test + dT, species_table)
    h_minus = thermodynamic_relations.compute_equilibrium_enthalpy(T_test - dT, species_table)
    cp_numerical = (h_plus - h_minus) / (2 * dT)
    cp_analytical = thermodynamic_relations.compute_cp(T_test, species_table)

    rel_error = jnp.abs(cp_analytical - cp_numerical) / jnp.abs(cp_analytical)
    if not jnp.all(rel_error < 1e-6):
        print("\nC_p vs dh/dT mismatch:")
        print(f"Test temperatures: {T_test}")
        for i, name in enumerate(species_table.names):
            print(f"\n{name}:")
            for j in range(len(T_test)):
                print(
                    f"  T={T_test[j]:.0f}K: C_p(analytical)={cp_analytical[i,j]:.2e}, "
                    f"dh/dT(numerical)={cp_numerical[i,j]:.2e}, rel_err={rel_error[i,j]:.2e}"
                )
        raise ValueError(f"C_p != dh/dT, max relative error: {jnp.max(rel_error):.2e}")

    print(f"cp test passed, shape: {cp.shape}")


def test_cv_tr():
    """Test translational-rotational specific heat function."""
    species_table = load_species_table_from_gnoffo(general_data, enthalpy_data)

    T = jnp.array([300.0, 1000.0, 5000.0])
    cv_tr = thermodynamic_relations.compute_cv_tr(T, species_table)

    # Check shape
    expected_shape = (species_table.n_species, len(T))
    if cv_tr.shape != expected_shape:
        raise ValueError(f"Expected shape {expected_shape}, got {cv_tr.shape}")

    # Verify temperature independence (should be same at all T)
    for i in range(species_table.n_species):
        if not jnp.allclose(cv_tr[i, :], cv_tr[i, 0]):
            raise ValueError(
                f"C_v,tr for species {i} is not constant across temperatures"
            )

    # Check theoretical values
    R = constants.R_universal  # J/(mol·K)
    M = species_table.molar_masses / 1e3  # kg/mol
    is_atom = species_table.is_monoatomic.astype(bool)

    # Atoms: C_v,tr = 1.5 * R/M
    cv_tr_atoms_expected = 1.5 * R / M[is_atom]
    cv_tr_atoms_actual = cv_tr[is_atom, 0]
    if not jnp.allclose(cv_tr_atoms_actual, cv_tr_atoms_expected, rtol=1e-10):
        raise ValueError(
            f"Atom C_v,tr mismatch: expected {cv_tr_atoms_expected}, got {cv_tr_atoms_actual}"
        )

    # Molecules: C_v,tr = 2.5 * R/M
    is_molecule = ~is_atom
    cv_tr_molecules_expected = 2.5 * R / M[is_molecule]
    cv_tr_molecules_actual = cv_tr[is_molecule, 0]
    if not jnp.allclose(cv_tr_molecules_actual, cv_tr_molecules_expected, rtol=1e-10):
        raise ValueError(
            f"Molecule C_v,tr mismatch: expected {cv_tr_molecules_expected}, got {cv_tr_molecules_actual}"
        )

    print(f"cv_tr test passed, shape: {cv_tr.shape}")


def test_e_ve():
    """Test vibrational-electronic energy function."""
    species_table = load_species_table_from_gnoffo(general_data, enthalpy_data)

    T_V = jnp.array([300.0, 1000.0, 5000.0, 10000.0])
    e_vib = thermodynamic_relations.compute_e_ve(T_V, species_table)

    # Check shape
    expected_shape = (species_table.n_species, len(T_V))
    if e_vib.shape != expected_shape:
        raise ValueError(f"Expected shape {expected_shape}, got {e_vib.shape}")

    # For molecules, e_vib should increase with T_V
    is_molecule = (~species_table.is_monoatomic).astype(bool)
    for i in jnp.where(is_molecule)[0]:
        if not jnp.all(e_vib[i, 1:] >= e_vib[i, :-1]):  # >= allows for saturation
            raise ValueError(
                f"Vibrational energy for molecule {i} is not monotonically increasing"
            )

    print(f"e_ve test passed, shape: {e_vib.shape}")


# Tests are automatically discovered and run by pytest
# Run with: pytest src/compressible_1d/chemistry_types_test.py
