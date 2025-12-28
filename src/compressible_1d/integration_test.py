"""Integration tests for cross-module functionality"""

import jax
import jax.numpy as jnp
from pathlib import Path

from compressible_1d import equation_manager_utils
from compressible_1d import equation_manager_types
from compressible_1d import numerics_types
from compressible_1d.chemistry_utils import load_species_table_from_gnoffo
from compressible_1d import constants
from compressible_1d import thermodynamic_relations

# Configure JAX for testing
jax.config.update("jax_enable_x64", True)

# Load test data
data_dir = Path(__file__).parent.parent.parent / "data"
general_data = str(data_dir / "air_5_gnoffo.json")
enthalpy_data = str(data_dir / "air_5_gnoffo_equilibrium_enthalpy.json")


def test_full_workflow_species_table_to_primitives():
    """Test complete workflow: load data -> create SpeciesTable -> extract primitives."""
    print("\n=== Full Workflow Test ===")

    # 1. Load species data
    print("Loading species data...")
    species_table = load_species_table_from_gnoffo(general_data, enthalpy_data)
    print(f"  Loaded {species_table.n_species} species: {species_table.names}")

    # 2. Create equation manager
    print("Creating equation manager...")
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

    equation_manager = equation_manager_types.EquationManager(
        species=species_table,
        reactions=None,
        numerics_config=numerics_config,
        boundary_condition="periodic",
    )

    # 3. Create synthetic conserved state
    print("Creating synthetic conserved state...")
    n_cells = 10
    n_species = species_table.n_species
    rho_total = 1.225  # kg/m^3
    rho_s = jnp.full((n_cells, n_species), rho_total / n_species)
    u = jnp.zeros(n_cells)
    rho_u = rho_total * u

    # Approximate energy for air at 300K
    T_approx = 300.0
    e_approx = 2e5  # J/kg (rough estimate)
    rho_E = rho_total * (e_approx + 0.5 * u**2)
    rho_Ev = jnp.zeros(n_cells)  # Equilibrium: minimal vibrational energy

    U = jnp.concatenate(
        [
            rho_s,
            rho_u[:, None],
            rho_E[:, None],
            rho_Ev[:, None],
        ],
        axis=1,
    )

    # 4. Extract primitives
    print("Extracting primitives...")
    Y_s, rho, T, T_v, p = equation_manager_utils.extract_primitives_from_U(
        U, equation_manager
    )

    # 5. Test thermodynamic functions with extracted T, T_v
    print("Testing thermodynamic functions...")
    h = thermodynamic_relations.compute_equilibrium_enthalpy(T, species_table)
    cp = thermodynamic_relations.compute_cp(T, species_table)
    cv_tr = thermodynamic_relations.compute_cv_tr(T, species_table)
    e_vib = thermodynamic_relations.compute_e_ve(T_v, species_table)

    # Check shapes
    if h.shape != (n_species, n_cells):
        raise ValueError(f"h shape mismatch: {h.shape}")
    if cp.shape != (n_species, n_cells):
        raise ValueError(f"cp shape mismatch: {cp.shape}")
    if cv_tr.shape != (n_species, n_cells):
        raise ValueError(f"cv_tr shape mismatch: {cv_tr.shape}")
    if e_vib.shape != (n_species, n_cells):
        raise ValueError(f"e_vib shape mismatch: {e_vib.shape}")

    # 6. Test is_monoatomic property
    print("Testing is_monoatomic property...")
    is_mono = species_table.is_monoatomic
    print(f"  is_monoatomic: {is_mono}")
    print(f"  Species: {species_table.names}")

    # Verify consistency
    if not jnp.allclose(is_mono, ~species_table.has_dissociation_energy):
        raise ValueError("is_monoatomic inconsistent with has_dissociation_energy")

    print("\n✓ Full workflow test passed!")
    print(f"  Extracted T: {T[0]:.2f} K")
    print(f"  Extracted p: {p[0]:.2f} Pa")
    print(f"  Extracted rho: {rho[0]:.4f} kg/m³")


def test_jax_jit_compilation():
    """Test that all key functions can be JIT compiled."""
    print("\n=== JAX JIT Compilation Test ===")

    species_table = load_species_table_from_gnoffo(general_data, enthalpy_data)

    # Test JIT compilation of thermodynamic functions
    @jax.jit
    def compute_all_thermodynamic_properties(T, T_v):
        h = thermodynamic_relations.compute_equilibrium_enthalpy(T, species_table)
        cp = thermodynamic_relations.compute_cp(T, species_table)
        cv_tr = thermodynamic_relations.compute_cv_tr(T, species_table)
        e_vib = thermodynamic_relations.compute_e_ve(T_v, species_table)
        return h, cp, cv_tr, e_vib

    T_test = jnp.array([300.0, 1000.0, 5000.0])
    T_v_test = jnp.array([300.0, 1000.0, 5000.0])

    # First call: compilation
    print("First call (compilation)...")
    h1, cp1, cv_tr1, e_vib1 = compute_all_thermodynamic_properties(T_test, T_v_test)

    # Second call: should use compiled version
    print("Second call (using compiled version)...")
    h2, cp2, cv_tr2, e_vib2 = compute_all_thermodynamic_properties(T_test, T_v_test)

    # Results should be identical
    if not jnp.allclose(h1, h2):
        raise ValueError("JIT compilation changed results for h")
    if not jnp.allclose(cp1, cp2):
        raise ValueError("JIT compilation changed results for cp")

    print("✓ JAX JIT compilation test passed!")


def test_jax_vmap_compatibility():
    """Test that functions can be vmapped over batch dimension."""
    print("\n=== JAX vmap Compatibility Test ===")

    species_table = load_species_table_from_gnoffo(general_data, enthalpy_data)

    # Create batch of temperature arrays
    batch_size = 5
    T_batch = jnp.array(
        [
            [300.0, 1000.0, 5000.0],
            [400.0, 1200.0, 6000.0],
            [500.0, 1500.0, 7000.0],
            [600.0, 2000.0, 8000.0],
            [700.0, 2500.0, 9000.0],
        ]
    )  # Shape: (batch_size, 3)

    # Define function to vmap
    def compute_enthalpy_single(T):
        return thermodynamic_relations.compute_equilibrium_enthalpy(T, species_table)

    # Apply vmap over batch dimension
    compute_enthalpy_batch = jax.vmap(compute_enthalpy_single)
    h_batch = compute_enthalpy_batch(T_batch)

    # Check output shape
    expected_shape = (batch_size, species_table.n_species, 3)
    if h_batch.shape != expected_shape:
        raise ValueError(f"Expected shape {expected_shape}, got {h_batch.shape}")

    # Verify results match non-batched version
    for i in range(batch_size):
        h_single = thermodynamic_relations.compute_equilibrium_enthalpy(T_batch[i], species_table)
        if not jnp.allclose(h_batch[i], h_single):
            raise ValueError(f"vmap results differ from single call at index {i}")

    print("✓ JAX vmap compatibility test passed!")


def test_thermodynamic_consistency():
    """Test thermodynamic relationships across modules."""
    print("\n=== Thermodynamic Consistency Test ===")

    species_table = load_species_table_from_gnoffo(general_data, enthalpy_data)

    # Test 1: C_p = dh/dT relationship
    print("Testing C_p = dh/dT...")
    # Avoid temperature boundaries (300, 1000, 6000, 15000 K) to prevent discontinuities
    T = jnp.array([500.0, 3000.0, 8000.0])
    dT = 0.1

    h_plus = thermodynamic_relations.compute_equilibrium_enthalpy(T + dT, species_table)
    h_minus = thermodynamic_relations.compute_equilibrium_enthalpy(T - dT, species_table)
    cp_numerical = (h_plus - h_minus) / (2 * dT)

    cp_analytical = thermodynamic_relations.compute_cp(T, species_table)

    rel_error = jnp.abs(cp_analytical - cp_numerical) / jnp.abs(cp_analytical)
    if not jnp.all(rel_error < 1e-6):
        raise ValueError(f"C_p != dh/dT, max error: {jnp.max(rel_error):.2e}")

    # Test 2: C_v,tr consistency for atoms vs molecules
    print("Testing C_v,tr for atoms vs molecules...")
    cv_tr = thermodynamic_relations.compute_cv_tr(T, species_table)

    R = constants.R_universal
    M = species_table.molar_masses / 1e3
    is_atom = species_table.is_monoatomic.astype(bool)

    # Atoms: C_v,tr = 1.5 R/M
    cv_tr_atoms_expected = 1.5 * R / M[is_atom]
    if not jnp.allclose(cv_tr[is_atom, 0], cv_tr_atoms_expected, rtol=1e-10):
        raise ValueError("Atom C_v,tr mismatch")

    # Molecules: C_v,tr = 2.5 R/M
    is_molecule = ~is_atom
    cv_tr_molecules_expected = 2.5 * R / M[is_molecule]
    if not jnp.allclose(cv_tr[is_molecule, 0], cv_tr_molecules_expected, rtol=1e-10):
        raise ValueError("Molecule C_v,tr mismatch")

    # Test 3: Vibrational energy monotonicity
    print("Testing vibrational energy monotonicity...")
    T_v = jnp.linspace(300.0, 10000.0, 20)
    e_vib = thermodynamic_relations.compute_e_ve(T_v, species_table)

    for i in jnp.where(is_molecule)[0]:
        de = jnp.diff(e_vib[i, :])
        if not jnp.all(de >= -1e-6):  # Allow small numerical errors
            raise ValueError(f"Vibrational energy not monotonic for molecule {i}")

    print("✓ Thermodynamic consistency test passed!")


def test_property_consistency():
    """Test consistency of SpeciesTable properties."""
    print("\n=== Property Consistency Test ===")

    species_table = load_species_table_from_gnoffo(general_data, enthalpy_data)

    # Test: is_monoatomic should be inverse of has_dissociation_energy
    is_mono = species_table.is_monoatomic
    has_diss = species_table.has_dissociation_energy

    if not jnp.allclose(is_mono, ~has_diss):
        raise ValueError("is_monoatomic != ~has_dissociation_energy")

    # Test: atoms should have no dissociation energy
    is_atom = is_mono.astype(bool)
    if jnp.any(jnp.isfinite(species_table.dissociation_energy[is_atom])):
        raise ValueError("Atoms should not have finite dissociation energy")

    # Test: molecules should have dissociation energy
    is_molecule = ~is_atom
    if not jnp.all(jnp.isfinite(species_table.dissociation_energy[is_molecule])):
        raise ValueError("Molecules must have finite dissociation energy")

    print("✓ Property consistency test passed!")


# Tests are automatically discovered and run by pytest
# Run with: pytest src/compressible_1d/integration_test.py
