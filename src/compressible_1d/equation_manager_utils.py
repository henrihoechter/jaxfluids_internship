from jaxtyping import Array, Float
import jax.numpy as jnp

from compressible_1d import equation_manager_types
from compressible_1d import constants
from compressible_1d import thermodynamic_relations


def compute_is_monoatomic(
    dissociation_energy: Float[Array, " n_species"],
) -> Float[Array, " n_species"]:
    """Compute boolean mask indicating which species are monoatomic (atoms).

    Atoms have no dissociation energy (NaN in array) because they cannot
    dissociate into smaller pieces. Molecules have finite dissociation energy.

    Args:
        dissociation_energy: Dissociation energy array [J], NaN for atoms

    Returns:
        Boolean array where True = atom (monoatomic), False = molecule
    """
    return ~jnp.isfinite(dissociation_energy)


def extract_primitives_from_U(
    U: Float[Array, "n_cells n_variables"],
    equation_manager: equation_manager_types.EquationManager,
) -> tuple[
    Float[Array, "n_cells n_species"],
    Float[Array, " n_cells"],
    Float[Array, " n_cells"],
    Float[Array, " n_cells"],
    Float[Array, " n_cells"],
]:
    """Extract primitive variables from conserved state vector.

    State vector: U = [rho_1, rho_2, ..., rho_ns, rho_u, rho_E, rho_Ev]
    This conversion is aligned with Gnoffo et al. (1989) Conservation Equations and
    Physical Models for Hypersonic Air Flows in Thermal and Chemical Nonequilibrium.

    Args:
        U: Conserved variables
        equation_manager: Equation manager containing species data

    Returns:
        Y_s: Mole fractions
        rho: Total density [kg/m^3]
        T: Translational temperature [K]
        Tv: Vibrational temperature [K]
        p: Pressure [Pa]
    """
    n_species = equation_manager.species.n_species

    # Extract partial densities
    rho_s = U[:, :n_species]  # [n_cells, n_species]
    rho_u = U[:, n_species]
    rho_E = U[:, n_species + 1]
    rho_Ev = U[:, n_species + 2]

    # Total density (Gnoffo 1989, Eq. 6)
    rho = jnp.sum(rho_s, axis=-1)

    # Velocity
    u = rho_u / rho

    # Kinetic energy
    E_kin = 0.5 * u**2

    # Vibrational energy per unit mass
    E_v = rho_Ev / rho

    # Total energy per unit mass
    E_total = rho_E / rho

    # Internal energy per unit mass (Gnoffo eq. 13)
    e = E_total - E_kin

    M_s = equation_manager.species.M_s

    # Mass fractions (Gnoffo eq. 88)
    c_s = rho_s / rho[:, None]  # Shape: (n_cells, n_species)

    # Solve for T_V using Newton-Raphson
    T_V_initial = jnp.full_like(rho, 298.16)  # [K]
    T_V = thermodynamic_relations.solve_vibrational_temperature_from_vibrational_energy(
        e_V_target=E_v,
        c_s=c_s.T,
        T_V_initial=T_V_initial,
        species_table=equation_manager.species,
        max_iterations=20,
        rtol=1e-6,
        atol=1.0,
    )  # Shape: (n_cells,)

    # Solve for T using direct formula
    # Compute cv_tr at a dummy temperature (it's constant for ideal gas)
    T_dummy = jnp.ones(1)
    cv_tr_all = thermodynamic_relations.compute_cv_tr(
        T_dummy, equation_manager.species
    )  # (n_species, 1)
    cv_tr_broadcast = jnp.broadcast_to(
        cv_tr_all[:, 0, None], (n_species, rho.shape[0])
    )  # (n_species, n_cells)

    # Compute reference internal energies
    e_s0 = thermodynamic_relations.compute_reference_internal_energy(
        equation_manager.species.h_s0,
        equation_manager.species.molar_masses,
        T_ref=298.16,
    )  # Shape: (n_species,)

    T = thermodynamic_relations.solve_T_from_internal_energy(
        e=e,
        e_V=E_v,
        c_s=c_s.T,
        cv_tr=cv_tr_broadcast,
        e_s0=e_s0,
        T_ref=298.16,
    )  # Shape: (n_cells,)

    # Mole fractions (Gnoffo eq. 7)
    Y_s = (rho_s / M_s[None, :]) / jnp.sum(rho_s / M_s[None, :], axis=-1, keepdims=True)

    # Calculate T_pressure according to Gnoffo eq. 8, 9a, 9b
    # For heavy particles: T_pressure = T
    # For electrons: T_pressure = T_V
    T_pressure = jnp.broadcast_to(T[:, None], (rho_s.shape[0], n_species))

    electron_idx = equation_manager.species.electron_index
    if electron_idx is not None:
        T_pressure = T_pressure.at[:, electron_idx].set(T_V)

    # Compute partial pressures and total pressure (Gnoffo eq. 8, 9a, 9b)
    # p_s = rho_s * R_s * T = rho_s * (R_universal / M_s) * T
    p_s = rho_s * constants.R_universal / M_s[None, :] * T_pressure
    p = jnp.sum(p_s, axis=-1)

    # Apply clipping to primitive variables
    clip_config = equation_manager.numerics_config.clipping
    rho = jnp.clip(rho, clip_config.rho_min, clip_config.rho_max)
    Y_s = jnp.clip(Y_s, clip_config.Y_min, clip_config.Y_max)
    T = jnp.clip(T, clip_config.T_min, clip_config.T_max)
    T_V = jnp.clip(T_V, clip_config.Tv_min, clip_config.Tv_max)
    p = jnp.clip(p, clip_config.p_min, clip_config.p_max)

    return Y_s, rho, T, T_V, p


def compute_U_from_primitives(
    Y_s: Float[Array, "n_cells n_species"],
    rho: Float[Array, " n_cells"],
    u: Float[Array, " n_cells"],
    T_tr: Float[Array, " n_cells"],
    T_V: Float[Array, " n_cells"],
    equation_manager: equation_manager_types.EquationManager,
) -> Float[Array, "n_cells n_variables"]:
    """Compute conserved state vector from primitive variables.

    This is the inverse of extract_primitives_from_U.
    State vector: U = [rho_1, rho_2, ..., rho_ns, rho_u, rho_E, rho_Ev]

    This conversion is aligned with Gnoffo et al. (1989) Conservation Equations and
    Physical Models for Hypersonic Air Flows in Thermal and Chemical Nonequilibrium.

    Args:
        Y_s: Mole fractions, shape (n_cells, n_species)
        rho: Total density [kg/m^3], shape (n_cells,)
        u: Velocity [m/s], shape (n_cells,)
        T: Translational temperature [K], shape (n_cells,)
        T_V: Vibrational temperature [K], shape (n_cells,)
        equation_manager: Equation manager containing species data

    Returns:
        U: Conserved variables [rho_1, ..., rho_ns, rho_u, rho_E, rho_Ev],
           shape (n_cells, n_variables)
    """
    n_species = equation_manager.species.n_species
    n_cells = rho.shape[0]
    M_s = equation_manager.species.M_s

    # Convert mole fractions to mass fractions (Gnoffo eq. 88 inverted)
    # Y_s = (rho_s / M_s) / sum(rho_s / M_s) = (c_s / M_s) / sum(c_s / M_s)
    # c_s = Y_s * M_s / sum(Y_s * M_s)
    Y_M = Y_s * M_s[None, :]  # (n_cells, n_species)
    c_s = Y_M / jnp.sum(Y_M, axis=-1, keepdims=True)  # Mass fractions

    # Partial densities (Gnoffo eq. 6 inverted)
    rho_s = c_s * rho[:, None]  # (n_cells, n_species)

    # Momentum
    rho_u = rho * u

    # Kinetic energy per unit mass
    E_kin = 0.5 * u**2

    # Compute vibrational energy per unit mass (Gnoffo eq. 98)
    e_v_species = thermodynamic_relations.compute_e_ve(
        T_V, equation_manager.species
    )  # (n_species, n_cells)
    E_v = jnp.sum(c_s.T * e_v_species, axis=0)  # Mixture vibrational energy (n_cells,)

    # Compute translational-rotational specific heat
    cv_tr_all = thermodynamic_relations.compute_cv_tr(
        T_tr, equation_manager.species
    )  # (n_species, n_cells)

    # Compute reference internal energies
    e_s0 = thermodynamic_relations.compute_reference_internal_energy(
        equation_manager.species.h_s0,
        equation_manager.species.molar_masses,
        T_ref=298.16,
    )  # (n_species,)

    # Mixture reference energy: e_0 = sum(c_s * e_s0)
    e_0 = jnp.sum(c_s * e_s0[None, :], axis=-1)  # (n_cells,)

    # Mixture translational-rotational specific heat
    cv_tr_mix = jnp.sum(c_s.T * cv_tr_all, axis=0)  # (n_cells,)

    # Internal energy from Gnoffo eq. 102, 103 (inverted from solve_T_from_internal_energy)
    # T = T_ref + (e - e_V - e_0) / C_v,tr
    # => e = e_0 + e_V + C_v,tr * (T - T_ref)
    T_ref = 298.16
    e = e_0 + E_v + cv_tr_mix * (T_tr - T_ref)

    # Total energy per unit mass (Gnoffo eq. 13 inverted)
    E_total = e + E_kin

    # Conserved variables
    rho_E = rho * E_total
    rho_Ev = rho * E_v

    # Assemble state vector U = [rho_1, ..., rho_ns, rho_u, rho_E, rho_Ev]
    n_variables = n_species + 3
    U = jnp.zeros((n_cells, n_variables))
    U = U.at[:, :n_species].set(rho_s)
    U = U.at[:, n_species].set(rho_u)
    U = U.at[:, n_species + 1].set(rho_E)
    U = U.at[:, n_species + 2].set(rho_Ev)

    return U
