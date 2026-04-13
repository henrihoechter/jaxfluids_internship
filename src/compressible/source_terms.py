"""Source terms for the multi-species two-temperature compressible equations."""

import jax
import jax.numpy as jnp
from jaxtyping import Float, Array

from compressible.equation_manager_types import EquationManager
from compressible import state as state_module
from . import chemistry
from . import constants
from . import thermodynamic_relations


@jax.named_call
def compute_source_terms(
    U: Float[Array, "n_cells n_variables"],
    equation_manager: EquationManager,
    primitives: state_module.Primitives | None = None,
) -> Float[Array, "n_cells n_variables"]:
    """Compute all source terms for the two-temperature equations.

    Combines species mass production, vibrational-translational relaxation, and
    vibrational reactive source. Momentum and total energy source terms are zero.

    Args:
        U: Conserved state.
        equation_manager: Physics and numerics configuration.
        primitives: Pre-extracted primitives (computed if None).

    Returns:
        Source array of the same shape as U. Species slots hold mass production
        rates [kg/m^3/s] and the last slot holds the combined vibrational source
        [W/m^3].
    """
    n_cells, n_variables = U.shape
    n_species = equation_manager.species.n_species

    S = jnp.zeros((n_cells, n_variables))

    if primitives is None:
        primitives = state_module.extract_primitives(U, equation_manager)

    omega_dot, Q_vib_chem = compute_chemical_source(
        U, equation_manager, primitives=primitives
    )

    S = S.at[:, :n_species].set(omega_dot)

    Q_TV = compute_vibrational_relaxation(U, equation_manager, primitives=primitives)

    Q_VV = jnp.zeros_like(Q_TV)  # TODO: implement vibrational-vibrational relaxation
    Q_eT = jnp.zeros_like(Q_TV)  # TODO: implement electron-translational relaxation
    Q_ion = jnp.zeros_like(Q_TV)  # TODO: implement electron-impact ionization loss

    S = S.at[:, n_variables - 1].set(Q_TV + Q_VV + Q_eT + Q_ion + Q_vib_chem)

    return S


def compute_chemical_source(
    U: Float[Array, "n_cells n_variables"],
    equation_manager: EquationManager,
    primitives: state_module.Primitives | None = None,
) -> tuple[
    Float[Array, "n_cells n_species"],
    Float[Array, " n_cells"],
]:
    """Compute chemical reaction source terms.

    Returns zeros for both outputs when chemistry is frozen (reactions=None).

    Args:
        U: Conserved state.
        equation_manager: Contains species table and reaction data.
        primitives: Pre-extracted primitives (computed if None).

    Returns:
        Tuple of (omega_dot, Q_vib_chem): species mass production rates [kg/m^3/s]
        and vibrational reactive source [W/m^3].
    """
    n_cells = U.shape[0]
    n_species = equation_manager.species.n_species

    if equation_manager.reactions is None:
        return jnp.zeros((n_cells, n_species)), jnp.zeros(n_cells)

    if primitives is None:
        primitives = state_module.extract_primitives(U, equation_manager)

    rho_s = U[:, :n_species]
    omega_dot, Q_vib_chem = chemistry.compute_all_chemical_sources(
        rho_s=rho_s,
        T=primitives.T,
        T_v=primitives.Tv,
        species_table=equation_manager.species,
        reaction_table=equation_manager.reactions,
    )

    return omega_dot, Q_vib_chem


def compute_vibrational_relaxation(
    U: Float[Array, "n_cells n_variables"],
    equation_manager: EquationManager,
    primitives: state_module.Primitives | None = None,
) -> Float[Array, " n_cells"]:
    """Compute the vibrational-translational relaxation source term.

    Per-species summation: Q = sum_s rho * c_s * (e_vs(T) - e_vs(Tv)) / tau_s.
    Atoms have tau = 1e30 and contribute negligibly.

    Args:
        U: Conserved state.
        equation_manager: Contains species table.
        primitives: Pre-extracted primitives (computed if None).

    Returns:
        Q_dot_v: Relaxation source [W/m^3].
    """
    if primitives is None:
        primitives = state_module.extract_primitives(U, equation_manager)

    Y_s = primitives.Y_s
    rho = primitives.rho
    T = primitives.T
    T_v = primitives.Tv
    p = primitives.p

    e_v_eq = thermodynamic_relations.compute_e_ve(T, equation_manager.species)
    e_v_actual = thermodynamic_relations.compute_e_ve(T_v, equation_manager.species)

    tau_v = compute_relaxation_time_casseau(Y_s, rho, T, p, equation_manager)

    delta_e_v_s = e_v_eq - e_v_actual

    M_s = equation_manager.species.molar_masses
    Y_M = Y_s * M_s[None, :]
    c_s = Y_M / jnp.sum(Y_M, axis=1, keepdims=True)
    Q_s = rho[None, :] * c_s.T * delta_e_v_s / (tau_v + 1e-30)

    return jnp.sum(Q_s, axis=0)


def compute_relaxation_time_casseau(
    Y_s: Float[Array, "n_cells n_species"],
    rho: Float[Array, " n_cells"],
    T: Float[Array, " n_cells"],
    p: Float[Array, " n_cells"],
    equation_manager: EquationManager,
) -> Float[Array, "n_species n_cells"]:
    """Compute vibrational relaxation time using pairwise MW and mixture-effective Park data.

    Mixture average for each vibrating species m:
        tau_m = sum_s(X_s) / sum_s(X_s / (tau_mw_ms + tau_p_ms))

    Atoms and electrons receive tau = 1e30. Based on Casseau's formulation.

    Args:
        Y_s: Mole fractions.
        rho: Density [kg/m^3].
        T: Translational temperature [K].
        p: Pressure [Pa].
        equation_manager: Contains species data.

    Returns:
        tau_full: Relaxation time per species [s].
    """
    species = equation_manager.species
    n_cells = T.shape[0]

    M_s = species.molar_masses  # [kg/mol]

    molecule_indices = species.vibrational_relaxation_molecule_indices
    partner_indices = species.vibrational_relaxation_partner_indices
    a_ms = species.vibrational_relaxation_a_ms
    b_ms = species.vibrational_relaxation_b_ms

    M_mix = jnp.sum(Y_s * M_s[None, :], axis=1)  # [kg/mol]
    n_tot = rho * constants.N_A / (M_mix + 1e-300)  # [1/m^3]
    n_s = Y_s * n_tot[:, None]

    X_s = jnp.take(Y_s, partner_indices, axis=1)
    n_partner = jnp.take(n_s, partner_indices, axis=1)

    p_atm = p / constants.ATM_TO_PA  # [atm]

    M_m = jnp.take(M_s, molecule_indices)
    M_p = jnp.take(M_s, partner_indices)
    sigma_species = species.park_sigma_species
    sigma_p = jnp.take(sigma_species, partner_indices)
    M_m_amu = M_m * 1000.0  # kg/mol -> g/mol
    M_p_amu = M_p * 1000.0
    mu_ms = (M_m_amu[:, None] * M_p_amu[None, :]) / (
        M_m_amu[:, None] + M_p_amu[None, :] + 1e-30
    )

    b_default = 0.015 * mu_ms**0.25
    b_ms = jnp.where(jnp.isnan(b_ms), b_default, b_ms)
    a_ms = jnp.where(jnp.isnan(a_ms), 0.0, a_ms)

    T_term = T ** (-1.0 / 3.0)
    exp_arg = a_ms[None, :, :] * (T_term[:, None, None] - b_ms[None, :, :]) - 18.42
    exp_arg = jnp.clip(exp_arg, -700.0, 700.0)
    tau_mw_ms = jnp.exp(exp_arg) / jnp.clip(p_atm, 1e-300, None)[:, None, None]

    # Casseau Eq. 2.62 uses sigma_v,m without a partner index, so the Park
    # correction should use a single effective sigma for each vibrating mode
    # within the current mixture rather than a distinct sigma_m-s for each pair.
    X_partner_sum = jnp.sum(X_s, axis=1, keepdims=True)
    sigma_eff = jnp.sum(X_s * sigma_p[None, :], axis=1, keepdims=True) / jnp.clip(
        X_partner_sum, 1e-300, None
    )
    sigma_v_m = sigma_eff * (50000.0 / jnp.clip(T[:, None], 1e-12, None)) ** 2

    cbar_m = jnp.sqrt(
        8.0 * constants.R_universal * T[:, None] / (jnp.pi * (M_m[None, :] + 1e-300))
    )

    denom_p = (
        cbar_m[:, :, None]
        * sigma_v_m[:, :, None]
        * jnp.clip(n_partner[:, None, :], 1e-300, None)
    )
    tau_p_ms = 1.0 / (denom_p + 1e-300)

    tau_ms = tau_mw_ms + tau_p_ms

    num = jnp.sum(X_s, axis=1)
    denom = jnp.sum(X_s[:, None, :] / jnp.clip(tau_ms, 1e-300, None), axis=2)
    tau_mix = num[:, None] / jnp.clip(denom, 1e-300, None)

    tau_full = jnp.full((species.n_species, n_cells), 1e30)
    tau_full = tau_full.at[molecule_indices, :].set(tau_mix.T)

    return tau_full


def compute_electron_neutral_collision_frequency(
    n_s: Float[Array, "n_cells n_species"],
    T_e: Float[Array, " n_cells"],
    M_e: float,
    sigma_es_a: Float[Array, " n_species"],
    sigma_es_b: Float[Array, " n_species"],
    sigma_es_c: Float[Array, " n_species"],
) -> Float[Array, "n_cells n_species"]:
    """Compute electron-neutral collision frequency (Eq. 65-66, NASA TP-2867).

    Eq. 65: nu_es = n_s * sigma_es * c_e
    Eq. 66: sigma_es = a + b*T_e + c*T_e^2

    Args:
        n_s: Number density [1/m^3].
        T_e: Electron temperature [K].
        M_e: Electron molar mass [kg/mol].
        sigma_es_a: Cross-section coefficient a [m^2].
        sigma_es_b: Cross-section coefficient b [m^2/K].
        sigma_es_c: Cross-section coefficient c [m^2/K^2].

    Returns:
        nu_es: Collision frequency [1/s]. Zero for species with NaN coefficients.
    """
    sigma_es = (
        sigma_es_a[None, :]
        + sigma_es_b[None, :] * T_e[:, None]
        + sigma_es_c[None, :] * T_e[:, None] ** 2
    )
    # NaN coefficients indicate ions/electrons that use the Coulomb formula instead.
    sigma_es = jnp.where(jnp.isnan(sigma_es), 0.0, sigma_es)

    m_e = M_e / constants.N_A
    c_e = jnp.sqrt(8.0 * constants.k * T_e / (jnp.pi * m_e))

    return n_s * sigma_es * c_e[:, None]


def compute_electron_ion_collision_frequency(
    n_s: Float[Array, "n_cells n_species"],
    n_e: Float[Array, " n_cells"],
    T_e: Float[Array, " n_cells"],
    M_e: float,
) -> Float[Array, "n_cells n_species"]:
    """Compute Coulomb collision frequency for electron-ion collisions (SI units).

    Derived from NASA TP-2867 Eq. 64 using the Debye screening length to evaluate
    the Coulomb logarithm in SI units.

    Args:
        n_s: Ion number densities [1/m^3].
        n_e: Electron number density [1/m^3].
        T_e: Electron temperature [K].
        M_e: Electron molar mass [kg/mol].

    Returns:
        nu_es: Coulomb collision frequency [1/s].
    """
    # TODO: this is not like proposed by gnoffo

    m_e = M_e / constants.N_A

    epsilon_0 = constants.epsilon_0
    e_SI = constants.e

    lambda_D_sq = epsilon_0 * constants.k * T_e / (n_e * e_SI**2 + 1e-30)
    ln_arg = 12.0 * jnp.pi * n_e * lambda_D_sq**1.5
    ln_Lambda = jnp.log(jnp.maximum(ln_arg, 1.0))

    prefactor = e_SI**4 / (16.0 * jnp.pi * epsilon_0**2 * jnp.sqrt(m_e))
    nu_es = prefactor * n_s * ln_Lambda[:, None] / (constants.k * T_e[:, None]) ** 1.5

    return nu_es


def compute_eT_relaxation(
    U: Float[Array, "n_cells n_variables"],
    equation_manager: EquationManager,
) -> Float[Array, " n_cells"]:
    """Compute electron-translational energy relaxation source term (Term 7).

    Q_eT = 2 * rho_e * (3R / 2M_e) * (T - T_v) * sum_s(nu_es / M_s)

    Returns zero if no electron species is present.

    Args:
        U: Conserved state.
        equation_manager: Contains species data.

    Returns:
        Q_eT: Source term for vibrational-electronic energy [W/m^3].
    """
    species = equation_manager.species
    n_cells = U.shape[0]

    electron_idx = species.electron_index
    if electron_idx is None:
        return jnp.zeros(n_cells)

    Y_s, rho, _, _, T, T_v, p = state_module.extract_primitives_from_U(
        U, equation_manager
    )

    M_s = species.molar_masses
    Y_M = Y_s * M_s[None, :]
    c_s = Y_M / jnp.sum(Y_M, axis=1, keepdims=True)
    c_e = c_s[:, electron_idx]
    rho_e = rho * c_e
    M_e = species.molar_masses[electron_idx]
    n_e = rho_e * constants.N_A / M_e
    T_e = T_v  # In 2-temp model, T_e = T_v

    n_s = rho[:, None] * c_s * constants.N_A / species.molar_masses[None, :]

    nu_es = jnp.zeros((n_cells, species.n_species))

    nu_es_neutral = compute_electron_neutral_collision_frequency(
        n_s,
        T_e,
        M_e,
        species.sigma_es_a,
        species.sigma_es_b,
        species.sigma_es_c,
    )
    neutral_mask = species.is_neutral
    nu_es = jnp.where(neutral_mask[None, :], nu_es_neutral, nu_es)

    nu_es_ion = compute_electron_ion_collision_frequency(n_s, n_e, T_e, M_e)
    ion_mask = species.is_ion
    nu_es = jnp.where(ion_mask[None, :], nu_es_ion, nu_es)

    heavy_mask = jnp.logical_not(species.is_electron)
    sum_nu_over_M = jnp.sum(
        jnp.where(heavy_mask[None, :], nu_es / species.molar_masses[None, :], 0.0),
        axis=1,
    )

    Q_eT = 2.0 * rho_e * (3.0 * constants.R_universal / 2.0) * (T - T_v) * sum_nu_over_M

    return Q_eT


def compute_electron_impact_ionization_loss(
    U: Float[Array, "n_cells n_variables"],
    equation_manager: EquationManager,
) -> Float[Array, " n_cells"]:
    """Compute electron energy loss from electron impact ionization (Term 8).

    Currently returns zero (frozen chemistry; requires active ionization reactions).

    Args:
        U: Conserved state.
        equation_manager: Contains species and reaction data.

    Returns:
        Q_ion: Energy loss rate [W/m^3].
    """
    return jnp.zeros(U.shape[0])
