"""Source terms module for multi-species two-temperature equations.

Implements vibrational relaxation source terms.
"""

import jax.numpy as jnp
from jaxtyping import Float, Array

from compressible_1d import constants
from compressible_1d import equation_manager_types
from compressible_1d import equation_manager_utils
from compressible_1d import thermodynamic_relations
from compressible_1d import chemistry


def compute_source_terms(
    U: Float[Array, "n_cells n_variables"],
    equation_manager: equation_manager_types.EquationManager,
) -> Float[Array, "n_cells n_variables"]:
    """Compute all source terms: chemistry + vibrational relaxation.

    For frozen chemistry (reactions=None): only vibrational relaxation is active.
    For reacting flow: includes species production and vibrational reactive source.

    Args:
        U: Conserved state [n_cells, n_variables]
        equation_manager: Contains species table, reactions, and config

    Returns:
        S: Source terms [n_cells, n_variables]
            S[:, 0:n_species] = ω̇_s (species mass production rates)
            S[:, n_species] = 0 (momentum, inviscid)
            S[:, n_species+1] = 0 (no chemical energy source)
            S[:, n_species+2] = Q_TV + Q_vib_chem (vibrational energy)

    Notes:
        - Frozen chemistry: species source terms are zero (ω̇_i = 0)
        - No momentum source (inviscid)
        - No total energy source for inviscid frozen chemistry
        - Vibrational energy relaxation: Q̇_v = ρ(e_v(T) - e_v(T_v))/τ_v
    """
    n_cells, n_variables = U.shape
    n_species = equation_manager.species.n_species

    # Initialize source terms to zero
    S = jnp.zeros((n_cells, n_variables))

    # === Chemical Source Terms ===
    omega_dot, Q_vib_chem = compute_chemical_source(U, equation_manager)

    # Species production rates
    S = S.at[:, :n_species].set(omega_dot)

    # No momentum source (inviscid)
    # S[:, n_species] = 0

    # === Vibrational-Electronic Energy Sources (Eq. 16 from NASA TP-2867) ===

    # Term 6: Vibrational-translational relaxation
    Q_TV = compute_vibrational_relaxation(U, equation_manager)
    # Q_TV = jnp.zeros_like(Q_TV) 

    Q_VV = jnp.zeros_like(Q_TV)  # TODO: implement vibrational-vibrational relaxation
    Q_eT = jnp.zeros_like(Q_TV)  # TODO: implement electron-translational relaxation
    Q_ion = jnp.zeros_like(Q_TV)  # TODO: implement electron-impact ionization loss

    # Term 7: Electron-translational relaxation
    # Q_eT = compute_eT_relaxation(U, equation_manager)

    # Term 8: Electron impact ionization loss (= 0 for frozen chemistry)
    # Q_ion = compute_electron_impact_ionization_loss(U, equation_manager)

    # Total source for vibrational-electronic energy (last variable = ρE_v)
    S = S.at[:, n_species + 2].set(Q_TV + Q_VV + Q_eT + Q_ion + Q_vib_chem)

    return S


def compute_chemical_source(
    U: Float[Array, "n_cells n_variables"],
    equation_manager: equation_manager_types.EquationManager,
) -> tuple[
    Float[Array, "n_cells n_species"],
    Float[Array, " n_cells"],
]:
    """Compute chemical reaction source terms.

    For frozen chemistry (reactions=None): returns zeros.
    For reacting flow: computes species production rates and vibrational reactive source.

    Args:
        U: Conserved state [n_cells, n_variables]
        equation_manager: Contains species table and reaction data

    Returns:
        omega_dot: Species mass production rates [kg/m³/s]. Shape [n_cells, n_species].
        Q_vib_chem: Vibrational reactive source [W/m³]. Shape [n_cells].
    """
    n_cells = U.shape[0]
    n_species = equation_manager.species.n_species

    if equation_manager.reactions is None:
        omega_dot = jnp.zeros((n_cells, n_species))
        Q_vib_chem = jnp.zeros(n_cells)
        return omega_dot, Q_vib_chem

    # Extract primitives (only need T and T_v for rate calculations)
    _, _, T, T_v, _ = equation_manager_utils.extract_primitives_from_U(
        U, equation_manager
    )

    rho_s = U[:, :n_species]

    omega_dot, Q_vib_chem = chemistry.compute_all_chemical_sources(
        rho_s=rho_s,
        T=T,
        T_v=T_v,
        species_table=equation_manager.species,
        reaction_table=equation_manager.reactions,
    )

    return omega_dot, Q_vib_chem


def compute_vibrational_relaxation(
    U: Float[Array, "n_cells n_variables"],
    equation_manager: equation_manager_types.EquationManager,
) -> Float[Array, "n_cells"]:
    """Compute vibrational relaxation source term Q_dot_v.

    Per-species summation (each species relaxes at its own rate):
        Q_dot_v = Σ_s [rho * c_s * (e_v_s(T) - e_v_s(T_v)) / tau_s]

    where:
    - e_v_s(T): equilibrium vibrational energy of species s at temperature T
    - e_v_s(T_v): actual vibrational energy of species s at T_v
    - tau_s: Millikan-White relaxation time with Park correction for species s

    Args:
        U: Conserved state [n_cells, n_variables]
        equation_manager: Contains species table

    Returns:
        Q_dot_v: Relaxation source term [n_cells] in W/m³
    """
    Y_s, rho, T, T_v, p = equation_manager_utils.extract_primitives_from_U(
        U, equation_manager
    )

    # Compute equilibrium vibrational energy at T per species
    # Shape: [n_species, n_cells]
    e_v_eq_species = thermodynamic_relations.compute_e_ve(T, equation_manager.species)

    # Compute actual vibrational energy at T_v per species
    # Shape: [n_species, n_cells]
    e_v_actual_species = thermodynamic_relations.compute_e_ve(
        T_v, equation_manager.species
    )

    # Compute per-species relaxation time
    # Shape: [n_species, n_cells]
    # tau_v = compute_relaxation_time(Y_s, rho, T, T_v, p, equation_manager)
    tau_v = compute_relaxation_time_2_casseau(Y_s, rho, T, p, equation_manager)

    # Compute per-species energy difference
    # delta_e_v_s has shape [n_species, n_cells]
    delta_e_v_s = e_v_eq_species - e_v_actual_species  # [n_species, n_cells]

    # Compute per-species relaxation rate: Q_s = rho * c_s * delta_e_v_s / tau_s
    # Atoms have tau = 1e30, so their contribution is effectively zero
    # c_s has shape [n_cells, n_species], need to transpose for broadcasting
    M_s = equation_manager.species.molar_masses
    Y_M = Y_s * M_s[None, :]
    c_s = Y_M / jnp.sum(Y_M, axis=1, keepdims=True)
    Q_s = rho[None, :] * c_s.T * delta_e_v_s / (tau_v + 1e-30)  # [n_species, n_cells]

    # Sum over all species (atoms contribute ~0 due to large tau and zero delta_e_v)
    Q_dot_v = jnp.sum(Q_s, axis=0)  # [n_cells]

    return Q_dot_v


def compute_relaxation_time(
    Y_s: Float[Array, "n_cells n_species"],
    rho: Float[Array, "n_cells"],
    T: Float[Array, "n_cells"],
    T_v: Float[Array, "n_cells"],
    p: Float[Array, "n_cells"],
    equation_manager: equation_manager_types.EquationManager,
) -> Float[Array, "n_species n_cells"]:
    """Compute Millikan-White relaxation time with Park correction per species.

    Implements equations (55), (56), and (58) from NASA TP-2867.

    Equation (55) - Millikan-White correlation:
        p·tau_s^MW = [Σ_j n_j · exp[A_s(T^(-1/3) - 0.015·mu_sj^(1/4)) - 18.42]] / [Σ_j n_j]

    Equation (56) - Park high-temperature correction:
        tau_s^P = (sigma_s · c_bar_s · n_s)^(-1)

    Equation (58) - Blending:
        <tau_s> = tau_s^MW + tau_s^P

    Args:
        Y_s: Mole fractions [n_cells n_species]
        rho: Density [n_cells]
        T: Translational temperature [n_cells]
        T_v: Vibrational temperature [n_cells]
        p: Pressure [n_cells]
        equation_manager: Contains species data

    Returns:
        tau_v: Relaxation time per species [n_species, n_cells] in seconds.
               Atoms have tau = 1e30 (effectively infinite).
    """
    species = equation_manager.species
    n_cells = T.shape[0]
    n_species = species.n_species

    # Extract species properties
    M_s = species.molar_masses  # [kg/mol]
    A_s = species.vibrational_relaxation_factor  # [-], NaN for atoms
    is_monoatomic = species.is_monoatomic  # [n_species]

    # Convert mole fractions to number densities
    # n_total = rho * N_A / M_mix, n_j = X_j * n_total
    # n_j has shape [n_cells, n_species]
    M_mix = jnp.sum(Y_s * M_s[None, :], axis=1)  # [kg/mol]
    n_total = rho * constants.N_A / (M_mix + 1e-30)  # [1/m³]
    n_j = Y_s * n_total[:, None]

    # Total number density
    # TODO: this does not reflect electrons to not be considered
    n_total = jnp.sum(n_j, axis=1)  # [n_cells]

    # Pressure in atmospheres
    p_atm = p / constants.ATM_TO_PA  # [atm]

    # Compute reduced masses for all species pairs
    # mu_sj = M_s * M_j / (M_s + M_j) in amu (use M_s in g/mol which equals amu)
    # Shape: [n_species, n_species]
    # Convert kg/mol -> g/mol (= amu) for reduced mass calculation
    M_s_amu = M_s * 1000.0  # g/mol = amu
    mu_sj = (M_s_amu[:, None] * M_s_amu[None, :]) / (
        M_s_amu[:, None] + M_s_amu[None, :] + 1e-30
    )  # [n_species, n_species]

    # Compute Millikan-White relaxation time for each molecular species
    # tau_MW for species s: weighted average over collision partners j
    # p * tau_MW_s = [Σ_j n_j * exp[A_s * (T^(-1/3) - 0.015 * mu_sj^0.25) - 18.42]] / [Σ_j n_j]

    # Compute the exponent argument for all pairs
    # T^(-1/3) has shape [n_cells]
    T_term = T ** (-1.0 / 3.0)  # [n_cells]

    # For each species s, compute sum over collision partners j
    # tau_MW shape: [n_cells, n_species]
    tau_MW = jnp.zeros((n_cells, n_species))

    # Vectorized computation over all species pairs
    # exp_arg[s, j] = A_s * (T^(-1/3) - 0.015 * mu_sj^0.25) - 18.42
    # Shape: [n_cells, n_species, n_species] for full computation

    # A_s values, replace NaN with 0 for computation (atoms won't be used anyway)
    A_s_safe = jnp.where(jnp.isnan(A_s), 0.0, A_s)  # [n_species]

    # Compute exponent for all (s, j) pairs at all cells
    # mu_sj^0.25 has shape [n_species, n_species]
    mu_term = 0.015 * (mu_sj**0.25)  # [n_species, n_species]

    # exp_arg[cell, s, j] = A_s[s] * (T_term[cell] - mu_term[s, j]) - 18.42
    # Broadcasting: T_term[n_cells, 1, 1], A_s[1, n_species, 1], mu_term[1, n_species, n_species]
    exp_arg = (
        A_s_safe[None, :, None] * (T_term[:, None, None] - mu_term[None, :, :]) - 18.42
    )  # [n_cells, n_species, n_species]

    # Clamp exponent to avoid overflow (exp(700) is close to float64 max)
    exp_arg = jnp.clip(exp_arg, -700, 700)

    # exp_values[cell, s, j] = exp(exp_arg[cell, s, j])
    exp_values = jnp.exp(exp_arg)  # [n_cells, n_species, n_species]

    # Weighted sum: numerator[cell, s] = sum_j(n_j[cell, j] * exp_values[cell, s, j])
    # n_j has shape [n_cells, n_species], need [n_cells, 1, n_species] for broadcasting
    numerator = jnp.sum(n_j[:, None, :] * exp_values, axis=2)  # [n_cells, n_species]

    # tau_MW_s = numerator / (p_atm * n_total)
    # Add small epsilon to avoid division by zero
    tau_MW = numerator / (
        p_atm[:, None] * n_total[:, None] + 1e-30
    )  # [n_cells, n_species]

    # Compute Park correction for each molecular species
    # tau_P_s = 1 / (sigma * c_bar_s * n_s)
    # c_bar_s = sqrt(8 * k * T / (pi * m_s))
    # m_s = M_s / N_A (mass per molecule in kg)

    m_s = M_s / constants.N_A  # [kg] mass per molecule, shape [n_species]

    # c_bar_s[cell, s] = sqrt(8 * k * T[cell] / (pi * m_s[s]))
    c_bar_s = jnp.sqrt(
        8.0 * constants.k * T[:, None] / (jnp.pi * m_s[None, :])
    )  # [n_cells, n_species]

    # tau_P_s = 1 / (sigma * c_bar_s * n_s)
    # n_s is the number density of molecule s itself

    # SIGMA_V = 1e-20  # [m²] Park's effective cross section for vibrational relaxation (10^-16 cm²)
    SIGMA_V = (
        3e-21 * (50000.0 / T) ** 2
    )  # [m2] definition according to Casseau, Eq. 2.64
    tau_P = 1.0 / (SIGMA_V * c_bar_s * n_j + 1e-30)  # [n_cells, n_species]

    # Blend: <tau_s> = tau_MW + tau_P (Eq. 58)
    tau_v = tau_MW + tau_P  # [n_cells, n_species]

    # Set tau = 1e30 for monoatomic species (no vibrational relaxation)
    tau_v = jnp.where(is_monoatomic[None, :], 1e30, tau_v)

    # Transpose to [n_species, n_cells] for consistency with other arrays
    return tau_v.T


def compute_relaxation_time_2_casseau(
    Y_s: Float[Array, "n_cells n_species"],
    rho: Float[Array, "n_cells"],
    T: Float[Array, "n_cells"],
    p: Float[Array, "n_cells"],
    equation_manager: equation_manager_types.EquationManager,
) -> Float[Array, "n_species n_cells"]:
    """Compute vibrational relaxation time using pairwise Park data (Casseau).

    Pairwise (molecular species m, collision partner s):
        tau_m-s = tau_m-s^MW + tau_m-s^P
        tau_m-s^P = 1 / (cbar_m * sigma_v,m * n_s)

    Mixture average (for each vibrating species m):
        tau_m = (sum_s X_s) / (sum_s X_s / tau_m-s)

    Notes:
    - Assumes Y_s are mole fractions X_s.
    - Electrons are excluded from collision partners.
    - Only molecular species are used for tau_m-s; atoms and electrons get tau = 1e30.
    """
    species = equation_manager.species
    n_cells = T.shape[0]

    M_s = species.molar_masses  # [kg/mol]

    molecule_indices = species.vibrational_relaxation_molecule_indices
    partner_indices = species.vibrational_relaxation_partner_indices
    a_ms = species.vibrational_relaxation_a_ms  # [m, s]
    b_ms = species.vibrational_relaxation_b_ms  # [m, s]

    # --- Number densities ---
    M_mix = jnp.sum(Y_s * M_s[None, :], axis=1)  # [kg/mol]
    n_tot = rho * constants.N_A / (M_mix + 1e-300)  # [n_cells] 1/m^3
    n_s = Y_s * n_tot[:, None]  # [n_cells, n_species] 1/m^3

    X_s = jnp.take(Y_s, partner_indices, axis=1)  # [n_cells, s]
    n_partner = jnp.take(n_s, partner_indices, axis=1)  # [n_cells, s]

    # Pressure in atm (MW formula uses atm)
    p_atm = p / constants.ATM_TO_PA

    # --- Reduced masses in "amu" = g/mol for molecules vs partners ---
    M_m = jnp.take(M_s, molecule_indices)  # [m]
    M_p = jnp.take(M_s, partner_indices)  # [s]
    M_m_amu = M_m * 1000.0  # kg/mol -> g/mol
    M_p_amu = M_p * 1000.0  # kg/mol -> g/mol
    mu_ms = (M_m_amu[:, None] * M_p_amu[None, :]) / (
        M_m_amu[:, None] + M_p_amu[None, :] + 1e-30
    )  # [m, s]

    b_default = 0.015 * mu_ms**0.25
    b_ms = jnp.where(jnp.isnan(b_ms), b_default, b_ms)
    a_ms = jnp.where(jnp.isnan(a_ms), 0.0, a_ms)

    T_term = T ** (-1.0 / 3.0)  # [n_cells]

    # tau_MW[cell, m, s] = (1/p_atm) * exp(a_ms * (T^-1/3 - b_ms) - 18.42)
    exp_arg = a_ms[None, :, :] * (T_term[:, None, None] - b_ms[None, :, :]) - 18.42
    exp_arg = jnp.clip(exp_arg, -700.0, 700.0)
    tau_mw_ms = jnp.exp(exp_arg) / jnp.clip(p_atm, 1e-300, None)[:, None, None]

    # --- Park correction, computed pairwise (m,s) ---
    sigma_m = 3e-21  # [m^2], Casseau
    sigma_v = sigma_m * (50000.0 / jnp.clip(T, 1e-12, None)) ** 2  # [n_cells] m^2

    cbar_m = jnp.sqrt(
        8.0 * constants.R_universal * T[:, None] / (jnp.pi * (M_m[None, :] + 1e-300))
    )  # [n_cells, m]

    denom_p = (
        cbar_m[:, :, None]
        * sigma_v[:, None, None]
        * jnp.clip(n_partner[:, None, :], 1e-300, None)
    )
    tau_p_ms = 1.0 / (denom_p + 1e-300)  # [cells, m, s]

    tau_ms = tau_mw_ms + tau_p_ms  # [cells, m, s]

    num = jnp.sum(X_s, axis=1)  # [cells]
    denom = jnp.sum(X_s[:, None, :] / jnp.clip(tau_ms, 1e-300, None), axis=2)
    tau_mix = num[:, None] / jnp.clip(denom, 1e-300, None)  # [cells, m]

    tau_full = jnp.full((species.n_species, n_cells), 1e30)
    tau_full = tau_full.at[molecule_indices, :].set(tau_mix.T)

    return tau_full


def compute_electron_neutral_collision_frequency(
    n_s: Float[Array, "n_cells n_species"],
    T_e: Float[Array, "n_cells"],
    M_e: float,
    sigma_es_a: Float[Array, " n_species"],
    sigma_es_b: Float[Array, " n_species"],
    sigma_es_c: Float[Array, " n_species"],
) -> Float[Array, "n_cells n_species"]:
    """Compute electron-neutral collision frequency nu_es.

    Implements equations (65) and (66) from NASA TP-2867:
        Eq. 65: nu_es = n_s * sigma_es * sqrt(8*k*T_e/(pi*m_e))
        Eq. 66: sigma_es = sigma_es_a + sigma_es_b*T_e + sigma_es_c*T_e^2

    Args:
        n_s: Number density of each species [n_cells, n_species] in [1/m³]
        T_e: Electron temperature (= T_V in 2-temp model) [n_cells] in [K]
        M_e: Electron molar mass [kg/mol]
        sigma_es_a: Cross-section coefficient a [n_species] in [m²]
        sigma_es_b: Cross-section coefficient b [n_species] in [m²/K]
        sigma_es_c: Cross-section coefficient c [n_species] in [m²/K²]

    Returns:
        nu_es: Collision frequency [n_cells, n_species] in [1/s]
               Returns 0 for species with NaN coefficients (ions, electrons)
    """
    # Cross-section: sigma_es = sigma_es_a + sigma_es_b*T_e + sigma_es_c*T_e^2
    # Shape: [n_cells, n_species]
    sigma_es = (
        sigma_es_a[None, :]
        + sigma_es_b[None, :] * T_e[:, None]
        + sigma_es_c[None, :] * T_e[:, None] ** 2
    )

    # Handle NaN coefficients (ions/electrons don't use this formula)
    sigma_es = jnp.where(jnp.isnan(sigma_es), 0.0, sigma_es)

    # Electron mass from molar mass: m_e = M_e / N_A
    m_e = M_e / constants.N_A  # [kg]

    # Electron thermal velocity: c_e = sqrt(8*k*T_e/(pi*m_e))
    c_e = jnp.sqrt(8.0 * constants.k * T_e / (jnp.pi * m_e))  # [n_cells]

    # Collision frequency: nu_es = n_s * sigma_es * c_e
    nu_es = n_s * sigma_es * c_e[:, None]

    return nu_es


def compute_electron_ion_collision_frequency(
    n_s: Float[Array, "n_cells n_species"],
    n_e: Float[Array, "n_cells"],
    T_e: Float[Array, "n_cells"],
    M_e: float,
) -> Float[Array, "n_cells n_species"]:
    """Compute Coulomb collision frequency for electron-ion collisions.

    Implements equation (64) from NASA TP-2867:
        nu_es = (8/3)(pi/m_e)^(1/2) × n_s e^4/(2kT_e)^(3/2) × ln[k^3 T_e^3/(pi n_e e^6)]

    Note: The elementary charge 'e' in the NASA TP-2867 formulas is in CGS/esu units
    (statcoulombs). However, using SI units with proper unit conversion gives the
    same result when e is in Coulombs and we use epsilon_0 (permittivity of free space).

    Args:
        n_s: Number density of ions [n_cells, n_species] in [1/m³]
        n_e: Electron number density [n_cells] in [1/m³]
        T_e: Electron temperature [K]
        M_e: Electron molar mass [kg/mol]

    Returns:
        nu_es: Coulomb collision frequency [n_cells, n_species] in [1/s]
    """

    # TODO: this is not like proposed by gnoffo

    # Electron mass from molar mass
    m_e = M_e / constants.N_A  # [kg]

    # For SI units, the Coulomb logarithm becomes:
    # ln_Lambda = ln(12pi n_e lambda_D^3) where lambda_D = sqrt(epsilon_0 k T_e / (n_e e^2))
    # Simplified: ln_Lambda ≈ ln((k T_e)^(3/2) / (e^3 sqrt(pi n_e)))
    # Using practical formula: ln_Lambda ≈ 23 - ln(sqrt(n_e) / T_e^(3/2)) for typical plasma

    epsilon_0 = constants.epsilon_0  # Permittivity of free space [F/m]
    e_SI = constants.e  # Elementary charge [C]

    # Debye length: lambda_D = sqrt(epsilon_0 k T_e / (n_e e^2))
    lambda_D_sq = epsilon_0 * constants.k * T_e / (n_e * e_SI**2 + 1e-30)

    # Coulomb logarithm: ln_Lambda = ln(12pi n_e lambda_D^3)
    # = ln(12pi) + ln(n_e) + 1.5*ln(lambda_D^2)
    # Ensure argument is positive
    ln_arg = 12.0 * jnp.pi * n_e * lambda_D_sq**1.5
    ln_Lambda = jnp.log(jnp.maximum(ln_arg, 1.0))

    # Collision frequency in SI units:
    # nu_ei = (n_i Z^2 e^4 ln_Lambda) / (4pi epsilon_0^2 m_e^2 v_e^3)
    # where v_e = sqrt(2 k T_e / m_e) (thermal velocity)
    # Simplifying: nu_ei = n_i Z^2 e^4 ln_Lambda / (16pi epsilon_0^2 m_e^(1/2) (k T_e)^(3/2))

    # For singly charged ions (Z=1):
    prefactor = e_SI**4 / (16.0 * jnp.pi * epsilon_0**2 * jnp.sqrt(m_e))
    nu_es = prefactor * n_s * ln_Lambda[:, None] / (constants.k * T_e[:, None]) ** 1.5

    return nu_es


def compute_eT_relaxation(
    U: Float[Array, "n_cells n_variables"],
    equation_manager: equation_manager_types.EquationManager,
) -> Float[Array, "n_cells"]:
    """Compute electron-translational energy relaxation source term (Term 7).

    Implements Term 7 from Eq. 16 (NASA TP-2867):
        Q_eT = 2 * rho_e * (3 * R_bar / (2 * M_e)) * (T - T_v) × sum_s nu_es / M_s

    This represents elastic collision energy exchange between electrons
    (at T_v in 2-temp model) and heavy particles (at T).

    Physical interpretation:
    - When T > T_v: heavy particles transfer energy to electrons → positive source
    - When T < T_v: electrons transfer energy to heavy particles → negative source

    Args:
        U: Conserved state [n_cells, n_variables]
        equation_manager: Contains species data

    Returns:
        Q_eT: Source term for vibrational-electronic energy [W/m³]
    """
    species = equation_manager.species
    n_cells = U.shape[0]

    # Check if electrons exist in the species table
    electron_idx = species.electron_index
    if electron_idx is None:
        return jnp.zeros(n_cells)  # No electrons → term = 0

    # Extract primitives
    Y_s, rho, T, T_v, p = equation_manager_utils.extract_primitives_from_U(
        U, equation_manager
    )

    # Electron properties
    M_s = species.molar_masses
    Y_M = Y_s * M_s[None, :]
    c_s = Y_M / jnp.sum(Y_M, axis=1, keepdims=True)
    c_e = c_s[:, electron_idx]
    rho_e = rho * c_e  # Electron density [kg/m³]
    M_e = species.molar_masses[electron_idx]  # Electron molar mass [kg/mol]
    n_e = rho_e * constants.N_A / M_e  # Electron number density [1/m³]
    T_e = T_v  # In 2-temp model, T_e = T_v

    # Compute number densities for all species [n_cells, n_species]
    n_s = rho[:, None] * c_s * constants.N_A / species.molar_masses[None, :]

    # Initialize collision frequencies to zero
    nu_es = jnp.zeros((n_cells, species.n_species))

    # Compute collision frequencies for neutrals (Eq. 65-66)
    nu_es_neutral = compute_electron_neutral_collision_frequency(
        n_s,
        T_e,
        M_e,
        species.sigma_es_a,
        species.sigma_es_b,
        species.sigma_es_c,
    )
    # Only use for neutral species (charge == 0)
    neutral_mask = species.is_neutral
    nu_es = jnp.where(neutral_mask[None, :], nu_es_neutral, nu_es)

    # Compute collision frequencies for ions (Eq. 64 - Coulomb collisions)
    nu_es_ion = compute_electron_ion_collision_frequency(n_s, n_e, T_e, M_e)
    # Only use for ionized species (charge > 0)
    ion_mask = species.is_ion
    nu_es = jnp.where(ion_mask[None, :], nu_es_ion, nu_es)

    # Sum: sum_s nu_es / M_s (exclude electrons from the sum)
    heavy_mask = jnp.logical_not(species.is_electron)
    sum_nu_over_M = jnp.sum(
        jnp.where(heavy_mask[None, :], nu_es / species.molar_masses[None, :], 0.0),
        axis=1,
    )

    # Term 7: Q_eT = 2 * rho_e * (3 * R_bar / (2 * M_e)) * (T - T_v) * sum_nu_over_M
    R_bar = constants.R_universal
    Q_eT = 2.0 * rho_e * (3.0 * R_bar / 2.0) * (T - T_v) * sum_nu_over_M

    return Q_eT


def compute_electron_impact_ionization_loss(
    U: Float[Array, "n_cells n_variables"],
    equation_manager: equation_manager_types.EquationManager,
) -> Float[Array, "n_cells"]:
    """Compute electron energy loss due to electron impact ionization (Term 8).

    Implements Term 8 from Eq. 16 (NASA TP-2867):
        Q_ion = -sum_s (n_dot_e_s * I_hat_s)

    where:
        n_dot_e_s = molar ionization rate of species s by electron impact [mol/m³/s]
        I_hat_s = first ionization energy of species s [J/mol]

    For FROZEN CHEMISTRY: returns 0 (no reactions → n_dot_e_s = 0)

    Future implementation requires:
        - Ionization reaction rates from chemical kinetics
        - Reactions: N + e_minus -> N_plus + 2e_minus, O + e_minus -> O_plus + 2e_minus, etc.

    Args:
        U: Conserved state [n_cells, n_variables]
        equation_manager: Contains species and reaction data

    Returns:
        Q_ion: Energy loss rate [W/m³] (negative = energy removed from electron mode)
    """
    # Frozen chemistry: no ionization reactions
    n_cells = U.shape[0]
    return jnp.zeros(n_cells)
