"""Source terms module for multi-species two-temperature equations.

Implements vibrational relaxation and frozen chemistry stubs.
"""

import jax.numpy as jnp
from jaxtyping import Float, Array

from compressible_1d import constants
from compressible_1d import equation_manager_types
from compressible_1d import equation_manager_utils
from compressible_1d import thermodynamic_relations


def compute_source_terms(
    U: Float[Array, "n_cells n_variables"],
    equation_manager: equation_manager_types.EquationManager,
) -> Float[Array, "n_cells n_variables"]:
    """Compute all source terms: chemistry + vibrational relaxation.

    For frozen chemistry: only vibrational relaxation is active.

    Args:
        U: Conserved state [n_cells, n_variables]
        equation_manager: Contains species table and config

    Returns:
        S: Source terms [n_cells, n_variables]

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

    # Frozen chemistry: no species source terms
    # S[:, :n_species] = 0

    # No momentum source (inviscid)
    # S[:, n_species] = 0

    # No total energy source (inviscid, frozen chemistry)
    # S[:, n_species + 1] = 0

    # Vibrational energy relaxation
    # Q_v = compute_vibrational_relaxation(U, equation_manager)
    Q_v = compute_vibrational_relaxation_new(U, equation_manager)
    S = S.at[:, n_species + 2].set(Q_v)  # Last variable is ρE_v

    return S


def compute_vibrational_relaxation(
    U: Float[Array, "n_cells n_variables"],
    equation_manager: equation_manager_types.EquationManager,
) -> Float[Array, "n_cells"]:
    """Compute vibrational relaxation source term Q_dot_v.

    Q_dot_v = rho * (e_v(T) - e_v(T_v)) / tau_v

    where:
    - e_v(T): equilibrium vibrational energy at temperature T
    - e_v(T_v): actual vibrational energy at T_v
    - tau_v: Millikan-White relaxation time with Park correction

    Args:
        U: Conserved state [n_cells, n_variables]
        equation_manager: Contains species table

    Returns:
        Q_dot_v: Relaxation source term [n_cells] in W/m³
    """
    Y_s, rho, T, T_v, p = equation_manager_utils.extract_primitives_from_U(
        U, equation_manager
    )

    # Compute equilibrium vibrational energy at T
    e_v_eq_species = thermodynamic_relations.compute_e_ve(
        T, equation_manager.species
    )  # [n_species, n_cells]
    e_v_eq = jnp.sum(Y_s * e_v_eq_species.T, axis=1)  # [n_cells]

    # Compute actual vibrational energy at T_v
    e_v_species = thermodynamic_relations.compute_e_ve(
        T_v, equation_manager.species
    )  # [n_species, n_cells]
    e_v_actual = jnp.sum(Y_s * e_v_species.T, axis=1)  # [n_cells]

    tau_v = tau_v = jnp.full_like(
        T, 1e-7
    )  # TODO: replace with actual relaxation time computation

    Q_dot_v = rho * (e_v_eq - e_v_actual) / (tau_v + 1e-14)

    return Q_dot_v


def compute_vibrational_relaxation_new(
    U: Float[Array, "n_cells n_variables"],
    equation_manager: equation_manager_types.EquationManager,
) -> Float[Array, "n_cells"]:
    """Compute vibrational relaxation source term Q_dot_v.

    Per-species summation (each species relaxes at its own rate):
        Q_dot_v = Σ_s [rho * Y_s * (e_v_s(T) - e_v_s(T_v)) / tau_s]

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
    tau_v = compute_relaxation_time(Y_s, rho, T, T_v, p, equation_manager)

    # Compute per-species energy difference
    # delta_e_v_s has shape [n_species, n_cells]
    delta_e_v_s = e_v_eq_species - e_v_actual_species  # [n_species, n_cells]

    # Compute per-species relaxation rate: Q_s = rho * Y_s * delta_e_v_s / tau_s
    # Atoms have tau = 1e30, so their contribution is effectively zero
    # Y_s has shape [n_cells, n_species], need to transpose for broadcasting
    Q_s = rho[None, :] * Y_s.T * delta_e_v_s / (tau_v + 1e-30)  # [n_species, n_cells]

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
        Y_s: Mass fractions [n_cells n_species]
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

    # Compute number densities n_j = rho * Y_s * N_A / M_s
    # n_j has shape [n_cells, n_species]
    n_j = rho[:, None] * Y_s * constants.N_A / M_s[None, :]  # [1/m³]

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
    SIGMA_V = 1e-20  # [m²] Park's effective cross section for vibrational relaxation (10^-16 cm²)
    tau_P = 1.0 / (SIGMA_V * c_bar_s * n_j + 1e-30)  # [n_cells, n_species]

    # Blend: <tau_s> = tau_MW + tau_P (Eq. 58)
    tau_v = tau_MW + tau_P  # [n_cells, n_species]

    # Set tau = 1e30 for monoatomic species (no vibrational relaxation)
    tau_v = jnp.where(is_monoatomic[None, :], 1e30, tau_v)

    # Transpose to [n_species, n_cells] for consistency with other arrays
    return tau_v.T


def compute_chemical_source(
    U: Float[Array, "n_cells n_variables"],
    equation_manager: equation_manager_types.EquationManager,
) -> Float[Array, "n_cells n_variables"]:
    """Compute chemical reaction source terms.

    For frozen chemistry: returns zeros.

    Args:
        U: Conserved state [n_cells, n_variables]
        equation_manager: Contains reaction data (unused for frozen)

    Returns:
        S_chem: Chemical source terms [n_cells, n_variables] (all zeros)
    """
    # Frozen chemistry: no reactions (ω̇_i = 0 for all species)
    return jnp.zeros_like(U)
