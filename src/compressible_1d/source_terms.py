"""Source terms module for multi-species two-temperature equations.

Implements vibrational relaxation and frozen chemistry stubs.
"""

import jax.numpy as jnp
from jaxtyping import Float, Array

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
    Q_v = compute_vibrational_relaxation(U, equation_manager)
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
    e_v_eq_species = thermodynamic_relations.compute_e_ve(T, equation_manager.species)  # [n_species, n_cells]
    e_v_eq = jnp.sum(Y_s * e_v_eq_species.T, axis=1)  # [n_cells]

    # Compute actual vibrational energy at T_v
    e_v_species = thermodynamic_relations.compute_e_ve(T_v, equation_manager.species)  # [n_species, n_cells]
    e_v_actual = jnp.sum(Y_s * e_v_species.T, axis=1)  # [n_cells]

    tau_v = compute_relaxation_time(Y_s, rho, T, T_v, p, equation_manager)

    Q_dot_v = rho * (e_v_eq - e_v_actual) / (tau_v + 1e-14)

    return Q_dot_v


def compute_relaxation_time(
    Y_s: Float[Array, "n_cells n_species"],
    rho: Float[Array, "n_cells"],
    T: Float[Array, "n_cells"],
    T_v: Float[Array, "n_cells"],
    p: Float[Array, "n_cells"],
    equation_manager: equation_manager_types.EquationManager,
) -> Float[Array, "n_cells"]:
    """Compute Millikan-White relaxation time with Park correction.

    For now, use simplified constant relaxation time.
    Later: implement full Millikan-White model from NASA TP-2867.

    τ_v = τ_MW * (p_ref / p)

    where τ_MW is the Millikan-White relaxation time at reference pressure.

    Args:
        Y_s: Mass fractions [n_cells, n_species]
        rho: Density [n_cells]
        T: Translational temperature [n_cells]
        T_v: Vibrational temperature [n_cells]
        p: Pressure [n_cells]
        equation_manager: Contains species data

    Returns:
        tau_v: Relaxation time [n_cells] in seconds
    """
    # Simplified: constant relaxation time (1 μs)
    # This is sufficient for testing the solver architecture
    # TODO: Implement Millikan-White from NASA TP-2867:
    #   τ_MW = (1/p) * exp[A(T^(-1/3) - 0.015μ^(1/4)) - 18.42]
    #   where A = characteristic temperature, μ = reduced mass
    #   Park correction adds limiting τ at high temperatures
    
    tau_v = jnp.full_like(T, 1e-6)  # 1 microsecond

    return tau_v


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
