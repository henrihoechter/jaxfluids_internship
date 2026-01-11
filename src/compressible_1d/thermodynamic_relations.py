"""Thermodynamic property calculations for species.

This module contains pure functions for computing thermodynamic properties.
High-level functions (compute_equilibrium_enthalpy, compute_cp, etc.) take
a SpeciesTable as argument, while low-level functions take raw arrays.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from compressible_1d import constants

if TYPE_CHECKING:
    from compressible_1d.chemistry_types import SpeciesTable


def compute_cp_from_polynomial(
    T: Float[Array, " N"],
    T_limit_low: Float[Array, "n_species n_ranges"],
    T_limit_high: Float[Array, "n_species n_ranges"],
    parameters: Float[Array, "n_species n_ranges n_parameters"],
    molar_masses: Float[Array, " n_species"],
) -> Float[Array, "n_species N"]:
    """Compute specific heat at constant pressure using polynomial curve fits.

    Implements Gnoffo eq. 31 via differentiation of enthalpy polynomial:
    Since h(T) = (R/M) * (a0*T + a1*T^2/2 + ... + a5)
    Then C_p = dh/dT = (R/M) * (a0 + a1*T + a2*T^2 + a3*T^3 + a4*T^4)

    Args:
        T: Temperature array [K], shape (N,)
        T_limit_low: Lower temperature bounds [K], shape (n_species, n_ranges)
        T_limit_high: Upper temperature bounds [K], shape (n_species, n_ranges)
        parameters: Enthalpy polynomial coefficients [a0, ..., a5], shape (n_species, n_ranges, 6)
        molar_masses: Molar mass [kg/mol], shape (n_species,)

    Returns:
        C_p [J/(kg·K)], shape (n_species, N)

    Notes:
        - Uses SAME polynomial coefficients as enthalpy (just differentiated)
        - Same temperature range masking as h_equilibrium
        - Consistent with existing enthalpy calculation structure
    """
    n_ranges = T_limit_low.shape[1]

    def cp_single_species(
        T_low: Float[Array, " n_ranges"],
        T_high: Float[Array, " n_ranges"],
        params: Float[Array, "n_ranges n_parameters"],
        M: float,
    ) -> Float[Array, " N"]:
        """Compute C_p for one species across all temperatures."""
        cp = jnp.zeros_like(T)

        for i in range(n_ranges):
            mask = (T >= T_low[i]) & (T < T_high[i])

            # Extract enthalpy coefficients [a0, a1, a2, a3, a4, a5]
            a = params[i, :]

            # C_p = dh/dT = (R/M) * (a0 + a1*T + a2*T^2 + a3*T^3 + a4*T^4)
            cp_range = (
                constants.R_universal
                / M
                * (a[0] + a[1] * T + a[2] * T**2 + a[3] * T**3 + a[4] * T**4)
            )

            cp = jnp.where(mask, cp_range, cp)

        return cp

    # Vectorize over species dimension (same pattern as h_equilibrium)
    cp_vectorized = jax.vmap(cp_single_species, in_axes=(0, 0, 0, 0))

    return cp_vectorized(T_limit_low, T_limit_high, parameters, molar_masses)


def compute_cv_trans_rot(
    T: Float[Array, " N"],
    is_monoatomic: Float[Array, " n_species"],
    molar_masses: Float[Array, " n_species"],
) -> Float[Array, "n_species N"]:
    """Compute translational-rotational specific heat for all species.

    Uses ideal gas model (temperature-independent):
    - Atoms (theta_rot = 0): C_v,tr = (3/2) R/M
    - Diatomic molecules: C_v,tr = (5/2) R/M

    Args:
        T: Temperature array [K], shape (N,) - not used but kept for API consistency
        is_monoatomic: Boolean mask indicating monoatomic species, shape (n_species,)
        molar_masses: Molar mass [kg/mol], shape (n_species,)

    Returns:
        C_v,tr [J/(kg·K)], shape (n_species, N)

    Notes:
        - Gnoffo eq. 29 uses C_v,t + C_v,r as constants
        - C_v,t = (3/2) R/M for all species
        - C_v,r = R/M for molecules, 0 for atoms
        - Combined: C_v,tr = (5/2) R/M (molecules), (3/2) R/M (atoms)
    """
    R = constants.R_universal  # J/(mol·K)
    M = molar_masses  # kg/mol

    # Compute C_v,tr per species (Gnoffo 1989, eq. 24, 25)
    cv_tr_species = jnp.where(
        ~is_monoatomic,
        2.5 * R / M,  # Diatomic: (5/2) R/M
        1.5 * R / M,  # Atom: (3/2) R/M
    )  # Shape: (n_species,)

    # Broadcast to (n_species, N) - constant across temperature
    N = T.shape[0]
    cv_tr = jnp.broadcast_to(cv_tr_species[:, None], (len(is_monoatomic), N))

    return cv_tr


def compute_cv_vib_electronic(
    T_V: Float[Array, " N"],
    T_limit_low: Float[Array, "n_species n_ranges"],
    T_limit_high: Float[Array, "n_species n_ranges"],
    parameters: Float[Array, "n_species n_ranges n_parameters"],
    is_monoatomic: Float[Array, " n_species"],
    molar_masses: Float[Array, " n_species"],
) -> Float[Array, "n_species N"]:
    """Compute vibrational-electronic specific heat.

    Implements Gnoffo equation chain:
    - eq. 31: C_p^s(T_V) via dh/dT using enthalpy polynomial
    - eq. 30: C_v^s(T_V) = C_p^s(T_V) - R/M_s
    - eq. 29: C_{v,V}^s(T_V) = C_v^s(T_V) - C_{v,t}^s - C_{v,r}^s

    Args:
        T_V: Vibrational temperature array [K], shape (N,)
        T_limit_low: Temperature range lower bounds [K], shape (n_species, n_ranges)
        T_limit_high: Temperature range upper bounds [K], shape (n_species, n_ranges)
        parameters: Enthalpy polynomial coefficients, shape (n_species, n_ranges, 6)
        is_monoatomic: Boolean mask indicating monoatomic species, shape (n_species,)
        molar_masses: Molar mass [kg/mol], shape (n_species,)

    Returns:
        C_{v,V} [J/(kg·K)], shape (n_species, N)

    Notes:
        - Evaluated at vibrational temperature T_V (not translational T)
        - Uses enthalpy polynomial derivatives (same as h_equilibrium)
        - Subtracts constant translational-rotational contributions
    """
    cp = compute_cp_from_polynomial(
        T_V, T_limit_low, T_limit_high, parameters, molar_masses
    )

    # Compute C_v = C_p - R/M (eq. 30)
    R = constants.R_universal
    M = molar_masses  # kg/mol
    R_over_M = R / M  # Shape: (n_species,)

    cv = cp - R_over_M[:, None]  # Broadcast to (n_species, N)

    # Compute C_v,tr
    T_dummy = jnp.ones_like(T_V)
    cv_tr = compute_cv_trans_rot(T_dummy, is_monoatomic, molar_masses)

    # C_v,V = C_v - C_v,tr (eq. 29)
    cv_vib = cv - cv_tr

    return cv_vib


def compute_e_vib_electronic(
    T_V: Float[Array, " N"],
    T_ref: float,
    T_limit_low: Float[Array, "n_species n_ranges"],
    T_limit_high: Float[Array, "n_species n_ranges"],
    parameters: Float[Array, "n_species n_ranges n_parameters"],
    is_monoatomic: Float[Array, " n_species"],
    molar_masses: Float[Array, " n_species"],
) -> Float[Array, "n_species N"]:
    """Compute vibrational-electronic internal energy by integrating C_{v,V}.

    Implements Gnoffo eq. 98:
    e_{v,s}(T_V) = ∫_{T_ref}^{T_V} C_{v,V}^s(T') dT'

    Uses analytical polynomial integration of C_{v,V} derived from enthalpy polynomials.

    Args:
        T_V: Vibrational temperature array [K], shape (N,)
        T_ref: Reference temperature [K] - passed as parameter, partialed at creation
        T_limit_low: Temperature range bounds [K], shape (n_species, n_ranges)
        T_limit_high: Temperature range bounds [K], shape (n_species, n_ranges)
        parameters: Enthalpy polynomial coefficients, shape (n_species, n_ranges, 6)
        theta_rot: Rotational characteristic temperature [K], shape (n_species,)
        molar_masses: Molar mass [kg/mol], shape (n_species,)

    Returns:
        e_v [J/kg], shape (n_species, N)

    Notes:
        - Integration performed analytically, not numerically
        - C_{v,V} derived from enthalpy polynomial (eqs. 29-31)
        - T_ref is pre-evaluated via functools.partial (not stored in SpeciesTable)
        - Same polynomial structure as h_equilibrium for consistency
    """
    n_ranges = T_limit_low.shape[1]

    def _e_v_single_species(
        T_low: Float[Array, " n_ranges"],
        T_high: Float[Array, " n_ranges"],
        params: Float[Array, "n_ranges n_parameters"],
        is_monoatomic_s: float,
        M_s: float,
    ) -> Float[Array, " N"]:
        """Integrate C_{v,V} for one species."""

        R = constants.R_universal
        M = M_s  # kg/mol

        # Compute C_v,tr constant for this species
        cv_tr_s = jnp.where(~is_monoatomic_s, 2.5 * R / M, 1.5 * R / M)

        e_v = jnp.zeros_like(T_V)

        for i in range(n_ranges):
            mask = (T_V >= T_low[i]) & (T_V < T_high[i])

            def _cv_V_integrated(T, idx_temperature_range: int):
                """Integrate: ∫C_{v,V} dT = b_0*T + b_1*T^2/2 + b_2*T^3/3
                + b_3*T^4/4 + b_4*T^5/5
                """
                a = params[idx_temperature_range, :]

                R_over_M = R / M
                b_0 = R_over_M * (a[0] - 1) - cv_tr_s  # Linear term
                b_1 = R_over_M * a[1]  # Quadratic term
                b_2 = R_over_M * a[2]  # Cubic term
                b_3 = R_over_M * a[3]  # Quartic term
                b_4 = R_over_M * a[4]  # Quintic term
                b_5 = R_over_M * a[5]  # Constant term (drops out in definite integral)

                return (
                    b_0 * T
                    + b_1 * T**2 / 2
                    + b_2 * T**3 / 3
                    + b_3 * T**4 / 4
                    + b_4 * T**5 / 5
                    + b_5
                )

            # e_v = ∫_{T_ref}^{T_V} C_{v,V} dT' = F(T_V) - F(T_ref)
            # TODO: this is sloppy: T_ref=298K is out of the validity bound of 0th range
            # but i still use it without checks as lower limit of 0th range is 300K
            e_v_range = _cv_V_integrated(T_V, i) - _cv_V_integrated(T_ref, 0)

            e_v = jnp.where(mask, e_v_range, e_v)

        return e_v

    # Vectorize over species
    e_v_vectorized = jax.vmap(_e_v_single_species, in_axes=(0, 0, 0, 0, 0))

    return e_v_vectorized(
        T_limit_low, T_limit_high, parameters, is_monoatomic, molar_masses
    )


# def compute_e_vibrational(
#     T_V: Float[Array, " N"],
#     species_table: SpeciesTable,
# ) -> Float[Array, "n_species N"]:
#     """Compute vibrational-electronic internal energy by integrating C_{v,V}.

#     Implements Gnoffo eq. 33:
#     e_{v,s}(T_V) = ∫_{T_ref}^{T_V} C_{v,V}^s(T') dT'

#     Uses analytical polynomial integration of C_{v,V} derived from enthalpy polynomials.

#     Args:
#         T_V: Vibrational temperature array [K], shape (N,)
#         T_ref: Reference temperature [K] - passed as parameter, partialed at creation
#         T_limit_low: Temperature range bounds [K], shape (n_species, n_ranges)
#         T_limit_high: Temperature range bounds [K], shape (n_species, n_ranges)
#         parameters: Enthalpy polynomial coefficients, shape (n_species, n_ranges, 6)
#         theta_rot: Rotational characteristic temperature [K], shape (n_species,)
#         molar_masses: Molar mass [kg/mol], shape (n_species,)

#     Returns:
#         e_v [J/kg], shape (n_species, N)

#     Notes:
#         - Integration performed analytically, not numerically
#         - C_{v,V} derived from enthalpy polynomial (eqs. 29-31)
#         - T_ref is passed as a parameter (not stored in SpeciesTable)
#         - Same polynomial structure as h_equilibrium for consistency
#     """
#     T_ref = species_table.T_ref
#     T_limit_high = species_table.T_limit_high
#     T_limit_low = species_table.T_limit_low
#     parameters = species_table.enthalpy_coeffs
#     molar_masses = species_table.molar_masses
#     n_ranges = T_limit_low.shape[1]

#     def e_v_single_species(
#         T_low: Float[Array, " n_ranges"],
#         T_high: Float[Array, " n_ranges"],
#         params: Float[Array, "n_ranges n_parameters"],
#         M_s: float,
#         T_ref: float,
#     ) -> Float[Array, " N"]:
#         """Integrate C_{v,V} for one species."""

#         R = constants.R_universal
#         M = M_s  # kg/mol

#         e_v = jnp.zeros_like(T_V)

#         e_v_t_ref = (
#             R
#             / M
#             * (
#                 (params[0, 0] - 1 - 2.5) * T_ref
#                 + params[0, 1] * T_ref**2 / 2
#                 + params[0, 2] * T_ref**3 / 3
#                 + params[0, 3] * T_ref**4 / 4
#                 + params[0, 4] * T_ref**5 / 5
#                 + params[0, 5]
#             )
#         )

#         for i in range(n_ranges):
#             mask = (T_V >= T_low[i]) & (T_V < T_high[i])

#             a = params[i, :]

#             def e_v_function(T):
#                 return (
#                     R
#                     / M
#                     * (
#                         (a[0] - 1 - 2.5) * T
#                         + a[1] * T**2 / 2
#                         + a[2] * T**3 / 3
#                         + a[3] * T**4 / 4
#                         + a[4] * T**5 / 5
#                         + a[5]
#                     )
#                 )

#             # e_v_range = e_v_function(T_V) - e_v_function(T_ref)
#             e_v_range = e_v_function(T_V) - e_v_t_ref

#             e_v = jnp.where(mask, e_v_range, e_v)

#         return e_v

#     # Vectorize over species
#     e_v_vectorized = jax.vmap(e_v_single_species, in_axes=(0, 0, 0, 0, None))

#     return e_v_vectorized(T_limit_low, T_limit_high, parameters, molar_masses, T_ref)


def compute_e_vibrational_from_harmonic_oscillator(
    T_V: Float[Array, " N"],
    characteristic_temperature: Float[Array, " n_species"],
    M: Float[Array, " n_species"],
) -> Float[Array, "n_species N"]:
    # TODO: this interface is not useful. ideally the characteristic_temperature was
    # obtained from the species_table
    R = constants.R_universal

    return (
        R
        / M
        * characteristic_temperature
        / (jnp.exp(characteristic_temperature / T_V) - 1.0)
    )


def compute_mixture_cv_trans_rot(
    c_s: Float[Array, "n_species ..."],
    cv_tr: Float[Array, "n_species ..."],
) -> Float[Array, "..."]:
    """Compute mixture translational-rotational specific heat.

    Implements Gnoffo eq. 39:
    C_{v,tr} = Σ_s c_s C_{v,tr}^s

    Args:
        c_s: Mass fractions, shape (n_species, ...)
        cv_tr: Species C_v,tr [J/(kg·K)], shape (n_species, ...)

    Returns:
        C_{v,tr} mixture [J/(kg·K)], shape (...)

    Notes:
        - Simple mass-weighted average
        - Used in computing translational temperature T
    """
    return jnp.sum(c_s * cv_tr, axis=0)


def compute_reference_internal_energy(
    h_s0: Float[Array, " n_species"],
    molar_masses: Float[Array, " n_species"],
    T_ref: float,
) -> Float[Array, " n_species"]:
    """Compute reference internal energy e_{s,0} from formation enthalpy.

    Converts formation enthalpy at reference state to internal energy:
    e_{s,0} = h_{s,0} - R*T_ref/M_s

    Args:
        h_s0: Formation enthalpy at reference [J/kg], shape (n_species,)
        molar_masses: Molar mass [kg/mol], shape (n_species,)
        T_ref: Reference temperature [K]

    Returns:
        e_{s,0} [J/kg], shape (n_species,)

    Notes:
        - Used in Gnoffo eq. 97 and implied by eq. 102
        - Relationship: h = e + pV/m = e + RT/M for ideal gas
        - T_ref = 298.16 K in current implementation
    """
    R = constants.R_universal
    M = molar_masses  # kg/mol

    e_s0 = h_s0 - R * T_ref / M

    return e_s0


def solve_vibrational_temperature_from_vibroelectric_energy(
    e_V_target: Float[Array, " N"],
    c_s: Float[Array, "n_species N"],
    T_V_initial: Float[Array, " N"],
    species_table: "SpeciesTable",
    max_iterations: int = 20,
    rtol: float = 1e-6,
    atol: float = 1.0,  # [K]
) -> Float[Array, " N"]:
    """Solve for vibrational temperature T_V from vibrational energy e_V.

    Optimized implementation using batched operations and analytical derivatives.

    Implements Gnoffo step 9: Find T_V such that
    sum_s c_s e_{v,s}(T_V) = e_V

    Uses Newton-Raphson with analytical Jacobian from C_{v,V}:
    T_V^{n+1} = T_V^n - f(T_V^n) / f'(T_V^n)

    where:
        f(T_V) = sum_s c_s e_{v,s}(T_V) - e_V
        f'(T_V) = sum_s c_s C_{v,V}^s(T_V)  (analytical derivative)

    Args:
        e_V_target: Target vibrational energy [J/kg], shape (N,)
        c_s: Mass fractions, shape (n_species, N)
        T_V_initial: Initial guess for T_V [K], shape (N,)
        species_table: SpeciesTable containing thermodynamic data
        max_iterations: Maximum Newton iterations (default 20)
        rtol: Relative tolerance for convergence
        atol: Absolute tolerance [K] for temperature convergence

    Returns:
        T_V [K], shape (N,)

    Notes:
        - Uses analytical derivative C_{v,V} instead of autodiff
        - Fully batched across all cells (no per-cell vmap)
        - Uses jax.lax.while_loop for early exit on convergence
        - Converges when |delta_T_V| < atol or |delta_T_V/T_V| < rtol for all cells
    """
    # Extract coefficient data from species_table
    T_ref = species_table.T_ref
    T_limit_low = species_table.T_limit_low
    T_limit_high = species_table.T_limit_high
    enthalpy_coeffs = species_table.enthalpy_coeffs
    is_monoatomic = species_table.is_monoatomic
    molar_masses = species_table.molar_masses

    def _compute_residual_and_jacobian(
        T_V: Float[Array, " N"],
    ) -> tuple[Float[Array, " N"], Float[Array, " N"]]:
        """Compute residual f(T_V) and Jacobian df/dT_V for all cells.

        Returns:
            residual: f(T_V) = sum_s c_s e_{v,s}(T_V) - e_V_target, shape (N,)
            jacobian: df/dT_V = sum_s c_s C_{v,V}^s(T_V), shape (N,)
        """
        # Compute e_v for all species at T_V: shape (n_species, N)
        e_v_species = compute_e_vib_electronic(
            T_V,
            T_ref,
            T_limit_low,
            T_limit_high,
            enthalpy_coeffs,
            is_monoatomic,
            molar_masses,
        )

        # Compute C_{v,V} for all species at T_V: shape (n_species, N)
        cv_v_species = compute_cv_vib_electronic(
            T_V,
            T_limit_low,
            T_limit_high,
            enthalpy_coeffs,
            is_monoatomic,
            molar_masses,
        )

        # Mixture vibrational energy: sum_s c_s e_{v,s}(T_V)
        e_V_computed = jnp.sum(c_s * e_v_species, axis=0)  # shape (N,)

        # Residual: f(T_V) = e_V_computed - e_V_target
        residual = e_V_computed - e_V_target

        # Jacobian: df/dT_V = sum_s c_s C_{v,V}^s(T_V)
        jacobian = jnp.sum(c_s * cv_v_species, axis=0)  # shape (N,)

        return residual, jacobian

    def _newton_step(
        carry: tuple[Float[Array, " N"], Float[Array, " N"], int],
    ) -> tuple[Float[Array, " N"], Float[Array, " N"], int]:
        """Perform one Newton-Raphson iteration."""
        T_V, delta_T_V_prev, iteration = carry

        # Compute residual and Jacobian
        residual, jacobian = _compute_residual_and_jacobian(T_V)

        # Newton update: delta_T_V = -f / (df/dT_V)
        # Clip jacobian to avoid division by zero (jacobian should be positive)
        delta_T_V_full = -residual / jnp.clip(jacobian, 1e-20, None)

        # Apply damping to prevent overshooting (limit step to 0.5 * T_V)
        max_step = 0.5 * T_V
        delta_T_V = jnp.clip(delta_T_V_full, -max_step, max_step)

        # Update temperature (ensure positive, minimum 50K)
        T_V_new = jnp.maximum(T_V + delta_T_V, 50.0)

        return T_V_new, delta_T_V, iteration + 1

    def _continue_condition(
        carry: tuple[Float[Array, " N"], Float[Array, " N"], int],
    ) -> bool:
        """Check if iteration should continue."""
        T_V, delta_T_V, iteration = carry

        # Check convergence for all cells
        abs_error = jnp.abs(delta_T_V)
        rel_error = abs_error / jnp.clip(jnp.abs(T_V), 1e-10, None)
        converged = (abs_error < atol) | (rel_error < rtol)
        all_converged = jnp.all(converged)

        # Continue if not all converged AND under max iterations
        return (~all_converged) & (iteration < max_iterations)

    # Initialize: T_V, delta_T_V (set to inf to ensure first iteration runs), iteration
    initial_carry = (T_V_initial, jnp.full_like(T_V_initial, jnp.inf), 0)

    # Run Newton-Raphson with early exit
    T_V_final, _, _ = jax.lax.while_loop(
        _continue_condition, _newton_step, initial_carry
    )

    return T_V_final


def solve_T_from_internal_energy(
    e: Float[Array, "..."],
    e_V: Float[Array, "..."],
    c_s: Float[Array, "n_species ..."],
    cv_tr: Float[Array, "n_species ..."],
    e_s0: Float[Array, " n_species"],
    T_ref: float,
) -> Float[Array, "..."]:
    """Solve for translational temperature T from internal energy.

    Implements Gnoffo step 13:
    T = T_ref + (e - e_V - e_0) / C_{v,tr}

    where:
    - e_0 = Σ_s c_s e_{s,0} (mixture reference energy)
    - C_{v,tr} = Σ_s c_s C_{v,tr}^s (mixture trans-rot specific heat)

    Args:
        e: Total mixture internal energy [J/kg], shape (...)
        e_V: Vibrational-electronic energy [J/kg], shape (...)
        c_s: Mass fractions, shape (n_species, ...)
        cv_tr: Trans-rot specific heat per species [J/(kg·K)], shape (n_species, ...)
        e_s0: Reference internal energy per species [J/kg], shape (n_species,)
        T_ref: Reference temperature [K]

    Returns:
        T [K], shape (...)

    Notes:
        - Direct algebraic solution (no iteration required)
        - Assumes C_{v,tr} is temperature-independent (ideal gas)
        - From eqs. 97, 102, 103 in Gnoffo
    """
    # Compute mixture reference energy: e_0 = Σ c_s e_{s,0}
    e_0 = jnp.sum(c_s * e_s0[:, None], axis=0)  # Shape: (...)

    # Compute mixture C_v,tr: C_{v,tr} = Σ c_s C_{v,tr}^s
    cv_tr_mix = compute_mixture_cv_trans_rot(c_s, cv_tr)  # Shape: (...)

    # Solve for T: T = T_ref + (e - e_V - e_0) / C_{v,tr}
    T = T_ref + (e - e_V - e_0) / jnp.clip(cv_tr_mix, 1e-10, None)

    return T


def compute_equilibrium_enthalpy(
    T: Float[Array, " N"],
    species_table: "SpeciesTable",
) -> Float[Array, "n_species N"]:
    """Compute equilibrium enthalpy for all species.

    Args:
        T: Temperature array [K], shape (N,)
        species_table: SpeciesTable containing coefficient data

    Returns:
        Specific enthalpy [J/kg] for all species, shape (n_species, N)
    """
    # return compute_equilibrium_enthalpy_polynomial(
    #     T,
    #     species_table.T_limit_low,
    #     species_table.T_limit_high,
    #     species_table.enthalpy_coeffs,
    #     species_table.molar_masses,
    # )
    raise NotImplementedError


def compute_cp(
    T: Float[Array, " N"],
    species_table: "SpeciesTable",
) -> Float[Array, "n_species N"]:
    """Compute specific heat at constant pressure for all species.

    Args:
        T: Temperature array [K], shape (N,)
        species_table: SpeciesTable containing coefficient data

    Returns:
        C_p [J/(kg·K)] for all species, shape (n_species, N)
    """
    return compute_cp_from_polynomial(
        T,
        species_table.T_limit_low,
        species_table.T_limit_high,
        species_table.enthalpy_coeffs,
        species_table.molar_masses,
    )


def compute_cv_tr(
    T: Float[Array, " N"],
    species_table: "SpeciesTable",
) -> Float[Array, "n_species N"]:
    """Compute translational-rotational specific heat for all species.

    Args:
        T: Temperature array [K], shape (N,) - not used but kept for API consistency
        species_table: SpeciesTable containing is_monoatomic and molar_masses

    Returns:
        C_v,tr [J/(kg·K)] for all species, shape (n_species, N)
    """
    return compute_cv_trans_rot(
        T,
        species_table.is_monoatomic,
        species_table.molar_masses,
    )


def compute_cv_ve(
    T_V: Float[Array, " N"],
    species_table: "SpeciesTable",
) -> Float[Array, "n_species N"]:
    """Compute vibrational-electronic specific heat for all species.

    Args:
        T_V: Vibrational temperature array [K], shape (N,)
        species_table: SpeciesTable containing coefficient data

    Returns:
        C_v,V [J/(kg·K)] for all species, shape (n_species, N)
    """
    return compute_cv_vib_electronic(
        T_V,
        species_table.T_limit_low,
        species_table.T_limit_high,
        species_table.enthalpy_coeffs,
        species_table.is_monoatomic,
        species_table.molar_masses,
    )


def compute_e_ve(
    T_V: Float[Array, " N"],
    species_table: "SpeciesTable",
) -> Float[Array, "n_species N"]:
    """Compute vibrational-electronic internal energy for all species.

    Args:
        T_V: Vibrational temperature array [K], shape (N,)
        species_table: SpeciesTable containing coefficient data and T_ref

    Returns:
        e_v [J/kg] for all species, shape (n_species, N)
    """
    return compute_e_vib_electronic(
        T_V,
        species_table.T_ref,
        species_table.T_limit_low,
        species_table.T_limit_high,
        species_table.enthalpy_coeffs,
        species_table.is_monoatomic,
        species_table.molar_masses,
    )


def compute_electronic_energy_from_levels(
    T_el: Float[Array, " N"],
    g_i: Float[Array, " n_levels"],
    theta_el_i: Float[Array, " n_levels"],
    R_s: float | jnp.ndarray,
) -> Float[Array, " N"]:
    """
    Compute electronic internal energy e_el,s(T_el) [J/kg] from discrete electronic levels.

    Implements:
        e_el,s = R_s * (sum_{i!=0} g_i * theta_i * exp(-theta_i/T_el)) /
                       (sum_i     g_i         exp(-theta_i/T_el))

    Args:
        T_el: Electronic temperature [K]. Can be scalar or array (...,).
        g_i: Degeneracy array for levels i, including ground state i=0. Shape (n_levels,).
        theta_el_i: Characteristic electronic temperatures [K] for each level,
                    including ground state theta=0. Shape (n_levels,).
        R_s: Species gas constant [J/(kg*K)] (Ru / M_s). Can be float or array broadcastable to T_el.

    Returns:
        e_el: Electronic internal energy [J/kg], same shape as T_el.
    """
    T_el = jnp.asarray(T_el)

    # Broadcast to shape (n_levels, ...) for vectorized evaluation
    theta = theta_el_i[:, None] if T_el.ndim > 0 else theta_el_i
    g = g_i[:, None] if T_el.ndim > 0 else g_i

    # Avoid division by zero at T=0
    T_safe = jnp.maximum(T_el, 1e-12)

    # Boltzmann weights: w_i = g_i * exp(-theta_i/T)
    w = g * jnp.exp(-theta / T_safe)

    # Partition function (denominator): sum_i w_i
    Z = jnp.sum(w, axis=0)

    # Numerator excludes ground state i=0
    numerator = jnp.sum(w[1:] * theta[1:], axis=0)

    # e_el = R_s * numerator / Z
    return R_s * numerator / jnp.clip(Z, 1e-300, None)
