"""Thermodynamic property calculations for species.

This module contains pure functions for computing thermodynamic properties.
High-level functions (compute_equilibrium_enthalpy, compute_cp, etc.) take
a SpeciesTable as argument, while low-level functions take raw arrays.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from . import constants
from .chemistry_types import SpeciesTable


def compute_cp_from_polynomial(
    T: Float[Array, " N"],
    T_limit_low: Float[Array, "n_species n_ranges"],
    T_limit_high: Float[Array, "n_species n_ranges"],
    parameters: Float[Array, "n_species n_ranges n_parameters"],
    molar_masses: Float[Array, " n_species"],
) -> Float[Array, "n_species N"]:
    """Compute specific heat at constant pressure using polynomial curve fits.

    Implements Gnoffo eq. 31 via differentiation of enthalpy polynomial:
    Since h(T) = (R/M) * (a0*T + a1*T^2/2 + ... + a5),
    C_p = dh/dT = (R/M) * (a0 + a1*T + a2*T^2 + a3*T^3 + a4*T^4).

    Args:
        T: Temperature array [K].
        T_limit_low: Lower temperature bounds [K].
        T_limit_high: Upper temperature bounds [K].
        parameters: Enthalpy polynomial coefficients [a0, ..., a5].
        molar_masses: Molar mass [kg/mol].

    Returns:
        C_p [J/(kg*K)] for all species.
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

    Uses Gnoffo eq. 29 (temperature-independent ideal gas model):
    C_v,tr = (5/2) R/M for molecules, (3/2) R/M for atoms.

    Args:
        T: Temperature array [K] (not used, kept for API consistency).
        is_monoatomic: Boolean mask indicating monoatomic species.
        molar_masses: Molar mass [kg/mol].

    Returns:
        C_v,tr [J/(kg*K)] for all species, broadcast to (n_species, N).
    """
    R = constants.R_universal  # J/(mol*K)
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


def compute_cv_t(
    molar_masses: Float[Array, " n_species"],
) -> Float[Array, " n_species"]:
    """Compute translational Cv for each species."""
    R_s = constants.R_universal / molar_masses
    return 1.5 * R_s


def compute_cv_r(
    molar_masses: Float[Array, " n_species"],
    is_monoatomic: Float[Array, " n_species"],
) -> Float[Array, " n_species"]:
    """Compute rotational Cv for each species."""
    R_s = constants.R_universal / molar_masses
    return jnp.where(is_monoatomic, 0.0, 1.0 * R_s)


def compute_cv_vib_electronic(
    T_V: Float[Array, " N"],
    T_limit_low: Float[Array, "n_species n_ranges"],
    T_limit_high: Float[Array, "n_species n_ranges"],
    parameters: Float[Array, "n_species n_ranges n_parameters"],
    is_monoatomic: Float[Array, " n_species"],
    molar_masses: Float[Array, " n_species"],
) -> Float[Array, "n_species N"]:
    """Compute vibrational-electronic specific heat.

    Implements Gnoffo eqs. 29-31:
    C_v,V = C_p(T_V) - R/M - C_v,tr, where C_p is evaluated at T_V.

    Args:
        T_V: Vibrational temperature [K].
        T_limit_low: Temperature range lower bounds [K].
        T_limit_high: Temperature range upper bounds [K].
        parameters: Enthalpy polynomial coefficients.
        is_monoatomic: Boolean mask indicating monoatomic species.
        molar_masses: Molar mass [kg/mol].

    Returns:
        C_v,V [J/(kg*K)] for all species.
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

    Implements Gnoffo eq. 98 analytically:
    e_{v,s}(T_V) = integral from T_ref to T_V of C_{v,V}^s(T') dT'.

    Args:
        T_V: Vibrational temperature [K].
        T_ref: Reference temperature [K] (pre-bound via functools.partial).
        T_limit_low: Temperature range lower bounds [K].
        T_limit_high: Temperature range upper bounds [K].
        parameters: Enthalpy polynomial coefficients.
        is_monoatomic: Boolean mask indicating monoatomic species.
        molar_masses: Molar mass [kg/mol].

    Returns:
        e_v [J/kg] for all species.
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
                """Integrate: integral C_{v,V} dT = b_0*T + b_1*T^2/2 + b_2*T^3/3
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

            # e_v = integral _{T_ref}^{T_V} C_{v,V} dT' = F(T_V) - F(T_ref)
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


def compute_e_vibrational_from_harmonic_oscillator(
    T_V: Float[Array, " N"],
    characteristic_temperature: Float[Array, " n_species"],
    M: Float[Array, " n_species"],
) -> Float[Array, "n_species N"]:
    """Compute vibrational energy for all species using a harmonic oscillator model."""
    # TODO(hhoechter): ideally characteristic_temperature should come from SpeciesTable
    T = jnp.atleast_1d(T_V)
    T_safe = jnp.maximum(T, 1e-12)
    theta = characteristic_temperature[:, None]
    R_over_M = constants.R_universal / M

    x = theta / T_safe[None, :]
    denom = jnp.expm1(x)
    denom_safe = jnp.where(denom == 0.0, 1.0, denom)

    e_vib = R_over_M[:, None] * theta / denom_safe
    return jnp.where(theta == 0.0, 0.0, e_vib)


def compute_cv_vibrational_from_harmonic_oscillator(
    T_V: Float[Array, " N"],
    characteristic_temperature: Float[Array, " n_species"],
    M: Float[Array, " n_species"],
) -> Float[Array, "n_species N"]:
    """Compute vibrational specific heat for a harmonic oscillator model."""
    T = jnp.atleast_1d(T_V)
    T_safe = jnp.maximum(T, 1e-12)
    theta = characteristic_temperature[:, None]
    R_over_M = constants.R_universal / M

    x = theta / T_safe[None, :]
    exp_x = jnp.exp(x)
    denom = jnp.expm1(x)
    denom_safe = jnp.where(denom == 0.0, 1.0, denom)

    cv_vib = R_over_M[:, None] * x**2 * exp_x / (denom_safe**2)
    return jnp.where(theta == 0.0, 0.0, cv_vib)


def compute_mixture_cv_trans_rot(
    c_s: Float[Array, "n_species ..."],
    cv_tr: Float[Array, "n_species ..."],
) -> Float[Array, "..."]:
    """Compute mixture translational-rotational specific heat.

    Implements Gnoffo eq. 39: C_{v,tr} = sum_s c_s C_{v,tr}^s.

    Args:
        c_s: Mass fractions.
        cv_tr: Species C_v,tr [J/(kg*K)].

    Returns:
        Mixture C_v,tr [J/(kg*K)].
    """
    return jnp.sum(c_s * cv_tr, axis=0)


def compute_reference_internal_energy(
    h_s0: Float[Array, " n_species"],
    molar_masses: Float[Array, " n_species"],
    T_ref: float,
) -> Float[Array, " n_species"]:
    """Compute reference internal energy e_{s,0} from formation enthalpy.

    Converts formation enthalpy to internal energy via the ideal gas relation:
    e_{s,0} = h_{s,0} - R*T_ref/M_s (Gnoffo eqs. 97, 102).

    Args:
        h_s0: Formation enthalpy at reference [J/kg].
        molar_masses: Molar mass [kg/mol].
        T_ref: Reference temperature [K].

    Returns:
        e_{s,0} [J/kg] for all species.
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

    Implements Gnoffo step 9 via batched Newton-Raphson with analytical Jacobian:
    find T_V such that sum_s c_s e_{v,s}(T_V) = e_V.

    Args:
        e_V_target: Target vibrational energy [J/kg].
        c_s: Mass fractions.
        T_V_initial: Initial guess for T_V [K].
        species_table: Species thermodynamic data.
        max_iterations: Maximum Newton iterations.
        rtol: Relative tolerance for convergence.
        atol: Absolute tolerance [K] for temperature convergence.

    Returns:
        T_V [K].
    """

    def _compute_residual_and_jacobian(
        T_V: Float[Array, " N"],
    ) -> tuple[Float[Array, " N"], Float[Array, " N"]]:
        """Compute residual f(T_V) and Jacobian df/dT_V for all cells.

        Returns:
            Tuple of (residual, jacobian): f(T_V) = sum_s c_s e_{v,s}(T_V) - e_V_target
            and df/dT_V = sum_s c_s C_{v,V}^s(T_V).
        """
        # Compute e_v and C_{v,V} for all species at T_V: shape (n_species, N)
        e_v_species = compute_e_ve(T_V, species_table)
        cv_v_species = compute_cv_ve(T_V, species_table)

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

    Implements Gnoffo step 13 (eqs. 97, 102, 103):
    T = T_ref + (e - e_V - e_0) / C_{v,tr}, a direct algebraic solve.

    Args:
        e: Total mixture internal energy [J/kg].
        e_V: Vibrational-electronic energy [J/kg].
        c_s: Mass fractions.
        cv_tr: Trans-rot specific heat per species [J/(kg*K)].
        e_s0: Reference internal energy per species [J/kg].
        T_ref: Reference temperature [K].

    Returns:
        T [K].
    """
    # Compute mixture reference energy: e_0 = sum c_s e_{s,0}
    e_0 = jnp.sum(c_s * e_s0[:, None], axis=0)  # Shape: (...)

    # Compute mixture C_v,tr: C_{v,tr} = sum c_s C_{v,tr}^s
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
        T: Temperature array [K].
        species_table: Species thermodynamic data.

    Returns:
        Specific enthalpy [J/kg] for all species.
    """
    T = jnp.atleast_1d(T)
    M = species_table.molar_masses
    R_over_M = constants.R_universal / M

    cv_tr = compute_cv_trans_rot(T, species_table.is_monoatomic, M)
    e_tr = cv_tr * T[None, :]
    e_ve = compute_e_ve(T, species_table)

    h = e_tr + e_ve + R_over_M[:, None] * T[None, :]

    T_ref = species_table.T_ref
    cv_tr_ref = compute_cv_trans_rot(
        jnp.array([T_ref]), species_table.is_monoatomic, M
    )
    e_tr_ref = cv_tr_ref[:, 0] * T_ref
    e_ve_ref = compute_e_ve(jnp.array([T_ref]), species_table)[:, 0]
    h_ref_current = e_tr_ref + e_ve_ref + R_over_M * T_ref
    offset = species_table.h_s0 - h_ref_current

    return h + offset[:, None]


def compute_cp(
    T: Float[Array, " N"],
    species_table: "SpeciesTable",
) -> Float[Array, "n_species N"]:
    """Compute specific heat at constant pressure for all species.

    Args:
        T: Temperature array [K].
        species_table: Species thermodynamic data.

    Returns:
        C_p [J/(kg*K)] for all species.
    """
    return species_table.energy_model.cp(T)


def compute_cv_tr(
    T: Float[Array, " N"],
    species_table: "SpeciesTable",
) -> Float[Array, "n_species N"]:
    """Compute translational-rotational specific heat for all species.

    Args:
        T: Temperature array [K] (not used, kept for API consistency).
        species_table: Species thermodynamic data.

    Returns:
        C_v,tr [J/(kg*K)] for all species.
    """
    return compute_cv_trans_rot(
        T,
        species_table.is_monoatomic,
        species_table.molar_masses,
    )


def compute_cp_tr(
    T: Float[Array, " N"],
    species_table: "SpeciesTable",
) -> Float[Array, "n_species N"]:
    """Compute translational-rotational C_p for all species.

    Uses C_p,tr = C_v,tr + R/M (ideal gas).
    """
    cv_tr = compute_cv_trans_rot(
        T,
        species_table.is_monoatomic,
        species_table.molar_masses,
    )
    R_over_M = constants.R_universal / species_table.molar_masses
    return cv_tr + R_over_M[:, None]


def compute_cv_ve(
    T_V: Float[Array, " N"],
    species_table: "SpeciesTable",
) -> Float[Array, "n_species N"]:
    """Compute vibrational-electronic specific heat for all species.

    Args:
        T_V: Vibrational temperature [K].
        species_table: Species thermodynamic data.

    Returns:
        C_v,V [J/(kg*K)] for all species.
    """
    return species_table.energy_model.cv_ve(T_V)


def compute_e_ve(
    T_V: Float[Array, " N"],
    species_table: "SpeciesTable",
) -> Float[Array, "n_species N"]:
    """Compute vibrational-electronic internal energy for all species.

    Args:
        T_V: Vibrational temperature [K].
        species_table: Species thermodynamic data.

    Returns:
        e_v [J/kg] for all species.
    """
    return species_table.energy_model.e_ve(T_V)


def compute_e_vib(
    T_V: Float[Array, " N"],
    species_table: "SpeciesTable",
) -> Float[Array, "n_species N"]:
    """Compute vibrational internal energy for all species."""
    return species_table.energy_model.e_vib(T_V)


def compute_e_el(
    T_V: Float[Array, " N"],
    species_table: "SpeciesTable",
) -> Float[Array, "n_species N"]:
    """Compute electronic internal energy for all species."""
    return species_table.energy_model.e_el(T_V)


def compute_electronic_energy_from_levels(
    T_el: Float[Array, " N"],
    g_i: Float[Array, " n_levels"],
    theta_el_i: Float[Array, " n_levels"],
    R_s: float | jnp.ndarray,
) -> Float[Array, " N"]:
    """Compute electronic internal energy e_el,s(T_el) [J/kg] from discrete electronic levels.

    Implements:
        e_el,s = R_s * (sum_{i!=0} g_i theta_i exp(-theta_i/T_el)) /
                       (sum_i     g_i         exp(-theta_i/T_el))

    Args:
        T_el: Electronic temperature [K].
        g_i: Degeneracy for each level including ground state i=0.
        theta_el_i: Characteristic temperatures [K] for each level, theta_0=0.
        R_s: Species gas constant [J/(kg*K)] (R_u / M_s).

    Returns:
        Electronic internal energy [J/kg], same shape as T_el.
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


def compute_electronic_energy_from_levels_batched(
    T_el: Float[Array, " N"],
    g_i: Float[Array, "n_species n_levels"],
    theta_el_i: Float[Array, "n_species n_levels"],
    molar_masses: Float[Array, " n_species"],
) -> Float[Array, "n_species N"]:
    """Compute electronic energy for all species using padded electronic levels."""
    R_s = constants.R_universal / molar_masses

    def _single_species(
        g: Float[Array, " n_levels"],
        theta: Float[Array, " n_levels"],
        R: float,
    ) -> Float[Array, " N"]:
        return compute_electronic_energy_from_levels(T_el, g, theta, R)

    return jax.vmap(_single_species, in_axes=(0, 0, 0))(g_i, theta_el_i, R_s)


def compute_cv_electronic_from_levels(
    T_el: Float[Array, " N"],
    g_i: Float[Array, " n_levels"],
    theta_el_i: Float[Array, " n_levels"],
    R_s: float | jnp.ndarray,
) -> Float[Array, " N"]:
    """Compute electronic specific heat from discrete electronic levels."""
    T_el = jnp.asarray(T_el)
    T_safe = jnp.maximum(T_el, 1e-12)

    theta = theta_el_i[:, None] if T_el.ndim > 0 else theta_el_i
    g = g_i[:, None] if T_el.ndim > 0 else g_i

    w = g * jnp.exp(-theta / T_safe)
    Z = jnp.sum(w, axis=0)

    # U = sum_i g_i * theta_i * exp(-theta_i/T) (ground state theta=0 is fine)
    U = jnp.sum(w * theta, axis=0)

    # Derivatives w.r.t T: d/dT exp(-theta/T) = exp(-theta/T) * theta / T^2
    T_inv2 = 1.0 / (T_safe**2)
    Z_prime = jnp.sum(w * theta, axis=0) * T_inv2
    U_prime = jnp.sum(w * theta**2, axis=0) * T_inv2

    # d(U/Z)/dT = (U' Z - U Z') / Z^2
    Z_safe = jnp.clip(Z, 1e-300, None)
    return R_s * (U_prime * Z_safe - U * Z_prime) / (Z_safe**2)


def compute_cv_electronic_from_levels_batched(
    T_el: Float[Array, " N"],
    g_i: Float[Array, "n_species n_levels"],
    theta_el_i: Float[Array, "n_species n_levels"],
    molar_masses: Float[Array, " n_species"],
) -> Float[Array, "n_species N"]:
    """Compute electronic specific heat for all species using padded levels."""
    R_s = constants.R_universal / molar_masses

    def _single_species(
        g: Float[Array, " n_levels"],
        theta: Float[Array, " n_levels"],
        R: float,
    ) -> Float[Array, " N"]:
        return compute_cv_electronic_from_levels(T_el, g, theta, R)

    return jax.vmap(_single_species, in_axes=(0, 0, 0))(g_i, theta_el_i, R_s)
