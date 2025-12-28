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


def compute_equilibrium_enthalpy_polynomial(
    T_V: Float[Array, " N"],
    T_limit_low: Float[Array, "n_species n_ranges"],
    T_limit_high: Float[Array, "n_species n_ranges"],
    parameters: Float[Array, "n_species n_ranges n_parameters"],
    molar_masses: Float[Array, " n_species"],
) -> Float[Array, "n_species N"]:
    """Compute equilibrium enthalpy for all species using polynomial curve fits.

    This function computes specific enthalpy h(T) for all species simultaneously
    using temperature-dependent polynomial fits from NASA or similar databases.

    Args:
        T_V: Temperature array [K], shape (N,)
        T_limit_low: Lower temperature bounds for each range [K], shape (n_species, n_ranges)
        T_limit_high: Upper temperature bounds for each range [K], shape (n_species, n_ranges)
        parameters: Polynomial coefficients for each range, shape (n_species, n_ranges, n_parameters)
        molar_masses: Molecular mass for each species [kg/kmol], shape (n_species,)

    Returns:
        Specific enthalpy [J/kg] for all species, shape (n_species, N)

    Notes:
        - Uses vmap to vectorize over species dimension
        - Polynomial form: h = (R/M) * (a1*T + a2*T^2/2 + ... + a6)
        - Temperature ranges are checked to select correct polynomial coefficients
    """

    n_ranges = T_limit_low.shape[1]

    def h_single_species(
        T_low: Float[Array, " n_ranges"],
        T_high: Float[Array, " n_ranges"],
        params: Float[Array, "n_ranges n_parameters"],
        M: float,
    ) -> Float[Array, " N"]:
        """Compute enthalpy for one species across all temperatures."""
        h_V = jnp.zeros_like(T_V)

        # Loop over temperature ranges (small fixed number, JAX will unroll)
        for i in range(n_ranges):
            # Mask for temperatures in this range
            mask = (T_V >= T_low[i]) & (T_V < T_high[i])

            # Extract polynomial coefficients for this range
            a = params[i, :]

            # Evaluate polynomial: h = (R/M) * (a0*T + a1*T^2/2 + ... + a5)
            h_range = (
                constants.R_universal
                / (M / 1e3)
                * (
                    a[0] * T_V**1 / 1
                    + a[1] * T_V**2 / 2
                    + a[2] * T_V**3 / 3
                    + a[3] * T_V**4 / 4
                    + a[4] * T_V**5 / 5
                    + a[5]
                )
            )

            # Update h_V only where mask is True
            h_V = jnp.where(mask, h_range, h_V)

        return h_V

    # Vectorize over species dimension (axis 0 of each input)
    h_vectorized = jax.vmap(h_single_species, in_axes=(0, 0, 0, 0))

    return h_vectorized(T_limit_low, T_limit_high, parameters, molar_masses)


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
        molar_masses: Molar mass [kg/kmol], shape (n_species,)

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
                / (M / 1e3)
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
        molar_masses: Molar mass [kg/kmol], shape (n_species,)

    Returns:
        C_v,tr [J/(kg·K)], shape (n_species, N)

    Notes:
        - Gnoffo eq. 29 uses C_v,t + C_v,r as constants
        - C_v,t = (3/2) R/M for all species
        - C_v,r = R/M for molecules, 0 for atoms
        - Combined: C_v,tr = (5/2) R/M (molecules), (3/2) R/M (atoms)
    """
    R = constants.R_universal  # J/(mol·K)
    M = molar_masses / 1e3  # Convert kg/kmol -> kg/mol

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
        molar_masses: Molar mass [kg/kmol], shape (n_species,)

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
    M = molar_masses / 1e3  # kg/kmol -> kg/mol
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
        molar_masses: Molar mass [kg/kmol], shape (n_species,)

    Returns:
        e_v [J/kg], shape (n_species, N)

    Notes:
        - Integration performed analytically, not numerically
        - C_{v,V} derived from enthalpy polynomial (eqs. 29-31)
        - T_ref is pre-evaluated via functools.partial (not stored in SpeciesTable)
        - Same polynomial structure as h_equilibrium for consistency
    """
    n_ranges = T_limit_low.shape[1]

    def e_v_single_species(
        T_low: Float[Array, " n_ranges"],
        T_high: Float[Array, " n_ranges"],
        params: Float[Array, "n_ranges n_parameters"],
        is_monoatomic_s: float,
        M_s: float,
    ) -> Float[Array, " N"]:
        """Integrate C_{v,V} for one species."""

        R = constants.R_universal
        M_kg_mol = M_s / 1e3

        # Compute C_v,tr constant for this species
        cv_tr_s = jnp.where(~is_monoatomic_s, 2.5 * R / M_kg_mol, 1.5 * R / M_kg_mol)

        e_v = jnp.zeros_like(T_V)

        for i in range(n_ranges):
            mask = (T_V >= T_low[i]) & (T_V < T_high[i])

            # Get enthalpy polynomial coefficients [a0, a1, a2, a3, a4, a5]
            a = params[i, :]

            # C_p = dh/dT = (R/M) * (a0 + a1*T + a2*T^2 + a3*T^3 + a4*T^4)
            # C_v = C_p - R/M = (R/M) * (a0 + a1*T + a2*T^2 + a3*T^3 + a4*T^4) - R/M
            #                 = (R/M) * (a0 - 1 + a1*T + a2*T^2 + a3*T^3 + a4*T^4)
            # C_{v,V} = C_v - C_{v,tr}

            # Compute C_{v,V} polynomial coefficients
            R_over_M = R / M_kg_mol
            b_0 = R_over_M * (a[0] - 1) - cv_tr_s  # Constant term
            b_1 = R_over_M * a[1]  # Linear term
            b_2 = R_over_M * a[2]  # Quadratic term
            b_3 = R_over_M * a[3]  # Cubic term
            b_4 = R_over_M * a[4]  # Quartic term

            # Integrate: ∫C_{v,V} dT = b_0*T + b_1*T^2/2 + b_2*T^3/3 + b_3*T^4/4 + b_4*T^5/5
            def integrate_cv(T):
                return (
                    b_0 * T
                    + b_1 * T**2 / 2
                    + b_2 * T**3 / 3
                    + b_3 * T**4 / 4
                    + b_4 * T**5 / 5
                )

            # e_v = ∫_{T_ref}^{T_V} C_{v,V} dT' = F(T_V) - F(T_ref)
            e_v_range = integrate_cv(T_V) - integrate_cv(T_ref)

            e_v = jnp.where(mask, e_v_range, e_v)

        return e_v

    # Vectorize over species
    e_v_vectorized = jax.vmap(e_v_single_species, in_axes=(0, 0, 0, 0, 0))

    return e_v_vectorized(
        T_limit_low, T_limit_high, parameters, is_monoatomic, molar_masses
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
        molar_masses: Molar mass [kg/kmol], shape (n_species,)
        T_ref: Reference temperature [K]

    Returns:
        e_{s,0} [J/kg], shape (n_species,)

    Notes:
        - Used in Gnoffo eq. 97 and implied by eq. 102
        - Relationship: h = e + pV/m = e + RT/M for ideal gas
        - T_ref = 298.16 K in current implementation
    """
    R = constants.R_universal
    M = molar_masses / 1e3  # kg/kmol -> kg/mol

    e_s0 = h_s0 - R * T_ref / M

    return e_s0


def solve_vibrational_temperature_from_vibrational_energy(
    e_V_target: Float[Array, "..."],
    c_s: Float[Array, "n_species ..."],
    T_V_initial: Float[Array, "..."],
    species_table: "SpeciesTable",
    max_iterations: int = 20,
    rtol: float = 1e-6,
    atol: float = 1.0,  # [K]
) -> Float[Array, "..."]:
    """Solve for vibrational temperature T_V from vibrational energy e_V.

    Implements Gnoffo step 9: Find T_V such that
    Σ_s c_s e_{v,s}(T_V) = e_V

    Uses Newton-Raphson with JAX autodiff for Jacobian:
    T_V^{n+1} = T_V^n - f(T_V^n) / f'(T_V^n)

    where f(T_V) = Σ_s c_s e_{v,s}(T_V) - e_V

    Args:
        e_V_target: Target vibrational energy [J/kg], shape (...)
        c_s: Mass fractions, shape (n_species, ...)
        T_V_initial: Initial guess for T_V [K], shape (...)
        species_table: SpeciesTable containing thermodynamic data
        max_iterations: Maximum Newton iterations
        rtol: Relative tolerance for convergence
        atol: Absolute tolerance [K] for temperature convergence

    Returns:
        T_V [K], shape (...)

    Notes:
        - Uses JAX autodiff to compute df/dT_V automatically
        - Converges when |ΔT_V| < atol or |ΔT_V/T_V| < rtol
        - If species have no vibrational modes (e_v = 0), returns T_V_initial
    """
    # Extract coefficient data from species_table for use in nested functions
    T_ref = species_table.T_ref
    T_limit_low = species_table.T_limit_low
    T_limit_high = species_table.T_limit_high
    enthalpy_coeffs = species_table.enthalpy_coeffs
    is_monoatomic = species_table.is_monoatomic
    molar_masses = species_table.molar_masses

    def compute_e_v_internal(T_V: Float[Array, " N"]) -> Float[Array, "n_species N"]:
        """Compute vibrational energy using extracted coefficient data."""
        return compute_e_vib_electronic(
            T_V, T_ref, T_limit_low, T_limit_high, enthalpy_coeffs, is_monoatomic, molar_masses
        )

    # Flatten input for scalar Newton iteration
    original_shape = e_V_target.shape
    e_V_flat = e_V_target.flatten()
    T_V_flat = T_V_initial.flatten()
    c_s_flat = c_s.reshape(c_s.shape[0], -1)  # (n_species, N_flat)

    # Create a scalar residual function for each cell
    def scalar_residual_fn(T_V_scalar, e_V_target_scalar, c_s_col):
        """Residual for a single cell: f(T_V) = Σ c_s e_{v,s}(T_V) - e_V_target"""
        # Compute e_v for all species at this T_V
        e_v_species = compute_e_v_internal(
            jnp.array([T_V_scalar])
        )  # (n_species, 1)

        # Compute mixture vibrational energy
        e_V_computed = jnp.sum(c_s_col * e_v_species[:, 0])

        # Residual
        return e_V_computed - e_V_target_scalar

    # Vectorize the scalar Newton iteration across all cells
    def compute_residual_and_jacobian(T_V_scalar, e_V_target_scalar, c_s_col):
        """Compute residual and its derivative for a single cell."""
        res_and_grad = jax.value_and_grad(
            lambda t: scalar_residual_fn(t, e_V_target_scalar, c_s_col)
        )
        return res_and_grad(T_V_scalar)

    # Vectorize over all cells
    vectorized_compute = jax.vmap(compute_residual_and_jacobian, in_axes=(0, 0, 1))

    # Newton-Raphson iteration
    for _ in range(max_iterations):
        # Compute residual and Jacobian for all cells
        residual, jacobian_diag = vectorized_compute(T_V_flat, e_V_flat, c_s_flat)

        # Newton update: ΔT_V = -f / (df/dT_V)
        delta_T_V_full = -residual / jnp.clip(jacobian_diag, 1e-20, None)

        # Apply damping to prevent overshooting (limit step to 0.5 * T_V)
        # This helps convergence when far from solution
        max_step = 0.5 * jnp.abs(T_V_flat)
        delta_T_V = jnp.clip(delta_T_V_full, -max_step, max_step)

        # Update temperature (ensure positive)
        T_V_new = jnp.maximum(T_V_flat + delta_T_V, 50.0)  # Min temperature 50K

        # Check convergence
        abs_error = jnp.abs(delta_T_V)
        rel_error = abs_error / jnp.clip(jnp.abs(T_V_new), 1e-10, None)

        converged = (abs_error < atol) | (rel_error < rtol)

        # Update only non-converged cells (for JIT compatibility, we can't break early)
        # Keep converged cells at their current value
        T_V_flat = jnp.where(converged, T_V_flat, T_V_new)

    # Reshape back to original shape
    T_V_result = T_V_flat.reshape(original_shape)

    return T_V_result


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


# =============================================================================
# High-level API: Functions that take SpeciesTable as argument
# =============================================================================


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
    return compute_equilibrium_enthalpy_polynomial(
        T,
        species_table.T_limit_low,
        species_table.T_limit_high,
        species_table.enthalpy_coeffs,
        species_table.molar_masses,
    )


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
