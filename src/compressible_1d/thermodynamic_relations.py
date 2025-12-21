"""Thermodynamic property calculations for species.

This module contains pure functions for computing thermodynamic properties
that can be used with functools.partial to create species-specific callables.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from compressible_1d import constants


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
            h_range = constants.R_universal / (M / 1e3) * (
                a[0] * T_V**1 / 1
                + a[1] * T_V**2 / 2
                + a[2] * T_V**3 / 3
                + a[3] * T_V**4 / 4
                + a[4] * T_V**5 / 5
                + a[5]
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
            cp_range = constants.R_universal / (M / 1e3) * (
                a[0] + a[1] * T + a[2] * T**2 + a[3] * T**3 + a[4] * T**4
            )

            cp = jnp.where(mask, cp_range, cp)

        return cp

    # Vectorize over species dimension (same pattern as h_equilibrium)
    cp_vectorized = jax.vmap(cp_single_species, in_axes=(0, 0, 0, 0))

    return cp_vectorized(T_limit_low, T_limit_high, parameters, molar_masses)
