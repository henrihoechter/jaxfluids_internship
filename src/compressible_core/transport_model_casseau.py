"""Transport properties based on Casseau (PhD, 2021).

Implements species viscosity using Blottner's formula, Eucken thermal
conductivity for trans-rot and vib-electronic modes, and Wilke mixing.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from compressible_core import constants, thermodynamic_relations
from compressible_core import transport_model_casseau_utils as _transport_model_casseau_utils
from compressible_core.transport_casseau_types import CasseauTransportTable

# Backwards-compatible re-export for external callers.
load_casseau_transport_table = _transport_model_casseau_utils.load_casseau_transport_table


def compute_species_viscosity_blottner(
    T: Float[Array, " n_cells"],
    table: CasseauTransportTable,
) -> Float[Array, "n_species n_cells"]:
    """Compute species viscosity using Blottner's formula (Eq. 2.15)."""
    log_T = jnp.log(jnp.clip(T, 1e-12, None))
    A = table.blottner_A[:, None]
    B = table.blottner_B[:, None]
    C = table.blottner_C[:, None]
    mu = 0.1 * jnp.exp((A * log_T + B) * log_T + C)
    return mu


def compute_species_viscosity_powerlaw(
    T: Float[Array, " n_cells"],
    table: CasseauTransportTable,
    molar_masses: Float[Array, " n_species"],
) -> Float[Array, "n_species n_cells"]:
    """Compute species viscosity using the power-law model (Eq. 2.13-2.14)."""
    m_s = molar_masses / constants.N_A  # [kg]
    d_ref = table.d_ref
    omega = table.omega

    mu_ref = (
        15.0
        * jnp.sqrt(jnp.pi * m_s * constants.k * table.T_ref)
        / (2.0 * jnp.pi * d_ref**2 * (5.0 - 2.0 * omega) * (7.0 - 2.0 * omega))
    )

    mu = mu_ref[:, None] * (T[None, :] / table.T_ref) ** omega[:, None]
    return mu


def compute_species_kappa_eucken(
    T: Float[Array, " n_cells"],
    T_v: Float[Array, " n_cells"],
    mu_s: Float[Array, "n_species n_cells"],
    molar_masses: Float[Array, " n_species"],
    is_monoatomic: Float[Array, " n_species"],
    cv_ve: Float[Array, "n_species n_cells"],
) -> tuple[Float[Array, "n_species n_cells"], Float[Array, "n_species n_cells"], Float[Array, "n_species n_cells"]]:
    """Compute species thermal conductivities using Eucken relations."""
    del T, T_v
    cv_t = thermodynamic_relations.compute_cv_t(molar_masses)
    cv_r = thermodynamic_relations.compute_cv_r(molar_masses, is_monoatomic)

    eta_t = 2.5 * mu_s * cv_t[:, None]
    eta_r = mu_s * cv_r[:, None]
    eta_v = 1.2 * mu_s * cv_ve
    return eta_t, eta_r, eta_v


def wilke_mixing(
    prop_s: Float[Array, "n_species n_cells"],
    mu_s: Float[Array, "n_species n_cells"],
    X_s: Float[Array, "n_cells n_species"],
    molar_masses: Float[Array, " n_species"],
) -> Float[Array, " n_cells"]:
    """Wilke mixing rule for mixture properties (Eq. 2.18-2.19)."""
    X_sc = X_s.T  # [n_species, n_cells]
    mu_safe = jnp.clip(mu_s, 1e-30, None)

    mu_ratio = mu_safe[:, None, :] / mu_safe[None, :, :]
    mass_ratio = (molar_masses[None, :, None] / molar_masses[:, None, None]) ** 0.25
    denom = jnp.sqrt(
        8.0
        * (
            1.0
            + (molar_masses[:, None] / molar_masses[None, :])
        )
    )

    term = (1.0 + jnp.sqrt(mu_ratio) * mass_ratio) ** 2 / denom[:, :, None]
    eye = jnp.eye(molar_masses.shape[0])[:, :, None]
    term = term * (1.0 - eye)

    phi = X_sc + jnp.sum(X_sc[None, :, :] * term, axis=1)
    prop_mix = jnp.sum(X_sc * prop_s / jnp.clip(phi, 1e-30, None), axis=0)
    return prop_mix


def compute_casseau_transport_properties(
    T: Float[Array, " n_cells"],
    T_v: Float[Array, " n_cells"],
    X_s: Float[Array, "n_cells n_species"],
    molar_masses: Float[Array, " n_species"],
    is_monoatomic: Float[Array, " n_species"],
    cv_ve: Float[Array, "n_species n_cells"],
    table: CasseauTransportTable,
) -> tuple[
    Float[Array, " n_cells"],
    Float[Array, " n_cells"],
    Float[Array, " n_cells"],
    Float[Array, " n_cells"],
]:
    """Return mixture mu, eta_t, eta_r, eta_v using Casseau models."""
    mu_s = compute_species_viscosity_blottner(T, table)
    eta_t_s, eta_r_s, eta_v_s = compute_species_kappa_eucken(
        T, T_v, mu_s, molar_masses, is_monoatomic, cv_ve
    )

    mu_mix = wilke_mixing(mu_s, mu_s, X_s, molar_masses)
    eta_t = wilke_mixing(eta_t_s, mu_s, X_s, molar_masses)
    eta_r = wilke_mixing(eta_r_s, mu_s, X_s, molar_masses)
    eta_v = wilke_mixing(eta_v_s, mu_s, X_s, molar_masses)

    return mu_mix, eta_t, eta_r, eta_v
