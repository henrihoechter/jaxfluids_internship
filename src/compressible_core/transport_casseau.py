"""Transport properties based on Casseau (PhD, 2021).

Implements species viscosity using Blottner's formula, Eucken thermal
conductivity for trans-rot and vib-electronic modes, and Wilke mixing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from compressible_core import constants


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class CasseauTransportTable:
    """Casseau transport data for a set of species."""

    species_names: tuple[str, ...] = field(metadata=dict(static=True))
    d_ref: Float[Array, " n_species"]  # [m]
    omega: Float[Array, " n_species"]  # [-]
    blottner_A: Float[Array, " n_species"]  # [-]
    blottner_B: Float[Array, " n_species"]  # [-]
    blottner_C: Float[Array, " n_species"]  # [-]
    T_ref: float = field(metadata=dict(static=True))


def load_casseau_transport_table(
    json_path: str | Path, species_names: Sequence[str]
) -> CasseauTransportTable:
    """Load Casseau transport coefficients for selected species."""
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    entries = {entry["name"]: entry for entry in data["species"]}

    missing = [name for name in species_names if name not in entries]
    if missing:
        raise ValueError(f"Casseau transport data missing for species: {missing}")

    d_ref = []
    omega = []
    A = []
    B = []
    C = []
    for name in species_names:
        entry = entries[name]
        d_ref.append(float(entry["d_ref"]))
        omega.append(float(entry["omega"]))
        A.append(float(entry["blottner_A"]))
        B.append(float(entry["blottner_B"]))
        C.append(float(entry["blottner_C"]))

    return CasseauTransportTable(
        species_names=tuple(species_names),
        d_ref=jnp.array(d_ref),
        omega=jnp.array(omega),
        blottner_A=jnp.array(A),
        blottner_B=jnp.array(B),
        blottner_C=jnp.array(C),
        T_ref=float(data.get("T_ref", 273.0)),
    )


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


def split_cv_tr(
    molar_masses: Float[Array, " n_species"],
    is_monoatomic: Float[Array, " n_species"],
) -> tuple[Float[Array, " n_species"], Float[Array, " n_species"]]:
    """Return translational and rotational Cv for each species."""
    R_s = constants.R_universal / molar_masses
    cv_t = 1.5 * R_s
    cv_r = jnp.where(is_monoatomic, 0.0, 1.0 * R_s)
    return cv_t, cv_r


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
    cv_t, cv_r = split_cv_tr(molar_masses, is_monoatomic)

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
