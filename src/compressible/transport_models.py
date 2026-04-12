"""Runtime transport models for the compressible solver."""

from __future__ import annotations

import functools

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from . import chemistry_types, constants, thermodynamic_relations
from .equation_manager_types import EquationManager
from .transport_models_types import (
    CasseauTransportTable,
    CollisionIntegralTable,
    TransportModel,
)

T_REF_LOW = 2000.0
T_REF_HIGH = 4000.0
LN_T_REF_LOW = jnp.log(T_REF_LOW)
LN_T_REF_HIGH = jnp.log(T_REF_HIGH)
R_UNIVERSAL = constants.R_universal


def build_gnoffo_transport_model(
    *,
    species_table: chemistry_types.SpeciesTable,
    collision_integrals: CollisionIntegralTable | None,
    include_diffusion: bool,
) -> TransportModel:
    """Build the Gnoffo transport model."""
    compute_transport_properties = functools.partial(
        compute_transport_properties_gnoffo,
        species_table=species_table,
        collision_integrals=collision_integrals,
        include_diffusion=include_diffusion,
    )
    return TransportModel(compute_transport_properties=compute_transport_properties)


def build_casseau_transport_model(
    *,
    species_table: chemistry_types.SpeciesTable,
    casseau_transport: CasseauTransportTable,
    collision_integrals: CollisionIntegralTable | None,
    include_diffusion: bool,
) -> TransportModel:
    """Build the Casseau transport model."""
    compute_transport_properties = functools.partial(
        compute_transport_properties_casseau,
        species_table=species_table,
        casseau_transport=casseau_transport,
        collision_integrals=collision_integrals,
        include_diffusion=include_diffusion,
    )
    return TransportModel(compute_transport_properties=compute_transport_properties)


def _zeros_transport(
    n_cells: int, n_species: int
) -> tuple[
    Float[Array, " n_cells"],
    Float[Array, " n_cells"],
    Float[Array, " n_cells"],
    Float[Array, " n_cells"],
    Float[Array, "n_cells n_species"],
]:
    """Return zero transport properties for inviscid cases."""
    return (
        jnp.zeros(n_cells),
        jnp.zeros(n_cells),
        jnp.zeros(n_cells),
        jnp.zeros(n_cells),
        jnp.zeros((n_cells, n_species)),
    )


def _compute_molar_concentrations(
    Y_s: Float[Array, "n_cells n_species"],
    molar_masses: Float[Array, " n_species"],
) -> tuple[
    Float[Array, "n_cells n_species"], Float[Array, "n_cells n_species"]
]:
    """Compute concentration arrays used by the transport models."""
    Y_M = Y_s * molar_masses[None, :]
    c_s = Y_M / jnp.sum(Y_M, axis=1, keepdims=True)
    gamma_s = c_s / molar_masses[None, :]
    return c_s, gamma_s


def interpolate_collision_integral(
    T: Float[Array, "..."],
    omega_2000K: Float[Array, " n_pairs"],
    omega_4000K: Float[Array, " n_pairs"],
) -> Float[Array, "... n_pairs"]:
    """Interpolate collision integrals in log-temperature space."""
    slope = (omega_4000K - omega_2000K) / (LN_T_REF_HIGH - LN_T_REF_LOW)
    ln_T = jnp.log(jnp.clip(T, 300.0, 50000.0))
    log10_omega = omega_2000K + slope * (ln_T[..., None] - LN_T_REF_LOW)
    return jnp.power(10.0, log10_omega)


def compute_modified_collision_integral_1(
    T: Float[Array, "..."],
    M_s: Float[Array, " n_species"],
    M_r: Float[Array, " n_species"],
    pi_omega_11: Float[Array, "... n_pairs"],
    pair_indices_sr: Int[Array, "n_species n_species"],
) -> Float[Array, "... n_species n_species"]:
    """Compute the first modified collision integral."""
    M_s_grid, M_r_grid = jnp.meshgrid(M_s, M_r, indexing="ij")
    mass_factor = jnp.sqrt(
        2.0
        * M_s_grid
        * M_r_grid
        / (jnp.pi * constants.R_universal * (M_s_grid + M_r_grid))
    )
    pi_omega_sr = pi_omega_11[..., pair_indices_sr]
    pi_omega_sr_m2 = pi_omega_sr * 1e-4
    T_factor = 1.0 / jnp.sqrt(T[..., None, None])
    return (8.0 / 3.0) * mass_factor * T_factor * pi_omega_sr_m2


def compute_modified_collision_integral_2(
    T: Float[Array, "..."],
    M_s: Float[Array, " n_species"],
    M_r: Float[Array, " n_species"],
    pi_omega_22: Float[Array, "... n_pairs"],
    pair_indices_sr: Int[Array, "n_species n_species"],
) -> Float[Array, "... n_species n_species"]:
    """Compute the second modified collision integral."""
    M_s_grid, M_r_grid = jnp.meshgrid(M_s, M_r, indexing="ij")
    mass_factor = jnp.sqrt(
        2.0 * M_s_grid * M_r_grid / (jnp.pi * R_UNIVERSAL * (M_s_grid + M_r_grid))
    )
    pi_omega_sr = pi_omega_22[..., pair_indices_sr]
    pi_omega_sr_m2 = pi_omega_sr * 1e-4
    T_factor = 1.0 / jnp.sqrt(T[..., None, None])
    return (16.0 / 5.0) * mass_factor * T_factor * pi_omega_sr_m2


def build_pair_index_matrix(
    species_names: tuple[str, ...],
    collision_integrals: CollisionIntegralTable,
) -> Int[Array, "n_species n_species"]:
    """Build a lookup matrix from species pairs to collision-integral rows."""
    n_species = len(species_names)
    indices = jnp.zeros((n_species, n_species), dtype=jnp.int32)

    for i, species_s in enumerate(species_names):
        for j, species_r in enumerate(species_names):
            try:
                pair_index = collision_integrals.get_pair_index(species_s, species_r)
            except ValueError:
                try:
                    pair_index = collision_integrals.get_pair_index(species_s, species_s)
                except ValueError:
                    pair_index = 0
            indices = indices.at[i, j].set(pair_index)

    return indices


def compute_mixture_viscosity(
    T: Float[Array, " n_cells"],
    gamma_s: Float[Array, "n_cells n_species"],
    M_s: Float[Array, " n_species"],
    delta_2_sr: Float[Array, "n_cells n_species n_species"],
) -> Float[Array, " n_cells"]:
    """Compute the Gnoffo mixture viscosity."""
    del T
    m_s = M_s / constants.N_A
    denominator_per_s = jnp.einsum("cr,csr->cs", gamma_s, delta_2_sr)
    return jnp.sum(
        m_s * gamma_s / jnp.clip(denominator_per_s, 1e-30, None), axis=-1
    )


def compute_translational_thermal_conductivity(
    T: Float[Array, " n_cells"],
    gamma_s: Float[Array, "n_cells n_species"],
    M_s: Float[Array, " n_species"],
    delta_2_sr: Float[Array, "n_cells n_species n_species"],
) -> Float[Array, " n_cells"]:
    """Compute the Gnoffo translational thermal conductivity."""
    del T
    M_s_grid, M_r_grid = jnp.meshgrid(M_s, M_s, indexing="ij")
    m_ratio = M_s_grid / M_r_grid
    a_sr = 1.0 + (1.0 - m_ratio) * (0.45 - 2.54 * m_ratio) / jnp.square(
        1.0 + m_ratio
    )
    weighted_delta = a_sr[None, :, :] * delta_2_sr
    denominator_per_s = jnp.einsum("cr,csr->cs", gamma_s, weighted_delta)
    return 15.0 * constants.k / 4.0 * jnp.sum(
        gamma_s / jnp.clip(denominator_per_s, 1e-30, None), axis=-1
    )


def compute_rotational_thermal_conductivity(
    T: Float[Array, " n_cells"],
    gamma_s: Float[Array, "n_cells n_species"],
    is_molecule: Bool[Array, " n_species"],
    delta_1_sr: Float[Array, "n_cells n_species n_species"],
) -> Float[Array, " n_cells"]:
    """Compute the Gnoffo rotational thermal conductivity."""
    del T
    denominator_per_s = jnp.einsum("cr,csr->cs", gamma_s, delta_1_sr)
    return constants.k * jnp.sum(
        is_molecule[None, :] * gamma_s / jnp.clip(denominator_per_s, 1e-30, None),
        axis=-1,
    )


def compute_vibrational_thermal_conductivity(
    T_v: Float[Array, " n_cells"],
    gamma_s: Float[Array, "n_cells n_species"],
    is_molecule: Bool[Array, " n_species"],
    delta_1_sr: Float[Array, "n_cells n_species n_species"],
) -> Float[Array, " n_cells"]:
    """Compute the Gnoffo vibrational thermal conductivity."""
    return compute_rotational_thermal_conductivity(
        T_v, gamma_s, is_molecule, delta_1_sr
    )


def compute_binary_diffusion_coefficient(
    T: Float[Array, " n_cells"],
    p: Float[Array, " n_cells"],
    delta_1_sr: Float[Array, "n_cells n_species n_species"],
) -> Float[Array, "n_cells n_species n_species"]:
    """Compute the Gnoffo binary diffusion coefficients."""
    return (
        constants.k
        * T[:, None, None]
        / (p[:, None, None] * jnp.clip(delta_1_sr, 1e-30, None))
    )


def compute_effective_diffusion_coefficient(
    gamma_s: Float[Array, "n_cells n_species"],
    M_s: Float[Array, " n_species"],
    D_sr: Float[Array, "n_cells n_species n_species"],
) -> Float[Array, "n_cells n_species"]:
    """Compute the Gnoffo effective diffusion coefficients."""
    n_species = gamma_s.shape[1]
    gamma_t = jnp.sum(gamma_s, axis=-1, keepdims=True)
    numerator = jnp.square(gamma_t) * M_s * (1.0 - M_s * gamma_s)
    off_diag_mask = 1.0 - jnp.eye(n_species)
    gamma_over_D = gamma_s[:, None, :] / jnp.clip(D_sr, 1e-30, None)
    denominator = jnp.sum(off_diag_mask * gamma_over_D, axis=-1)
    return numerator / jnp.clip(denominator, 1e-30, None)


def compute_species_viscosity_blottner(
    T: Float[Array, " n_cells"],
    table: CasseauTransportTable,
) -> Float[Array, "n_species n_cells"]:
    """Compute species viscosity using the Casseau Blottner law."""
    log_T = jnp.log(jnp.clip(T, 1e-12, None))
    A = table.blottner_A[:, None]
    B = table.blottner_B[:, None]
    C = table.blottner_C[:, None]
    return 0.1 * jnp.exp((A * log_T + B) * log_T + C)


def compute_species_viscosity_powerlaw(
    T: Float[Array, " n_cells"],
    table: CasseauTransportTable,
    molar_masses: Float[Array, " n_species"],
) -> Float[Array, "n_species n_cells"]:
    """Compute species viscosity using the Casseau power-law form."""
    m_s = molar_masses / constants.N_A
    d_ref = table.d_ref
    omega = table.omega
    mu_ref = (
        15.0
        * jnp.sqrt(jnp.pi * m_s * constants.k * table.T_ref)
        / (2.0 * jnp.pi * d_ref**2 * (5.0 - 2.0 * omega) * (7.0 - 2.0 * omega))
    )
    return mu_ref[:, None] * (T[None, :] / table.T_ref) ** omega[:, None]


def compute_species_kappa_eucken(
    T: Float[Array, " n_cells"],
    T_v: Float[Array, " n_cells"],
    mu_s: Float[Array, "n_species n_cells"],
    molar_masses: Float[Array, " n_species"],
    is_monoatomic: Bool[Array, " n_species"],
    cv_ve: Float[Array, "n_species n_cells"],
) -> tuple[
    Float[Array, "n_species n_cells"],
    Float[Array, "n_species n_cells"],
    Float[Array, "n_species n_cells"],
]:
    """Compute species conductivities using the Casseau Eucken relations."""
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
    """Mix a species property with the Wilke rule."""
    X_sc = X_s.T
    mu_safe = jnp.clip(mu_s, 1e-30, None)
    mu_ratio = mu_safe[:, None, :] / mu_safe[None, :, :]
    mass_ratio = (molar_masses[None, :, None] / molar_masses[:, None, None]) ** 0.25
    denom = jnp.sqrt(
        8.0 * (1.0 + (molar_masses[:, None] / molar_masses[None, :]))
    )
    term = (1.0 + jnp.sqrt(mu_ratio) * mass_ratio) ** 2 / denom[:, :, None]
    term = term * (1.0 - jnp.eye(molar_masses.shape[0])[:, :, None])
    phi = X_sc + jnp.sum(X_sc[None, :, :] * term, axis=1)
    return jnp.sum(X_sc * prop_s / jnp.clip(phi, 1e-30, None), axis=0)


def compute_casseau_transport_properties(
    T: Float[Array, " n_cells"],
    T_v: Float[Array, " n_cells"],
    X_s: Float[Array, "n_cells n_species"],
    molar_masses: Float[Array, " n_species"],
    is_monoatomic: Bool[Array, " n_species"],
    cv_ve: Float[Array, "n_species n_cells"],
    table: CasseauTransportTable,
) -> tuple[
    Float[Array, " n_cells"],
    Float[Array, " n_cells"],
    Float[Array, " n_cells"],
    Float[Array, " n_cells"],
]:
    """Compute the Casseau mixture transport properties."""
    mu_s = compute_species_viscosity_blottner(T, table)
    eta_t_s, eta_r_s, eta_v_s = compute_species_kappa_eucken(
        T, T_v, mu_s, molar_masses, is_monoatomic, cv_ve
    )
    mu_mix = wilke_mixing(mu_s, mu_s, X_s, molar_masses)
    eta_t = wilke_mixing(eta_t_s, mu_s, X_s, molar_masses)
    eta_r = wilke_mixing(eta_r_s, mu_s, X_s, molar_masses)
    eta_v = wilke_mixing(eta_v_s, mu_s, X_s, molar_masses)
    return mu_mix, eta_t, eta_r, eta_v


def compute_transport_properties_gnoffo(
    T: Float[Array, " n_cells"],
    T_v: Float[Array, " n_cells"],
    p: Float[Array, " n_cells"],
    Y_s: Float[Array, "n_cells n_species"],
    rho: Float[Array, " n_cells"],
    *,
    species_table: chemistry_types.SpeciesTable,
    collision_integrals: CollisionIntegralTable | None,
    include_diffusion: bool,
) -> tuple[
    Float[Array, " n_cells"],
    Float[Array, " n_cells"],
    Float[Array, " n_cells"],
    Float[Array, " n_cells"],
    Float[Array, "n_cells n_species"],
]:
    """Compute transport properties with the Gnoffo model."""
    del rho
    n_cells = T.shape[0]
    n_species = species_table.n_species

    if collision_integrals is None:
        return _zeros_transport(n_cells, n_species)

    M_s = species_table.molar_masses
    _c_s, gamma_s = _compute_molar_concentrations(Y_s, M_s)
    pair_indices = build_pair_index_matrix(species_table.names, collision_integrals)

    pi_omega_11 = interpolate_collision_integral(
        T,
        collision_integrals.omega_11_2000K,
        collision_integrals.omega_11_4000K,
    )
    pi_omega_22 = interpolate_collision_integral(
        T,
        collision_integrals.omega_22_2000K,
        collision_integrals.omega_22_4000K,
    )
    pi_omega_11_Tv = interpolate_collision_integral(
        T_v,
        collision_integrals.omega_11_2000K,
        collision_integrals.omega_11_4000K,
    )

    delta_1 = compute_modified_collision_integral_1(
        T, M_s, M_s, pi_omega_11, pair_indices
    )
    delta_2 = compute_modified_collision_integral_2(
        T, M_s, M_s, pi_omega_22, pair_indices
    )
    delta_1_Tv = compute_modified_collision_integral_1(
        T_v, M_s, M_s, pi_omega_11_Tv, pair_indices
    )

    mu = compute_mixture_viscosity(T, gamma_s, M_s, delta_2)
    eta_t = compute_translational_thermal_conductivity(T, gamma_s, M_s, delta_2)
    is_molecule = ~species_table.is_monoatomic.astype(bool)
    eta_r = compute_rotational_thermal_conductivity(T, gamma_s, is_molecule, delta_1)
    eta_v = compute_vibrational_thermal_conductivity(
        T_v, gamma_s, is_molecule, delta_1_Tv
    )

    if include_diffusion:
        D_sr = compute_binary_diffusion_coefficient(T, p, delta_1)
        D_s = compute_effective_diffusion_coefficient(gamma_s, M_s, D_sr)
    else:
        D_s = jnp.zeros((n_cells, n_species))

    return mu, eta_t, eta_r, eta_v, D_s


def compute_transport_properties_casseau(
    T: Float[Array, " n_cells"],
    T_v: Float[Array, " n_cells"],
    p: Float[Array, " n_cells"],
    Y_s: Float[Array, "n_cells n_species"],
    rho: Float[Array, " n_cells"],
    *,
    species_table: chemistry_types.SpeciesTable,
    casseau_transport: CasseauTransportTable,
    collision_integrals: CollisionIntegralTable | None,
    include_diffusion: bool,
) -> tuple[
    Float[Array, " n_cells"],
    Float[Array, " n_cells"],
    Float[Array, " n_cells"],
    Float[Array, " n_cells"],
    Float[Array, "n_cells n_species"],
]:
    """Compute transport properties with the Casseau model."""
    del rho
    n_cells = T.shape[0]
    n_species = species_table.n_species

    M_s = species_table.molar_masses
    _c_s, gamma_s = _compute_molar_concentrations(Y_s, M_s)
    Y_sum = jnp.sum(Y_s, axis=1, keepdims=True)
    X_s = Y_s / jnp.clip(Y_sum, 1e-30, None)
    cv_ve = thermodynamic_relations.compute_cv_ve(T_v, species_table)
    mu, eta_t, eta_r, eta_v = compute_casseau_transport_properties(
        T,
        T_v,
        X_s,
        M_s,
        species_table.is_monoatomic.astype(bool),
        cv_ve,
        casseau_transport,
    )

    if include_diffusion and collision_integrals is not None:
        pair_indices = build_pair_index_matrix(species_table.names, collision_integrals)
        pi_omega_11 = interpolate_collision_integral(
            T,
            collision_integrals.omega_11_2000K,
            collision_integrals.omega_11_4000K,
        )
        delta_1 = compute_modified_collision_integral_1(
            T, M_s, M_s, pi_omega_11, pair_indices
        )
        D_sr = compute_binary_diffusion_coefficient(T, p, delta_1)
        D_s = compute_effective_diffusion_coefficient(gamma_s, M_s, D_sr)
    else:
        D_s = jnp.zeros((n_cells, n_species))

    return mu, eta_t, eta_r, eta_v, D_s


def compute_transport_properties(
    T: Float[Array, " n_cells"],
    T_v: Float[Array, " n_cells"],
    p: Float[Array, " n_cells"],
    Y_s: Float[Array, "n_cells n_species"],
    rho: Float[Array, " n_cells"],
    equation_manager: EquationManager,
) -> tuple[
    Float[Array, " n_cells"],
    Float[Array, " n_cells"],
    Float[Array, " n_cells"],
    Float[Array, " n_cells"],
    Float[Array, "n_cells n_species"],
]:
    """Compute transport properties for the active transport model."""
    transport_model = equation_manager.transport_model
    if transport_model is None:
        return _zeros_transport(T.shape[0], equation_manager.species.n_species)
    return transport_model.compute_transport_properties(T, T_v, p, Y_s, rho)
