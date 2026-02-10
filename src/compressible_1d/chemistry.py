"""Chemical kinetics interface (Casseau CVDV-QP).

Implements reaction-agnostic Arrhenius kinetics, Casseau equilibrium constants,
and the CVDV-QP chemistry-vibration coupling in a JAX-compatible way.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from compressible_1d import chemistry_types, constants
from compressible_1d.chemistry_types import ChemistryModel, ChemistryModelConfig

# Fixed level count for JAX-static vibrational sums
CVDV_MAX_LEVELS = 200
_EXP_CLIP = 700.0
_TINY = 1e-300


def arrhenius_rate(
    T_c: Float[Array, "n_reactions n_cells"],
    C_f: Float[Array, " n_reactions"],
    n_f: Float[Array, " n_reactions"],
    E_f_over_k: Float[Array, " n_reactions"],
) -> Float[Array, "n_reactions n_cells"]:
    """Compute Arrhenius forward rate coefficient.

    Args:
        T_c: Controlling temperature [K]. Shape [n_reactions, n_cells].
        C_f: Pre-exponential factor [m^3/mol/s] or [m^6/mol^2/s]. Shape [n_reactions].
        n_f: Temperature exponent [-]. Shape [n_reactions].
        E_f_over_k: Activation energy / k [K]. Shape [n_reactions].

    Returns:
        k_f: Forward rate coefficient [m^3/mol/s] or [m^6/mol^2/s].
            Shape [n_reactions, n_cells].
    """
    T_safe = jnp.clip(T_c, 1e-12, None)
    return (
        C_f[:, None]
        * jnp.power(T_safe, n_f[:, None])
        * jnp.exp(jnp.clip(-E_f_over_k[:, None] / T_safe, -_EXP_CLIP, _EXP_CLIP))
    )


def reaction_rate_products(
    log_n_s: Float[Array, "n_cells n_species"],
    stoich: Float[Array, "n_reactions n_species"],
) -> Float[Array, "n_reactions n_cells"]:
    """Compute log product of concentrations for each reaction.

    Args:
        log_n_s: log(molar concentration) log[n_s] with n_s in [mol/m^3].
            Shape [n_cells, n_species].
        stoich: Stoichiometric coefficients for reactants or products.
            Shape [n_reactions, n_species].

    Returns:
        log_prod: log(Π n_s^{nu_s}) for each reaction. Shape [n_reactions, n_cells].
    """
    return jnp.einsum("rs,sc->rc", stoich, log_n_s.T)


def _interp_casseau_coeffs(
    coeffs: Float[Array, "n_refs 6"],
    log_n_mix: Float[Array, " n_cells"],
) -> Float[Array, "n_cells 5"]:
    """Interpolate Casseau coefficients in log-log space for one reaction."""
    n_ref = jnp.clip(coeffs[:, 0], _TINY, None)
    log_n_ref = jnp.log(n_ref)
    A = coeffs[:, 1:]  # [n_refs, 5]

    idx = jnp.searchsorted(log_n_ref, log_n_mix, side="right") - 1
    idx = jnp.clip(idx, 0, n_ref.shape[0] - 2)
    x0 = log_n_ref[idx]
    x1 = log_n_ref[idx + 1]
    w = (log_n_mix - x0) / (x1 - x0 + _TINY)

    A0 = A[idx]
    A1 = A[idx + 1]
    return (1.0 - w)[:, None] * A0 + w[:, None] * A1


def equilibrium_constant_casseau(
    T_c: Float[Array, "..."],
    n_mix: Float[Array, " n_cells"],
    coeffs: Float[Array, "n_reactions n_refs 6"],
) -> Float[Array, "n_reactions n_cells"]:
    """Compute Casseau equilibrium constant K_eq (Eq. 2.69).

    Args:
        T_c: Controlling temperature for backward reaction [K]. Shape [n_cells]
            or [n_reactions, n_cells].
        n_mix: Mixture number density [1/m^3]. Shape [n_cells].
        coeffs: Casseau coefficient tables per reaction with columns
            [n_ref_m3, A0, A1, A2, A3, A4]. Shape [n_reactions, n_refs, 6].

    Returns:
        K_eq: Equilibrium constant (dimensionless). Shape [n_reactions, n_cells].
    """
    log_n_mix = jnp.log(jnp.clip(n_mix, _TINY, None))
    A_interp = jax.vmap(_interp_casseau_coeffs, in_axes=(0, None))(coeffs, log_n_mix)
    # Map A0..A4 -> A1..A5 in Eq. 2.69
    A1 = A_interp[..., 0]
    A2 = A_interp[..., 1]
    A3 = A_interp[..., 2]
    A4 = A_interp[..., 3]
    A5 = A_interp[..., 4]

    if T_c.ndim == 1:
        T_rc = jnp.broadcast_to(T_c[None, :], (coeffs.shape[0], T_c.shape[0]))
    else:
        T_rc = T_c

    T_safe = jnp.clip(T_rc, 1e-12, None)
    x = 1.0e4 / T_safe
    exponent = (
        A1 * (T_safe / 1.0e4)
        + A2
        + A3 * jnp.log(x)
        + A4 * x
        + A5 * x**2
    )
    return jnp.exp(jnp.clip(exponent, -_EXP_CLIP, _EXP_CLIP))


def _cvdv_level_counts(
    theta_vib: Float[Array, " n_items"],
    dissociation_energy: Float[Array, " n_items"],
    molar_masses: Float[Array, " n_items"],
) -> Float[Array, " n_items"]:
    """Compute CVDV vibrational level cutoff N_m for each item."""
    theta_safe = jnp.clip(theta_vib, 1e-12, None)
    D_safe = jnp.where(jnp.isfinite(dissociation_energy), dissociation_energy, 0.0)
    m_particle = molar_masses / constants.N_A  # [kg]
    N_m = jnp.floor(D_safe * m_particle / (constants.k * theta_safe))
    return jnp.clip(N_m, 0.0, CVDV_MAX_LEVELS - 1.0)


def _cvdv_partition_function(
    T_sc: Float[Array, "n_items n_cells"],
    theta_vib: Float[Array, " n_items"],
    N_m: Float[Array, " n_items"],
) -> Float[Array, "n_items n_cells"]:
    """Compute partition function Z(T) with fixed-level masking."""
    alpha = jnp.arange(CVDV_MAX_LEVELS)[None, None, :]  # [1,1,L]
    theta = theta_vib[:, None, None]  # [n_items,1,1]
    T = T_sc[:, :, None]

    exponent = -alpha * theta / (T + _TINY)
    exponent = jnp.clip(exponent, -_EXP_CLIP, _EXP_CLIP)
    mask = alpha <= N_m[:, None, None]
    return jnp.sum(jnp.exp(exponent) * mask, axis=2)


def _cvdv_average_energy(
    T_sc: Float[Array, "n_items n_cells"],
    theta_vib: Float[Array, " n_items"],
    N_m: Float[Array, " n_items"],
) -> Float[Array, "n_items n_cells"]:
    """Compute average vibrational energy per particle [J]."""
    alpha = jnp.arange(CVDV_MAX_LEVELS)[None, None, :]
    theta = theta_vib[:, None, None]
    T = T_sc[:, :, None]

    exponent = -alpha * theta / (T + _TINY)
    exponent = jnp.clip(exponent, -_EXP_CLIP, _EXP_CLIP)
    mask = alpha <= N_m[:, None, None]

    exp_term = jnp.exp(exponent) * mask
    Z = jnp.sum(exp_term, axis=2)
    epsilon = alpha * constants.k * theta
    numerator = jnp.sum(epsilon * exp_term, axis=2)
    return numerator / (Z + _TINY)


def _dissociation_species_indices(
    reactant_stoich: Float[Array, "n_reactions n_species"],
    product_stoich: Float[Array, "n_reactions n_species"],
    is_monoatomic: Float[Array, " n_species"],
) -> tuple[Float[Array, " n_reactions"], Float[Array, " n_reactions"]]:
    """Infer dissociating species indices for each reaction.

    Returns:
        indices: Index of inferred dissociating species (argmax of candidates).
        counts: Number of candidate dissociating species per reaction.
    """
    candidates = (reactant_stoich > product_stoich) & (~is_monoatomic[None, :])
    counts = jnp.sum(candidates, axis=1)
    indices = jnp.argmax(candidates, axis=1)
    return indices, counts


def validate_cvdv_reaction_table(
    species_table: chemistry_types.SpeciesTable,
    reaction_table: chemistry_types.ReactionTable,
) -> None:
    """Validate that each dissociation reaction has exactly one molecular reactant.

    This is a non-JIT helper intended for pre-validation.
    """
    import numpy as np

    reactant = np.asarray(reaction_table.reactant_stoich)
    product = np.asarray(reaction_table.product_stoich)
    is_mono = np.asarray(species_table.is_monoatomic).astype(bool)
    is_dissoc = np.asarray(reaction_table.is_dissociation) > 0.5

    candidates = (reactant > product) & (~is_mono[None, :])
    counts = candidates.sum(axis=1)
    bad = np.where(is_dissoc & (counts != 1))[0]
    if bad.size:
        raise ValueError(
            "CVDV dissociation reactions must have exactly one molecular reactant. "
            f"Invalid reactions: {bad.tolist()}"
        )


def _compute_forward_rate_coefficient_cvdv_qp(
    T_tr: Float[Array, " n_cells"],
    T_v: Float[Array, " n_cells"],
    species_table: chemistry_types.SpeciesTable,
    reaction_table: chemistry_types.ReactionTable,
) -> Float[Array, "n_reactions n_cells"]:
    """Compute forward rate coefficient using CVDV-QP for dissociation reactions."""
    n_reactions = reaction_table.n_reactions
    n_cells = T_tr.shape[0]

    T_tr_rc = jnp.broadcast_to(T_tr[None, :], (n_reactions, n_cells))
    T_v_rc = jnp.broadcast_to(T_v[None, :], (n_reactions, n_cells))

    is_dissoc = reaction_table.is_dissociation > 0.5
    is_eid = reaction_table.is_electron_impact > 0.5

    # Arrhenius for non-dissociation reactions
    T_control = jnp.where(is_eid[:, None], T_v_rc, T_tr_rc)
    k_f_arr = arrhenius_rate(
        T_control, reaction_table.C_f, reaction_table.n_f, reaction_table.E_f_over_k
    )

    # CVDV-QP for dissociation reactions
    dissoc_idx, _ = _dissociation_species_indices(
        reaction_table.reactant_stoich,
        reaction_table.product_stoich,
        species_table.is_monoatomic.astype(bool),
    )
    M_s = species_table.molar_masses
    D_s = species_table.dissociation_energy
    theta_v = species_table.theta_vib

    M_m = M_s[dissoc_idx]
    D_m = D_s[dissoc_idx]
    theta_m = theta_v[dissoc_idx]

    # Safe parameters for non-dissociation reactions
    M_m = jnp.where(is_dissoc, M_m, 1.0)
    D_m = jnp.where(is_dissoc, D_m, 0.0)
    theta_m = jnp.where(is_dissoc, theta_m, 1.0)

    R_m = constants.R_universal / M_m  # [J/(kg K)]
    U_m = D_m / (3.0 * R_m + _TINY)  # [K]
    U_m = jnp.where(is_dissoc, U_m, 1.0e30)

    inv_TF = (
        1.0 / jnp.clip(T_v_rc, 1e-12, None)
        - 1.0 / jnp.clip(T_tr_rc, 1e-12, None)
        - 1.0 / jnp.clip(U_m[:, None], 1e-12, None)
    )
    inv_TF = jnp.where(jnp.abs(inv_TF) < _TINY, jnp.sign(inv_TF) * _TINY, inv_TF)
    T_F = 1.0 / inv_TF

    N_m = _cvdv_level_counts(theta_m, D_m, M_m)
    Z_Ttr = _cvdv_partition_function(T_tr_rc, theta_m, N_m)
    Z_Tv = _cvdv_partition_function(T_v_rc, theta_m, N_m)
    Z_TF = _cvdv_partition_function(T_F, theta_m, N_m)
    Z_U = _cvdv_partition_function(-U_m[:, None], theta_m, N_m)

    k_f_base = arrhenius_rate(
        T_tr_rc, reaction_table.C_f, reaction_table.n_f, reaction_table.E_f_over_k
    )
    ratio = (Z_Ttr * Z_TF) / (Z_Tv * Z_U + _TINY)
    k_f_cvdv = k_f_base * ratio

    return jnp.where(is_dissoc[:, None], k_f_cvdv, k_f_arr)


def _compute_species_production_rates(
    molar_masses: Float[Array, " n_species"],
    net_stoich: Float[Array, "n_reactions n_species"],
    rates: Float[Array, "n_reactions n_cells"],
) -> Float[Array, "n_cells n_species"]:
    """Compute species mass production rates from reaction rates.

    Args:
        molar_masses: Species molar masses [kg/mol]. Shape [n_species].
        net_stoich: Net stoichiometry (products - reactants). Shape [n_reactions, n_species].
        rates: Reaction rates [mol/m^3/s]. Shape [n_reactions, n_cells].

    Returns:
        omega_dot: Mass production rates [kg/m^3/s]. Shape [n_cells, n_species].
    """
    n_dot = jnp.einsum("rs,rc->cs", net_stoich, rates)
    return n_dot * molar_masses[None, :]


def _compute_vibrational_source_cvdv_qp(
    omega_dot_f: Float[Array, "n_cells n_species"],
    omega_dot_b: Float[Array, "n_cells n_species"],
    T_tr: Float[Array, " n_cells"],
    T_v: Float[Array, " n_cells"],
    species_table: chemistry_types.SpeciesTable,
) -> Float[Array, " n_cells"]:
    """Compute CVDV-QP chemistry-vibration source term.

    Args:
        omega_dot_f: Forward-only species mass production [kg/m^3/s].
            Shape [n_cells, n_species].
        omega_dot_b: Backward-only species mass production [kg/m^3/s].
            Shape [n_cells, n_species].
        T_tr: Translational temperature [K]. Shape [n_cells].
        T_v: Vibrational temperature [K]. Shape [n_cells].
        species_table: Species data.

    Returns:
        Q_vib_chem: Vibrational reactive source [W/m^3]. Shape [n_cells].
    """
    M_s = species_table.molar_masses
    D_s = species_table.dissociation_energy
    theta_v = species_table.theta_vib
    is_molecule = jnp.isfinite(D_s)

    R_m = constants.R_universal / M_s
    U_m = D_s / (3.0 * R_m + _TINY)
    U_m = jnp.where(is_molecule, U_m, 1.0e30)

    n_species = M_s.shape[0]
    n_cells = T_tr.shape[0]
    T_tr_sc = jnp.broadcast_to(T_tr[None, :], (n_species, n_cells))
    T_v_sc = jnp.broadcast_to(T_v[None, :], (n_species, n_cells))

    inv_TF = (
        1.0 / jnp.clip(T_v_sc, 1e-12, None)
        - 1.0 / jnp.clip(T_tr_sc, 1e-12, None)
        - 1.0 / jnp.clip(U_m[:, None], 1e-12, None)
    )
    inv_TF = jnp.where(jnp.abs(inv_TF) < _TINY, jnp.sign(inv_TF) * _TINY, inv_TF)
    T_F = 1.0 / inv_TF

    N_m = _cvdv_level_counts(theta_v, D_s, M_s)
    E_TF = _cvdv_average_energy(T_F, theta_v, N_m)
    E_U = _cvdv_average_energy(-U_m[:, None], theta_v, N_m)

    m_particle = M_s / constants.N_A
    omega_f = omega_dot_f.T
    omega_b = omega_dot_b.T

    Q_m = (E_TF * omega_f + E_U * omega_b) / (m_particle[:, None] + _TINY)
    Q_m = jnp.where(is_molecule[:, None], Q_m, 0.0)
    return jnp.sum(Q_m, axis=0)


def build_cvdv_qp_chemistry_model() -> chemistry_types.ChemistryModel:
    """Build the CVDV-QP chemistry model (Casseau/Marrone-Treanor)."""

    def forward_rate_coefficient(
        T: Float[Array, " n_cells"],
        T_v: Float[Array, " n_cells"],
        species_table: chemistry_types.SpeciesTable,
        reaction_table: chemistry_types.ReactionTable,
    ) -> Float[Array, "n_reactions n_cells"]:
        return _compute_forward_rate_coefficient_cvdv_qp(
            T, T_v, species_table, reaction_table
        )

    def vibrational_reactive_source(
        rho_s: Float[Array, "n_cells n_species"],
        omega_dot_f: Float[Array, "n_cells n_species"],
        omega_dot_b: Float[Array, "n_cells n_species"],
        T: Float[Array, " n_cells"],
        T_v: Float[Array, " n_cells"],
        species_table: chemistry_types.SpeciesTable,
        reaction_table: chemistry_types.ReactionTable,
    ) -> Float[Array, " n_cells"]:
        del rho_s, reaction_table
        return _compute_vibrational_source_cvdv_qp(
            omega_dot_f, omega_dot_b, T, T_v, species_table
        )

    return ChemistryModel(
        forward_rate_coefficient=forward_rate_coefficient,
        vibrational_reactive_source=vibrational_reactive_source,
    )


def build_park_chemistry_model() -> chemistry_types.ChemistryModel:
    """Placeholder chemistry model for Park."""

    def _not_implemented_forward(*_args, **_kwargs) -> Float[Array, "n_reactions n_cells"]:
        raise NotImplementedError(
            "Park chemistry model is not implemented in src/compressible_1d/chemistry.py."
        )

    def _not_implemented_vibrational(*_args, **_kwargs) -> Float[Array, " n_cells"]:
        raise NotImplementedError(
            "Park chemistry model is not implemented in src/compressible_1d/chemistry.py."
        )

    return ChemistryModel(
        forward_rate_coefficient=_not_implemented_forward,
        vibrational_reactive_source=_not_implemented_vibrational,
    )


def compute_all_chemical_sources(
    rho_s: Float[Array, "n_cells n_species"],
    T: Float[Array, " n_cells"],
    T_v: Float[Array, " n_cells"],
    species_table: chemistry_types.SpeciesTable,
    reaction_table: chemistry_types.ReactionTable,
) -> tuple[
    Float[Array, "n_cells n_species"],
    Float[Array, " n_cells"],
]:
    """Compute chemical source terms and vibrational reactive source.

    Args:
        rho_s: Partial densities [kg/m^3]. Shape [n_cells, n_species].
        T: Translational temperature [K]. Shape [n_cells].
        T_v: Vibrational temperature [K]. Shape [n_cells].
        species_table: Species data.
        reaction_table: Reaction mechanism data.

    Returns:
        omega_dot: Species mass production rates [kg/m^3/s].
            Shape [n_cells, n_species].
        Q_vib_chem: Vibrational reactive source [W/m^3]. Shape [n_cells].
    """
    M_s = species_table.molar_masses  # [kg/mol]
    n_s = rho_s / (M_s[None, :] + _TINY)  # [mol/m^3]
    n_mix = jnp.sum(n_s, axis=1) * constants.N_A  # [1/m^3]

    log_n_s = jnp.log(jnp.clip(n_s, _TINY, None))
    log_prod_f = reaction_rate_products(log_n_s, reaction_table.reactant_stoich)
    log_prod_b = reaction_rate_products(log_n_s, reaction_table.product_stoich)

    k_f = reaction_table.chemistry_model.forward_rate_coefficient(
        T, T_v, species_table, reaction_table
    )
    is_eid = reaction_table.is_electron_impact > 0.5
    T_b = jnp.where(is_eid[:, None], T_v[None, :], T[None, :])
    K_eq = equilibrium_constant_casseau(
        T_b, n_mix, reaction_table.equilibrium_coeffs_casseau
    )
    # Casseau coefficients are typically fitted in kmol-based concentrations.
    # Convert K_eq to mol-based concentrations using (10^3)^{Δν}.
    delta_nu = jnp.sum(reaction_table.net_stoich, axis=1)
    K_eq = K_eq * jnp.power(1.0e3, delta_nu)[:, None]
    k_b = k_f / (K_eq + _TINY)

    R_f = k_f * jnp.exp(log_prod_f)  # [mol/m^3/s]
    R_b = k_b * jnp.exp(log_prod_b)  # [mol/m^3/s]

    omega_dot = _compute_species_production_rates(
        M_s, reaction_table.net_stoich, R_f - R_b
    )

    dissoc_mask = (reaction_table.is_dissociation > 0.5)[:, None]
    R_f_d = R_f * dissoc_mask
    R_b_d = R_b * dissoc_mask
    omega_dot_f = _compute_species_production_rates(
        M_s, reaction_table.net_stoich, R_f_d
    )
    omega_dot_b = _compute_species_production_rates(
        M_s, reaction_table.net_stoich, R_b_d
    )

    Q_vib_chem = reaction_table.chemistry_model.vibrational_reactive_source(
        rho_s, omega_dot_f, omega_dot_b, T, T_v, species_table, reaction_table
    )

    return omega_dot, Q_vib_chem
