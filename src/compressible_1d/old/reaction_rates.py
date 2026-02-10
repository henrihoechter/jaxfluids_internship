"""Reaction rate calculations for chemical kinetics.

Implements the chemical kinetic model from NASA TP-2867 (Gnoffo et al. 1989).

Key equations:
    - Eq. 41: Species mass production rate
    - Eq. 42-43: Forward and backward reaction rates
    - Eq. 45: Park dissociation temperature
    - Eq. 46a-b: Arrhenius rate coefficients
    - Eq. 47: Equilibrium constant polynomial
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal
import numpy as np
import jax.numpy as jnp
from jaxtyping import Array, Float

from compressible_1d import chemistry_types, constants


ForwardRateFn = Callable[
    [
        Float[Array, " n_cells"],
        Float[Array, " n_cells"],
        chemistry_types.SpeciesTable,
        chemistry_types.ReactionTable,
    ],
    Float[Array, "n_reactions n_cells"],
]
VibrationalSourceFn = Callable[
    [
        Float[Array, "n_cells n_species"],
        Float[Array, "n_cells n_species"],
        Float[Array, "n_cells n_species"],
        Float[Array, " n_cells"],
        Float[Array, " n_cells"],
        chemistry_types.SpeciesTable,
        chemistry_types.ReactionTable,
    ],
    Float[Array, " n_cells"],
]


@dataclass(frozen=True, eq=False)
class ChemistryModel:
    """Container for chemical kinetics model callables."""

    forward_rate_coefficient: ForwardRateFn
    vibrational_reactive_source: VibrationalSourceFn


@dataclass(frozen=True)
class ChemistryModelConfig:
    """Configuration for selecting chemical kinetics models."""

    model: Literal["park", "cvdv_qp"] = "park"
    park_vibrational_source: Literal["energy", "qp_constant"] = "energy"
    qp_constant: float = 0.3


def compute_rate_controlling_temperature(
    T: Float[Array, " n_cells"],
    T_v: Float[Array, " n_cells"],
    is_dissociation: Float[Array, " n_reactions"],
    is_electron_impact: Float[Array, " n_reactions"],
) -> Float[Array, "n_reactions n_cells"]:
    """Compute rate-controlling temperature T_q for each reaction.

    Park model (ref. 32 in TP-2867):
        - Dissociation reactions: T_d = sqrt(T * T_v) (Eq. 45)
        - Electron impact ionization: T_v (T_e in 3-temp model)
        - All other reactions: T

    Args:
        T: Translational temperature [K]. Shape [n_cells].
        T_v: Vibrational temperature [K]. Shape [n_cells].
        is_dissociation: Boolean flags for dissociation reactions. Shape [n_reactions].
        is_electron_impact: Boolean flags for electron impact reactions. Shape [n_reactions].

    Returns:
        T_q: Rate-controlling temperature [K]. Shape [n_reactions, n_cells].
    """
    n_reactions = is_dissociation.shape[0]

    # Park dissociation temperature (Eq. 45)
    T_d = jnp.sqrt(T * T_v)  # [n_cells]

    # T_d = T**0.7 * T_v**0.3

    # Broadcast to [n_reactions, n_cells]
    T_broadcast = jnp.broadcast_to(T[None, :], (n_reactions, T.shape[0]))
    T_v_broadcast = jnp.broadcast_to(T_v[None, :], (n_reactions, T.shape[0]))
    T_d_broadcast = jnp.broadcast_to(T_d[None, :], (n_reactions, T.shape[0]))

    # Select temperature based on reaction type
    # Default: T (heavy particle translational temperature)
    T_q = T_broadcast

    # Dissociation reactions: use T_d = sqrt(T * T_v)
    T_q = jnp.where(is_dissociation[:, None] > 0.5, T_d_broadcast, T_q)

    # Electron impact reactions: use T_v
    T_q = jnp.where(is_electron_impact[:, None] > 0.5, T_v_broadcast, T_q)

    return T_q


def compute_forward_rate_coefficient(
    T_q: Float[Array, "n_reactions n_cells"],
    C_f: Float[Array, " n_reactions"],
    n_f: Float[Array, " n_reactions"],
    E_f_over_k: Float[Array, " n_reactions"],
) -> Float[Array, "n_reactions n_cells"]:
    """Compute forward rate coefficient k_f using Arrhenius form.

    Equation 46a from NASA TP-2867:
        k_{f,r} = C_{f,r} * T_q^{n_{f,r}} * exp(-E_{f,r} / (k * T_q))

    Note: C_f is in SI per-mol units (m³/mol/s for bimolecular,
    m⁶/mol²/s for termolecular).

    Args:
        T_q: Rate-controlling temperature [K]. Shape [n_reactions, n_cells].
        C_f: Pre-exponential factor [SI per-mol units]. Shape [n_reactions].
        n_f: Temperature exponent [-]. Shape [n_reactions].
        E_f_over_k: Activation energy / k [K]. Shape [n_reactions].

    Returns:
        k_f: Forward rate coefficient [SI per-mol units]. Shape [n_reactions, n_cells].
    """
    # Arrhenius form: k_f = C_f * T^n_f * exp(-E_f/kT)
    # Note: E_f_over_k is already E_f/k, so we just divide by T
    exponent = -E_f_over_k[:, None] / (T_q + 1e-30)

    # Clip exponent to avoid overflow
    exponent = jnp.clip(exponent, -700, 700)

    k_f = C_f[:, None] * jnp.power(T_q, n_f[:, None]) * jnp.exp(exponent)

    return k_f


def compute_equilibrium_constant(
    T: Float[Array, " n_cells"],
    equilibrium_coeffs: Float[Array, "n_reactions 5"],
    delta_nu: Float[Array, " n_reactions"] | None = None,
) -> Float[Array, "n_reactions n_cells"]:
    """Compute equilibrium constant K_c using polynomial curve fit.

    Equation 47 from NASA TP-2867:
        K_{c,r} = exp(B_1 + B_2*ln(Z) + B_3*Z + B_4*Z² + B_5*Z³)
        where Z = 10000/T

    Note: This polynomial returns K_c in mol/cm³ units (Gnoffo/NASA convention).
    If delta_nu is provided, the result is converted to mol/m³ units via
    K_c *= (1e6)^{delta_nu}.

    Args:
        T: Temperature [K]. Shape [n_cells].
        equilibrium_coeffs: Polynomial coefficients [B_1, B_2, B_3, B_4, B_5].
            Shape [n_reactions, 5].
        delta_nu: Net change in stoichiometric coefficients per reaction,
            sum(products) - sum(reactants). Shape [n_reactions].

    Returns:
        K_c: Equilibrium constant. Shape [n_reactions, n_cells].
    """
    # Z = 10000/T (Eq. 48)
    Z = 10000.0 / (T + 1e-30)  # [n_cells]

    # Extract coefficients
    B_1 = equilibrium_coeffs[:, 0]  # [n_reactions]
    B_2 = equilibrium_coeffs[:, 1]
    B_3 = equilibrium_coeffs[:, 2]
    B_4 = equilibrium_coeffs[:, 3]
    B_5 = equilibrium_coeffs[:, 4]

    # Compute polynomial argument
    # K_c = exp(B_1 + B_2*ln(Z) + B_3*Z + B_4*Z² + B_5*Z³)
    ln_Z = jnp.log(Z + 1e-30)  # [n_cells]

    # Broadcast for [n_reactions, n_cells]
    exponent = (
        B_1[:, None]
        + B_2[:, None] * ln_Z[None, :]
        + B_3[:, None] * Z[None, :]
        + B_4[:, None] * Z[None, :] ** 2
        + B_5[:, None] * Z[None, :] ** 3
    )

    # Clip to avoid overflow
    exponent = jnp.clip(exponent, -700, 700)

    K_c = jnp.exp(exponent)

    return K_c


def compute_equilibrium_constant_casseau(
    T: Float[Array, " n_cells"],
    c_total_m3: Float[Array, " n_cells"],
    coeff_table: Float[Array, "n_refs 6"],
    delta_nu: float | None = None,
) -> Float[Array, " n_cells"]:
    """Compute equilibrium constant using Casseau eq. 2.69 with density-dependent coefficients.

    coeff_table columns: [n_ref_m3, A0, A1, A2, A3, A4].
    Coefficients are tabulated versus mixture number density in cm^-3; the loader
    converts n_ref to mol/m^3 for interpolation. The resulting K_eq is in mol/cm^3
    units (same convention as NASA TP-2867). If delta_nu is provided, convert to
    mol/m^3 units via (1e6) ** delta_nu.
    """
    n_ref = coeff_table[:, 0]
    coeffs = coeff_table[:, 1:]
    n_clamped = jnp.clip(c_total_m3, n_ref.min(), n_ref.max())

    A_vals = []
    for i in range(coeffs.shape[1]):
        A_vals.append(jnp.interp(n_clamped, n_ref, coeffs[:, i]))
    A0, A1, A2, A3, A4 = A_vals

    Z = 10000.0 / (T + 1e-30)
    exponent = (
        A0 * (T / 10000.0)
        + A1
        + A2 * jnp.log(Z + 1e-30)
        + A3 * Z
        + A4 * Z**2
    )
    exponent = jnp.clip(exponent, -700, 700)
    K_eq = jnp.exp(exponent)
    # if delta_nu is not None:
    #     K_eq = K_eq * jnp.power(1e6, delta_nu)

    return K_eq


def compute_equilibrium_constant_casseau_from_table(
    T: Float[Array, " n_cells"],
    c_total_m3: Float[Array, " n_cells"],
    coeff_tables: Float[Array, "n_reactions n_refs 6"],
    delta_nu: Float[Array, " n_reactions"],
) -> Float[Array, "n_reactions n_cells"]:
    """Compute Casseau equilibrium constants for multiple reactions."""
    K_list = []
    for r in range(coeff_tables.shape[0]):
        K_list.append(
            compute_equilibrium_constant_casseau(
                T,
                c_total_m3,
                coeff_tables[r],
                delta_nu[r],
            )
        )
    return jnp.stack(K_list, axis=0)


def compute_backward_rate_coefficient(
    k_f: Float[Array, "n_reactions n_cells"],
    K_c: Float[Array, "n_reactions n_cells"],
) -> Float[Array, "n_reactions n_cells"]:
    """Compute backward rate coefficient from forward rate and equilibrium constant.

    Equation 46b from NASA TP-2867:
        k_{b,r} = k_{f,r}(T) / K_{c,r}(T)

    Args:
        k_f: Forward rate coefficient. Shape [n_reactions, n_cells].
        K_c: Equilibrium constant. Shape [n_reactions, n_cells].

    Returns:
        k_b: Backward rate coefficient. Shape [n_reactions, n_cells].
    """
    # Avoid division by zero
    k_b = k_f / (K_c + 1e-30)
    return k_b


def compute_reaction_rates(
    rho_s: Float[Array, "n_cells n_species"],
    M_s: Float[Array, " n_species"],
    k_f: Float[Array, "n_reactions n_cells"],
    k_b: Float[Array, "n_reactions n_cells"],
    reactant_stoich: Float[Array, "n_reactions n_species"],
    product_stoich: Float[Array, "n_reactions n_species"],
) -> tuple[Float[Array, "n_reactions n_cells"], Float[Array, "n_reactions n_cells"]]:
    """Compute forward and backward reaction rates R_f and R_b.

    Rate law using molar concentration:
        R_{f,r} = k_{f,r} * ∏_s C_s^{α_{s,r}}
        R_{b,r} = k_{b,r} * ∏_s C_s^{β_{s,r}}

    where C_s = ρ_s / M_s is the molar concentration [mol/m³].

    Note: Third-body (M) reactions are handled by including the collision partner
    explicitly in the stoichiometry. For example, "N2 + M -> 2N + M" with M=N
    becomes "N2 + N -> 3N" with reactants {"N2": 1, "N": 1} and products {"N": 3}.

    Args:
        rho_s: Partial densities [kg/m³]. Shape [n_cells, n_species].
        M_s: Molar masses [kg/mol]. Shape [n_species].
        k_f: Forward rate coefficients [SI per-mol]. Shape [n_reactions, n_cells].
        k_b: Backward rate coefficients [SI per-mol]. Shape [n_reactions, n_cells].
        reactant_stoich: Reactant stoichiometry α_{s,r}. Shape [n_reactions, n_species].
        product_stoich: Product stoichiometry β_{s,r}. Shape [n_reactions, n_species].

    Returns:
        R_f: Forward reaction rates [mol/m³/s]. Shape [n_reactions, n_cells].
        R_b: Backward reaction rates [mol/m³/s]. Shape [n_reactions, n_cells].
    """
    # Convert to molar concentrations [mol/m³]
    C_s = rho_s / M_s[None, :]  # [n_cells, n_species]

    # Compute concentration products for forward reaction
    # ∏_s n_s^{α_{s,r}} for each reaction r
    # Using log-sum-exp for numerical stability: exp(Σ_s α_{s,r} * log(n_s))
    log_n_s = jnp.log(C_s + 1e-30)  # [n_cells, n_species]

    # Forward: ∏_s n_s^{α_{s,r}}
    # Sum over species: Σ_s α_{s,r} * log(n_s)
    log_prod_forward = jnp.einsum(
        "rs,cs->rc", reactant_stoich, log_n_s
    )  # [n_reactions, n_cells]

    # Backward: ∏_s n_s^{β_{s,r}}
    log_prod_backward = jnp.einsum(
        "rs,cs->rc", product_stoich, log_n_s
    )  # [n_reactions, n_cells]

    # Compute reaction rates in log space to avoid overflow for large number densities.
    tiny = jnp.finfo(k_f.dtype).tiny
    log_k_f = jnp.log(k_f + tiny)
    log_k_b = jnp.log(k_b + tiny)
    log_R_f = log_k_f + log_prod_forward
    log_R_b = log_k_b + log_prod_backward

    exp_min = jnp.log(jnp.finfo(k_f.dtype).tiny)
    exp_max = jnp.log(jnp.finfo(k_f.dtype).max)
    log_R_f = jnp.clip(log_R_f, exp_min, exp_max-exp_min)
    log_R_b = jnp.clip(log_R_b, exp_min, exp_max-exp_min)

    # R_f = k_f * ∏_s n_s^{α_{s,r}}
    # R_b = k_b * ∏_s n_s^{β_{s,r}}
    R_f = jnp.exp(log_R_f)  # [n_reactions, n_cells]
    R_b = jnp.exp(log_R_b)  # [n_reactions, n_cells]

    return R_f, R_b


def compute_species_production_rates(
    M_s: Float[Array, " n_species"],
    net_stoich: Float[Array, "n_reactions n_species"],
    R_f: Float[Array, "n_reactions n_cells"],
    R_b: Float[Array, "n_reactions n_cells"],
) -> Float[Array, "n_cells n_species"]:
    """Compute species mass production rates ω̇_s.

    Equation 41 from NASA TP-2867 (molar concentration form):
        ω̇_s = M_s * Σ_r (β_{s,r} - α_{s,r}) * (R_{f,r} - R_{b,r})

    Args:
        M_s: Molar masses [kg/mol]. Shape [n_species].
        net_stoich: Net stoichiometry (β - α). Shape [n_reactions, n_species].
        R_f: Forward reaction rates [mol/m³/s]. Shape [n_reactions, n_cells].
        R_b: Backward reaction rates [mol/m³/s]. Shape [n_reactions, n_cells].

    Returns:
        omega_dot: Mass production rates [kg/m³/s]. Shape [n_cells, n_species].
    """
    # Net reaction rate for each reaction
    R_net = R_f - R_b  # [n_reactions, n_cells]

    # Particle production rate for each species
    # ṅ_s = Σ_r (β_{s,r} - α_{s,r}) * R_net_r
    # net_stoich: [n_reactions, n_species]
    # R_net: [n_reactions, n_cells]
    # Result: [n_cells, n_species]
    n_dot = jnp.einsum("rs,rc->cs", net_stoich, R_net)  # [n_cells, n_species]

    # Mass production rate (convert mol to kg)
    omega_dot = M_s[None, :] * n_dot  # [n_cells, n_species]

    return omega_dot


def compute_species_production_rates_from_rates(
    M_s: Float[Array, " n_species"],
    net_stoich: Float[Array, "n_reactions n_species"],
    rates: Float[Array, "n_reactions n_cells"],
) -> Float[Array, "n_cells n_species"]:
    """Compute species mass production rates from a single reaction-rate array."""
    n_dot = jnp.einsum("rs,rc->cs", net_stoich, rates)  # [n_cells, n_species]
    return M_s[None, :] * n_dot


def compute_vibrational_reactive_source(
    omega_dot: Float[Array, "n_cells n_species"],
    e_v_s: Float[Array, "n_species n_cells"],
    e_el_s: Float[Array, "n_species n_cells"],
    is_monoatomic: Float[Array, " n_species"],
    preferential_factor: float,
) -> Float[Array, " n_cells"]:
    """Compute vibrational energy reactive source term Q_vib_chem.

    Equation 51 from NASA TP-2867 (simplified form):
        Q_vib_chem = Σ_{s=mol} ω̇_s * D̂_s
        D̂_s = ĉ_2 * e_{v,s}(T_v)

    where:
        - ĉ_2 = preferential_factor (1.0 for nonpreferential dissociation)
        - e_{v,s}(T_v) is the vibrational energy at vibrational temperature
        - Sum is only over molecular species (atoms have no vibrational modes)

    Physical interpretation:
        - When molecules dissociate (ω̇_s < 0), they carry vibrational energy away
        - When atoms recombine (ω̇_s > 0), new molecules gain vibrational energy
        - With preferential dissociation (ĉ_2 > 1), more vibrational energy is removed

    Args:
        omega_dot: Mass production rates [kg/m³/s]. Shape [n_cells, n_species].
        e_v_s: Vibrational energy per species at T_v [J/kg]. Shape [n_species, n_cells].
        is_monoatomic: Boolean mask for atoms. Shape [n_species].
        preferential_factor: Factor ĉ_2 (1.0 = nonpreferential).

    Returns:
        Q_vib_chem: Vibrational energy reactive source [W/m³]. Shape [n_cells].
    """
    # D̂_s = ĉ_2 * e_v_s(T_v)
    D_hat_s = preferential_factor * e_v_s  # [n_species, n_cells]

    # Only molecules contribute (atoms have is_monoatomic = True)
    # Mask out atoms by setting their contribution to zero
    is_molecule = 1.0 - is_monoatomic  # [n_species]
    D_hat_s_masked = D_hat_s * is_molecule[:, None]  # [n_species, n_cells]

    # Q_vib_chem = Σ_s ω̇_s * D̂_s
    # omega_dot: [n_cells, n_species], D_hat_s: [n_species, n_cells]
    Q_vib_chem = jnp.sum(omega_dot * D_hat_s_masked.T, axis=1)  # [n_cells]

    return Q_vib_chem


def compute_vibrational_reactive_source_casseau(
    omega_dot: Float[Array, "n_cells n_species"],
    e_v_s: Float[Array, "n_species n_cells"],
    e_el_s: Float[Array, "n_species n_cells"],
    is_monoatomic: Float[Array, " n_species"],
    preferential_factor: float,
) -> Float[Array, " n_cells"]:
    """Compute vibrational energy reactive source term Q_vib_chem.

    Equation 51 from NASA TP-2867 (simplified form):
        Q_vib_chem = Σ_{s=mol} ω̇_s * D̂_s
        D̂_s = ĉ_2 * e_{v,s}(T_v)

    where:
        - ĉ_2 = preferential_factor (1.0 for nonpreferential dissociation)
        - e_{v,s}(T_v) is the vibrational energy at vibrational temperature
        - Sum is only over molecular species (atoms have no vibrational modes)

    Physical interpretation:
        - When molecules dissociate (ω̇_s < 0), they carry vibrational energy away
        - When atoms recombine (ω̇_s > 0), new molecules gain vibrational energy
        - With preferential dissociation (ĉ_2 > 1), more vibrational energy is removed

    Args:
        omega_dot: Mass production rates [kg/m³/s]. Shape [n_cells, n_species].
        e_v_s: Vibrational energy per species at T_v [J/kg]. Shape [n_species, n_cells].
        is_monoatomic: Boolean mask for atoms. Shape [n_species].
        preferential_factor: Factor ĉ_2 (1.0 = nonpreferential).

    Returns:
        Q_vib_chem: Vibrational energy reactive source [W/m³]. Shape [n_cells].
    """
    # D̂_s = ĉ_2 * e_v_s(T_v)
    # D_hat_s = preferential_factor * e_v_s  # [n_species, n_cells]
    D_hat_s = jnp.array([0.3 * 3.36e7, 0.0])[:, None]
    # D_hat_s = 1.0 * e_v_s

    # Only molecules contribute (atoms have is_monoatomic = True)
    # Mask out atoms by setting their contribution to zero
    is_molecule = 1.0 - is_monoatomic  # [n_species]
    D_hat_s_masked = D_hat_s * is_molecule[:, None]  # [n_species, n_cells]

    # Q_vib_chem = Σ_s ω̇_s * D̂_s
    # omega_dot: [n_cells, n_species], D_hat_s: [n_species, n_cells]
    Q_vib_chem = jnp.sum(omega_dot * D_hat_s_masked.T + e_el_s.T, axis=1)  # [n_cells]

    return Q_vib_chem


def _cvdv_partition_function(
    temperature: Float[Array, "n_species ..."],
    level_factor: Float[Array, "n_species n_levels"],
    level_mask: Float[Array, "n_species n_levels"],
) -> Float[Array, "n_species ..."]:
    """Compute vibrational partition function with level truncation."""
    if temperature.ndim == 1:
        if temperature.shape[0] == level_factor.shape[0]:
            temperature = temperature[:, None]
        else:
            temperature = jnp.broadcast_to(
                temperature[None, :],
                (level_factor.shape[0], temperature.shape[0]),
            )

    exp_arg = -level_factor[:, :, None] / (temperature[:, None, :] + 1e-30)
    exp_arg = jnp.where(level_mask[:, :, None], exp_arg, -jnp.inf)
    return jnp.sum(jnp.exp(exp_arg), axis=1)


def _cvdv_partition_function_negative(
    temperature: Float[Array, "n_species ..."],
    level_factor: Float[Array, "n_species n_levels"],
    level_mask: Float[Array, "n_species n_levels"],
) -> Float[Array, "n_species ..."]:
    """Compute partition function for the negative-temperature form Z(-U_m)."""
    if temperature.ndim == 1:
        if temperature.shape[0] == level_factor.shape[0]:
            temperature = temperature[:, None]
        else:
            temperature = jnp.broadcast_to(
                temperature[None, :],
                (level_factor.shape[0], temperature.shape[0]),
            )

    exp_arg = level_factor[:, :, None] / (temperature[:, None, :] + 1e-30)
    exp_arg = jnp.where(level_mask[:, :, None], exp_arg, -jnp.inf)
    exp_arg = jnp.clip(exp_arg, -700.0, 700.0)
    return jnp.sum(jnp.exp(exp_arg), axis=1)


def _cvdv_average_energy(
    temperature: Float[Array, "n_species ..."],
    level_factor: Float[Array, "n_species n_levels"],
    energy_levels: Float[Array, "n_species n_levels"],
    level_mask: Float[Array, "n_species n_levels"],
) -> Float[Array, "n_species ..."]:
    """Compute average vibrational energy per particle for the CVDV model."""
    if temperature.ndim == 1:
        if temperature.shape[0] == level_factor.shape[0]:
            temperature = temperature[:, None]
        else:
            temperature = jnp.broadcast_to(
                temperature[None, :],
                (level_factor.shape[0], temperature.shape[0]),
            )

    exp_arg = -level_factor[:, :, None] / (temperature[:, None, :] + 1e-30)
    exp_arg = jnp.where(level_mask[:, :, None], exp_arg, -jnp.inf)
    weight = jnp.exp(exp_arg)
    partition = jnp.sum(weight, axis=1)
    numerator = jnp.sum(energy_levels[:, :, None] * weight, axis=1)
    return numerator / (partition + 1e-30)


def _cvdv_average_energy_negative(
    temperature: Float[Array, "n_species ..."],
    level_factor: Float[Array, "n_species n_levels"],
    energy_levels: Float[Array, "n_species n_levels"],
    level_mask: Float[Array, "n_species n_levels"],
) -> Float[Array, "n_species ..."]:
    """Compute average energy for the negative-temperature form."""
    if temperature.ndim == 1:
        if temperature.shape[0] == level_factor.shape[0]:
            temperature = temperature[:, None]
        else:
            temperature = jnp.broadcast_to(
                temperature[None, :],
                (level_factor.shape[0], temperature.shape[0]),
            )

    exp_arg = level_factor[:, :, None] / (temperature[:, None, :] + 1e-30)
    exp_arg = jnp.where(level_mask[:, :, None], exp_arg, -jnp.inf)
    exp_arg = jnp.clip(exp_arg, -700.0, 700.0)
    weight = jnp.exp(exp_arg)
    partition = jnp.sum(weight, axis=1)
    numerator = jnp.sum(energy_levels[:, :, None] * weight, axis=1)
    return numerator / (partition + 1e-30)

def _cvdv_dissociation_species_indices(
    net_stoich: Float[Array, "n_reactions n_species"],
    is_dissociation: Float[Array, " n_reactions"],
    is_monoatomic: Float[Array, " n_species"],
) -> jnp.ndarray:
    net_stoich_np = np.asarray(net_stoich)
    is_dissociation_np = np.asarray(is_dissociation) > 0.5
    is_monoatomic_np = np.asarray(is_monoatomic) > 0.5

    indices: list[int] = []
    for reaction in range(net_stoich_np.shape[0]):
        if not is_dissociation_np[reaction]:
            indices.append(0)
            continue
        candidates = np.where((net_stoich_np[reaction] < 0.0) & (~is_monoatomic_np))[0]
        if candidates.size != 1:
            raise ValueError(
                "CVDV model expects exactly one dissociating molecule per reaction."
            )
        indices.append(int(candidates[0]))

    return jnp.array(indices)


def build_park_chemistry_model(
    config: ChemistryModelConfig | None = None,
) -> ChemistryModel:
    """Build the Park chemistry model (Arrhenius + Park vibrational source)."""
    if config is None:
        config = ChemistryModelConfig()

    def forward_rate_coefficient(
        T: Float[Array, " n_cells"],
        T_v: Float[Array, " n_cells"],
        species_table: chemistry_types.SpeciesTable,
        reaction_table: chemistry_types.ReactionTable,
    ) -> Float[Array, "n_reactions n_cells"]:
        T_q = compute_rate_controlling_temperature(
            T, T_v, reaction_table.is_dissociation, reaction_table.is_electron_impact
        )
        return compute_forward_rate_coefficient(
            T_q, reaction_table.C_f, reaction_table.n_f, reaction_table.E_f_over_k
        )

    def vibrational_reactive_source(
        omega_dot: Float[Array, "n_cells n_species"],
        omega_dot_f: Float[Array, "n_cells n_species"],
        omega_dot_b: Float[Array, "n_cells n_species"],
        T: Float[Array, " n_cells"],
        T_v: Float[Array, " n_cells"],
        species_table: chemistry_types.SpeciesTable,
        reaction_table: chemistry_types.ReactionTable,
    ) -> Float[Array, " n_cells"]:
        if config.park_vibrational_source == "energy":
            e_ve = species_table.energy_model.e_ve(T_v)
            D0 = reaction_table.preferential_factor * e_ve
            return jnp.sum(omega_dot * D0.T, axis=1)

        e_el = species_table.energy_model.e_el(T_v)
        dissociation_energy = jnp.where(
            jnp.isfinite(species_table.dissociation_energy),
            species_table.dissociation_energy,
            0.0,
        )
        if config.park_vibrational_source == "qp_constant":
            alpha = config.qp_constant
            D0 = (alpha * dissociation_energy)[:, None]
            return jnp.sum(omega_dot * D0.T, axis=1)

        raise ValueError(
            f"Unknown park_vibrational_source '{config.park_vibrational_source}'."
        )

    return ChemistryModel(
        forward_rate_coefficient=forward_rate_coefficient,
        vibrational_reactive_source=vibrational_reactive_source,
    )


def build_cvdv_qp_chemistry_model(
    species_table: chemistry_types.SpeciesTable,
    net_stoich: Float[Array, "n_reactions n_species"],
    is_dissociation: Float[Array, " n_reactions"],
) -> ChemistryModel:
    """Build the CVDV-Qp chemistry model from Casseau (Marrone-Treanor)."""
    theta_vib = np.asarray(species_table.theta_vib)
    dissociation_energy = np.asarray(species_table.dissociation_energy)
    molar_masses = np.asarray(species_table.molar_masses)
    R_specific = constants.R_universal / molar_masses
    is_monoatomic = np.asarray(species_table.is_monoatomic) > 0.5

    missing_theta = (~is_monoatomic) & (~np.isfinite(theta_vib) | (theta_vib <= 0.0))
    if np.any(missing_theta):
        names = np.asarray(species_table.names)[missing_theta]
        raise ValueError(
            "CVDV model requires theta_vib for molecular species. "
            f"Missing for: {tuple(names)}"
        )

    theta_vib = np.nan_to_num(theta_vib, nan=0.0)
    dissociation_energy = np.nan_to_num(dissociation_energy, nan=0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        n_levels = np.where(
            theta_vib > 0.0,
            np.floor(dissociation_energy / (R_specific * theta_vib)),
            0,
        ).astype(int)
    n_levels = np.maximum(n_levels, 0)

    max_levels = int(n_levels.max()) if n_levels.size else 0
    levels = jnp.arange(max_levels + 1)
    level_mask = jnp.array(levels[None, :] <= n_levels[:, None])
    level_factor = jnp.array(theta_vib)[:, None] * levels[None, :]
    energy_levels = level_factor * constants.k

    dissociation_indices = _cvdv_dissociation_species_indices(
        net_stoich,
        is_dissociation,
        species_table.is_monoatomic,
    )

    U_m = jnp.array(dissociation_energy / (3.0 * R_specific))
    U_m_safe = jnp.where(U_m > 0.0, U_m, 1.0)

    Z_U = _cvdv_partition_function_negative(U_m_safe, level_factor, level_mask)[:, 0]
    E_U = _cvdv_average_energy_negative(
        U_m_safe, level_factor, energy_levels, level_mask
    )[:, 0]

    def forward_rate_coefficient(
        T: Float[Array, " n_cells"],
        T_v: Float[Array, " n_cells"],
        species_table: chemistry_types.SpeciesTable,
        reaction_table: chemistry_types.ReactionTable,
    ) -> Float[Array, "n_reactions n_cells"]:
        T_q = compute_rate_controlling_temperature(
            T, T_v, reaction_table.is_dissociation, reaction_table.is_electron_impact
        )
        k_f = compute_forward_rate_coefficient(
            T_q, reaction_table.C_f, reaction_table.n_f, reaction_table.E_f_over_k
        )

        inv_TF = (
            1.0 / (T_v[None, :] + 1e-30)
            - 1.0 / (T[None, :] + 1e-30)
            - 1.0 / (U_m_safe[:, None] + 1e-30)
        )
        inv_TF = jnp.clip(inv_TF, a_min=1e-30)
        T_F = 1.0 / inv_TF

        T_tr_species = jnp.broadcast_to(T[None, :], (level_factor.shape[0], T.shape[0]))
        T_v_species = jnp.broadcast_to(
            T_v[None, :], (level_factor.shape[0], T_v.shape[0])
        )
        Z_Ttr = _cvdv_partition_function(T_tr_species, level_factor, level_mask)
        Z_Tv = _cvdv_partition_function(T_v_species, level_factor, level_mask)
        Z_TF = _cvdv_partition_function(T_F, level_factor, level_mask)

        idx = dissociation_indices
        Z_Ttr_r = Z_Ttr[idx]
        Z_Tv_r = Z_Tv[idx]
        Z_TF_r = Z_TF[idx]
        Z_U_r = Z_U[idx][:, None]

        k_f_cvdv = (
            reaction_table.C_f[:, None]
            * jnp.power(T[None, :], reaction_table.n_f[:, None])
            * jnp.exp(-reaction_table.E_f_over_k[:, None] / (T[None, :] + 1e-30))
        )
        ratio = (Z_Ttr_r * Z_TF_r) / (Z_Tv_r * Z_U_r + 1e-30)
        k_f = jnp.where(
            reaction_table.is_dissociation[:, None] > 0.5,
            k_f_cvdv * ratio,
            k_f,
        )

        return k_f

    def vibrational_reactive_source(
        omega_dot: Float[Array, "n_cells n_species"],
        omega_dot_f: Float[Array, "n_cells n_species"],
        omega_dot_b: Float[Array, "n_cells n_species"],
        T: Float[Array, " n_cells"],
        T_v: Float[Array, " n_cells"],
        species_table: chemistry_types.SpeciesTable,
        reaction_table: chemistry_types.ReactionTable,
    ) -> Float[Array, " n_cells"]:
        inv_TF = (
            1.0 / (T_v[None, :] + 1e-30)
            - 1.0 / (T[None, :] + 1e-30)
            - 1.0 / (U_m_safe[:, None] + 1e-30)
        )
        inv_TF = jnp.clip(inv_TF, a_min=1e-30)
        T_F = 1.0 / inv_TF

        E_Tv = _cvdv_average_energy(T_F, level_factor, energy_levels, level_mask)
        E_Tr = E_U[:, None]

        mass_per_particle = species_table.molar_masses / constants.N_A
        term_f = omega_dot_f * (E_Tv.T / mass_per_particle[None, :])
        term_b = omega_dot_b * (E_Tr.T / mass_per_particle[None, :])
        return jnp.sum(term_f + term_b, axis=1)

    return ChemistryModel(
        forward_rate_coefficient=forward_rate_coefficient,
        vibrational_reactive_source=vibrational_reactive_source,
    )


def build_chemistry_model_from_config(
    config: ChemistryModelConfig,
    *,
    species_table: chemistry_types.SpeciesTable,
    net_stoich: Float[Array, "n_reactions n_species"],
    is_dissociation: Float[Array, " n_reactions"],
) -> ChemistryModel:
    """Build a chemistry model from a configuration object."""
    model = config.model.lower()
    if model == "park":
        return build_park_chemistry_model(config)
    if model == "cvdv_qp":
        return build_cvdv_qp_chemistry_model(
            species_table, net_stoich=net_stoich, is_dissociation=is_dissociation
        )

    raise ValueError(f"Unknown chemistry model '{config.model}'.")


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
    """Compute all chemical source terms.

    This is the main entry point for chemical kinetics calculations.
    Uses the chemistry model stored in the reaction table when available.

    Args:
        rho_s: Partial densities [kg/m³]. Shape [n_cells, n_species].
        T: Translational temperature [K]. Shape [n_cells].
        T_v: Vibrational temperature [K]. Shape [n_cells].
        species_table: Species data including molar masses and formation enthalpies.
        reaction_table: Reaction mechanism data.

    Returns:
        omega_dot: Mass production rates [kg/m³/s]. Shape [n_cells, n_species].
        Q_vib_chem: Vibrational reactive source [W/m³]. Shape [n_cells].
    """
    M_s = species_table.molar_masses
    chemistry_model = reaction_table.chemistry_model

    # Forward rate coefficients
    k_f = chemistry_model.forward_rate_coefficient(
        T, T_v, species_table, reaction_table
    )

    # Equilibrium constants (evaluated at translational temperature T)
    c_total_m3 = jnp.sum(rho_s / (M_s[None, :] / constants.N_A), axis=1)  # 1/m3
    delta_nu = jnp.sum(reaction_table.net_stoich, axis=1)

    K_c = compute_equilibrium_constant_casseau_from_table(
        T, c_total_m3, reaction_table.equilibrium_coeffs_casseau, delta_nu
    )

    # Backward rate coefficients (Eq. 46b)
    k_b = compute_backward_rate_coefficient(k_f, K_c)

    # Reaction rates (Eqs. 42-43)
    R_f, R_b = compute_reaction_rates(
        rho_s,
        M_s,
        k_f,
        k_b,
        reaction_table.reactant_stoich,
        reaction_table.product_stoich,
    )

    # Species production rates (Eq. 41)
    omega_dot = compute_species_production_rates(
        M_s, reaction_table.net_stoich, R_f, R_b
    )

    omega_dot_f = compute_species_production_rates_from_rates(
        M_s, reaction_table.net_stoich, R_f
    )
    omega_dot_b = compute_species_production_rates_from_rates(
        M_s, reaction_table.net_stoich, R_b
    )

    Q_vib_chem = chemistry_model.vibrational_reactive_source(
        omega_dot,
        omega_dot_f,
        omega_dot_b,
        T,
        T_v,
        species_table,
        reaction_table,
    )

    return omega_dot, Q_vib_chem
