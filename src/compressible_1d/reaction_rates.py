"""Reaction rate calculations for chemical kinetics.

Implements the chemical kinetic model from NASA TP-2867 (Gnoffo et al. 1989).

Key equations:
    - Eq. 41: Species mass production rate
    - Eq. 42-43: Forward and backward reaction rates
    - Eq. 45: Park dissociation temperature
    - Eq. 46a-b: Arrhenius rate coefficients
    - Eq. 47: Equilibrium constant polynomial
"""

import jax.numpy as jnp
from jaxtyping import Array, Float

from compressible_1d import chemistry_types


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
    # T_d = jnp.sqrt(T * T_v)  # [n_cells]

    T_d = T**0.7 * T_v**0.3

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

    Note: C_f is in cgs units (cm³/mol/s for bimolecular, cm⁶/mol²/s for termolecular).
    The unit conversion to mks is handled in compute_reaction_rates().

    Args:
        T_q: Rate-controlling temperature [K]. Shape [n_reactions, n_cells].
        C_f: Pre-exponential factor [cgs units]. Shape [n_reactions].
        n_f: Temperature exponent [-]. Shape [n_reactions].
        E_f_over_k: Activation energy / k [K]. Shape [n_reactions].

    Returns:
        k_f: Forward rate coefficient [cgs units]. Shape [n_reactions, n_cells].
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
) -> Float[Array, "n_reactions n_cells"]:
    """Compute equilibrium constant K_c using polynomial curve fit.

    Equation 47 from NASA TP-2867:
        K_{c,r} = exp(B_1 + B_2*ln(Z) + B_3*Z + B_4*Z² + B_5*Z³)
        where Z = 10000/T

    Note: K_c is dimensionless for exchange reactions, but has units for
    dissociation reactions. The units depend on the net change in moles.

    Args:
        T: Temperature [K]. Shape [n_cells].
        equilibrium_coeffs: Polynomial coefficients [B_1, B_2, B_3, B_4, B_5].
            Shape [n_reactions, 5].

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

    Equations 42-43 from NASA TP-2867:
        R_{f,r} = 1000 * k_{f,r} * ∏_s (0.001 * ρ_s/M_s)^{α_{s,r}}
        R_{b,r} = 1000 * k_{b,r} * ∏_s (0.001 * ρ_s/M_s)^{β_{s,r}}

    The factors 1000 and 0.001 handle cgs ↔ mks unit conversion:
        - ρ_s is in kg/m³ (mks)
        - M_s is in kg/mol (mks)
        - k_f, k_b are in cgs units (cm³/mol/s or cm⁶/mol²/s)
        - Concentrations 0.001*ρ_s/M_s convert to mol/cm³ (cgs)
        - Factor 1000 converts result back to mol/m³/s (mks)

    Note: Third-body (M) reactions are handled by including the collision partner
    explicitly in the stoichiometry. For example, "N2 + M -> 2N + M" with M=N
    becomes "N2 + N -> 3N" with reactants {"N2": 1, "N": 1} and products {"N": 3}.

    Args:
        rho_s: Partial densities [kg/m³]. Shape [n_cells, n_species].
        M_s: Molar masses [kg/mol]. Shape [n_species].
        k_f: Forward rate coefficients [cgs]. Shape [n_reactions, n_cells].
        k_b: Backward rate coefficients [cgs]. Shape [n_reactions, n_cells].
        reactant_stoich: Reactant stoichiometry α_{s,r}. Shape [n_reactions, n_species].
        product_stoich: Product stoichiometry β_{s,r}. Shape [n_reactions, n_species].

    Returns:
        R_f: Forward reaction rates [mol/m³/s]. Shape [n_reactions, n_cells].
        R_b: Backward reaction rates [mol/m³/s]. Shape [n_reactions, n_cells].
    """
    # Convert to molar concentrations in cgs units [mol/cm³]
    # c_s = 0.001 * rho_s / M_s
    c_s = 0.001 * rho_s / M_s[None, :]  # [n_cells, n_species]

    # Compute concentration products for forward reaction
    # ∏_s c_s^{α_{s,r}} for each reaction r
    # Using log-sum-exp for numerical stability: exp(Σ_s α_{s,r} * log(c_s))
    log_c_s = jnp.log(c_s + 1e-30)  # [n_cells, n_species]

    # Forward: ∏_s c_s^{α_{s,r}}
    # Sum over species: Σ_s α_{s,r} * log(c_s)
    log_prod_forward = jnp.einsum(
        "rs,cs->rc", reactant_stoich, log_c_s
    )  # [n_reactions, n_cells]
    prod_forward = jnp.exp(jnp.clip(log_prod_forward, -700, 700))

    # Backward: ∏_s c_s^{β_{s,r}}
    log_prod_backward = jnp.einsum(
        "rs,cs->rc", product_stoich, log_c_s
    )  # [n_reactions, n_cells]
    prod_backward = jnp.exp(jnp.clip(log_prod_backward, -700, 700))

    # Compute reaction rates (Eqs. 42-43)
    # R_f = 1000 * k_f * ∏_s c_s^{α_{s,r}}
    # R_b = 1000 * k_b * ∏_s c_s^{β_{s,r}}
    R_f = 1000.0 * k_f * prod_forward  # [n_reactions, n_cells]
    R_b = 1000.0 * k_b * prod_backward  # [n_reactions, n_cells]

    return R_f, R_b


def compute_species_production_rates(
    M_s: Float[Array, " n_species"],
    net_stoich: Float[Array, "n_reactions n_species"],
    R_f: Float[Array, "n_reactions n_cells"],
    R_b: Float[Array, "n_reactions n_cells"],
) -> Float[Array, "n_cells n_species"]:
    """Compute species mass production rates ω̇_s.

    Equation 41 from NASA TP-2867:
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

    # Molar production rate for each species
    # ṅ_s = Σ_r (β_{s,r} - α_{s,r}) * R_net_r
    # net_stoich: [n_reactions, n_species]
    # R_net: [n_reactions, n_cells]
    # Result: [n_cells, n_species]
    n_dot = jnp.einsum("rs,rc->cs", net_stoich, R_net)  # [n_cells, n_species]

    # Mass production rate
    # ω̇_s = M_s * ṅ_s
    omega_dot = M_s[None, :] * n_dot  # [n_cells, n_species]

    return omega_dot


def compute_chemical_energy_source(
    omega_dot: Float[Array, "n_cells n_species"],
    h_s0: Float[Array, " n_species"],
) -> Float[Array, " n_cells"]:
    """Compute chemical energy source term Q_chem.

    The chemical energy release/absorption:
        Q_chem = -Σ_s ω̇_s * h_{s,0}

    where h_{s,0} is the formation enthalpy at reference temperature.
    Negative sign: exothermic reactions (products have lower h_s0) release energy.

    Args:
        omega_dot: Mass production rates [kg/m³/s]. Shape [n_cells, n_species].
        h_s0: Formation enthalpy at T_ref [J/kg]. Shape [n_species].

    Returns:
        Q_chem: Chemical energy source [W/m³]. Shape [n_cells].
    """
    # Q_chem = -Σ_s ω̇_s * h_{s,0}
    Q_chem = -jnp.sum(omega_dot * h_s0[None, :], axis=1)
    return Q_chem


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
    D_hat_s = jnp.array([0.3 * 3.36e7, 0.3 * 3.36e7])[:, None]

    # Only molecules contribute (atoms have is_monoatomic = True)
    # Mask out atoms by setting their contribution to zero
    is_molecule = 1.0 - is_monoatomic  # [n_species]
    D_hat_s_masked = D_hat_s * is_molecule[:, None]  # [n_species, n_cells]

    # Q_vib_chem = Σ_s ω̇_s * D̂_s
    # omega_dot: [n_cells, n_species], D_hat_s: [n_species, n_cells]
    Q_vib_chem = jnp.sum(omega_dot * D_hat_s_masked.T + e_el_s.T, axis=1)  # [n_cells]

    return Q_vib_chem


def compute_all_chemical_sources(
    rho_s: Float[Array, "n_cells n_species"],
    T: Float[Array, " n_cells"],
    T_v: Float[Array, " n_cells"],
    species_table: chemistry_types.SpeciesTable,
    reaction_table: chemistry_types.ReactionTable,
) -> tuple[
    Float[Array, "n_cells n_species"],
    Float[Array, " n_cells"],
    Float[Array, " n_cells"],
]:
    """Compute all chemical source terms.

    This is the main entry point for chemical kinetics calculations.

    Args:
        rho_s: Partial densities [kg/m³]. Shape [n_cells, n_species].
        T: Translational temperature [K]. Shape [n_cells].
        T_v: Vibrational temperature [K]. Shape [n_cells].
        species_table: Species data including molar masses and formation enthalpies.
        reaction_table: Reaction mechanism data.

    Returns:
        omega_dot: Mass production rates [kg/m³/s]. Shape [n_cells, n_species].
        Q_chem: Chemical energy source [W/m³]. Shape [n_cells].
        Q_vib_chem: Vibrational reactive source [W/m³]. Shape [n_cells].
    """
    M_s = species_table.molar_masses
    h_s0 = species_table.h_s0

    # Compute rate-controlling temperature for each reaction
    T_q = compute_rate_controlling_temperature(
        T, T_v, reaction_table.is_dissociation, reaction_table.is_electron_impact
    )

    # Forward rate coefficients (Eq. 46a)
    k_f = compute_forward_rate_coefficient(
        T_q, reaction_table.C_f, reaction_table.n_f, reaction_table.E_f_over_k
    )

    # Equilibrium constants (Eq. 47) - always evaluated at T, not T_q
    K_c = compute_equilibrium_constant(T, reaction_table.equilibrium_coeffs)

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

    # Chemical energy source
    Q_chem = compute_chemical_energy_source(omega_dot, h_s0)

    # Vibrational reactive source (Eq. 51)
    e_v_s = species_table.energy_model.e_ve(T_v)  # [n_species, n_cells]
    e_el_s = species_table.energy_model.e_el(T_v)  # [n_species, n_cells]
    Q_vib_chem = compute_vibrational_reactive_source_casseau(
        omega_dot,
        e_v_s,
        e_el_s,
        species_table.is_monoatomic,
        reaction_table.preferential_factor,
    )

    return omega_dot, Q_chem, Q_vib_chem
