"""Viscous flux computation for two-temperature Navier-Stokes-Fourier equations.

This module implements viscous flux calculations following NASA TP-2867.
The viscous flux includes:
- Species diffusion flux
- Viscous stress (shear)
- Heat conduction (translational + rotational + vibrational)
- Vibrational energy diffusion

For the 1D case with state vector U = [ρ_s, ρu, ρE, ρE_v]:
F_viscous = [
    ρ·D_s·∂c_s/∂x,                                              # Species diffusion
    -τ,                                                          # Viscous stress
    -τ·u + (η_t+η_r)·∂T/∂x + η_v·∂T_v/∂x + Σ h_s·j_s,           # Total energy
    η_v·∂T_v/∂x + Σ h_{v,s}·j_s                                  # Vibrational energy
]
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jaxtyping import Array, Float

from compressible_1d import equation_manager_utils
from compressible_core import transport_models
from compressible_core import thermodynamic_relations

if TYPE_CHECKING:
    from compressible_1d.equation_manager_types import EquationManager


def compute_gradients_at_interfaces(
    phi: Float[Array, "n_cells_with_halo"],
    dx: float,
) -> Float[Array, "n_cells_with_halo-1"]:
    """Compute gradients at cell interfaces using central differences.

    Args:
        phi: Field values at cell centers [n_cells_with_halo]
        dx: Grid spacing [m]

    Returns:
        dphi_dx: Gradient at interfaces [n_cells_with_halo - 1]
    """
    # Central difference: dphi/dx at interface i+1/2 = (phi_{i+1} - phi_i) / dx
    dphi_dx = (phi[1:] - phi[:-1]) / dx

    return dphi_dx


def compute_gradients_at_interfaces_multispecies(
    phi: Float[Array, "n_cells_with_halo n_species"],
    dx: float,
) -> Float[Array, "n_cells_with_halo-1 n_species"]:
    """Compute gradients at cell interfaces for multi-species field.

    Args:
        phi: Field values at cell centers [n_cells_with_halo, n_species]
        dx: Grid spacing [m]

    Returns:
        dphi_dx: Gradient at interfaces [n_cells_with_halo - 1, n_species]
    """
    dphi_dx = (phi[1:, :] - phi[:-1, :]) / dx

    return dphi_dx


def compute_interface_values(
    phi: Float[Array, "n_cells_with_halo"],
) -> Float[Array, "n_cells_with_halo-1"]:
    """Compute values at cell interfaces by simple averaging.

    Args:
        phi: Field values at cell centers [n_cells_with_halo]

    Returns:
        phi_face: Values at interfaces [n_cells_with_halo - 1]
    """
    return 0.5 * (phi[:-1] + phi[1:])


def compute_interface_values_multispecies(
    phi: Float[Array, "n_cells_with_halo n_species"],
) -> Float[Array, "n_cells_with_halo-1 n_species"]:
    """Compute values at cell interfaces for multi-species field.

    Args:
        phi: Field values at cell centers [n_cells_with_halo, n_species]

    Returns:
        phi_face: Values at interfaces [n_cells_with_halo - 1, n_species]
    """
    return 0.5 * (phi[:-1, :] + phi[1:, :])


def compute_viscous_stress_1d(
    du_dx: Float[Array, " n_interfaces"],
    mu: Float[Array, " n_interfaces"],
) -> Float[Array, " n_interfaces"]:
    """Compute viscous stress in 1D.

    For 1D flow:
        τ = (4/3) μ ∂u/∂x

    This comes from the general stress tensor where τ_xx = (4/3)μ(∂u/∂x)
    for 1D flow with zero transverse velocity.

    Args:
        du_dx: Velocity gradient [1/s]
        mu: Dynamic viscosity [Pa·s]

    Returns:
        tau: Viscous stress [Pa]
    """
    return (4.0 / 3.0) * mu * du_dx


def compute_species_diffusion_flux(
    rho: Float[Array, " n_interfaces"],
    D_s: Float[Array, "n_interfaces n_species"],
    dc_s_dx: Float[Array, "n_interfaces n_species"],
) -> Float[Array, "n_interfaces n_species"]:
    """Compute species diffusion flux.

    j_s = -ρ D_s ∂c_s/∂x

    where c_s is the mass fraction of species s.

    Args:
        rho: Density [kg/m³]
        D_s: Effective diffusion coefficients [m²/s]
        dc_s_dx: Mass fraction gradients [1/m]

    Returns:
        j_s: Species diffusion flux [kg/(m²·s)]
    """
    return -rho[:, None] * D_s * dc_s_dx


def compute_heat_flux(
    eta_t: Float[Array, " n_interfaces"],
    eta_r: Float[Array, " n_interfaces"],
    eta_v: Float[Array, " n_interfaces"],
    dT_dx: Float[Array, " n_interfaces"],
    dTv_dx: Float[Array, " n_interfaces"],
) -> tuple[Float[Array, " n_interfaces"], Float[Array, " n_interfaces"]]:
    """Compute heat fluxes for translational-rotational and vibrational modes.

    q_tr = -(η_t + η_r) ∂T/∂x     (translational + rotational)
    q_v = -η_v ∂T_v/∂x           (vibrational)

    Args:
        eta_t: Translational thermal conductivity [W/(m·K)]
        eta_r: Rotational thermal conductivity [W/(m·K)]
        eta_v: Vibrational thermal conductivity [W/(m·K)]
        dT_dx: Translational temperature gradient [K/m]
        dTv_dx: Vibrational temperature gradient [K/m]

    Returns:
        q_tr: Trans-rot heat flux [W/m²]
        q_v: Vibrational heat flux [W/m²]
    """
    q_tr = -(eta_t + eta_r) * dT_dx
    q_v = -eta_v * dTv_dx

    return q_tr, q_v


def compute_viscous_flux(
    U: Float[Array, "n_cells_with_halo n_variables"],
    equation_manager: "EquationManager",
) -> Float[Array, "n_interfaces n_variables"]:
    """Compute viscous flux at cell interfaces.

    For the 1D two-temperature model, the viscous flux is:
    F_v = [
        -j_s,                                    # Species: -rho D_s ∂c_s/∂x
        -tau,                                    # Momentum: -(4/3)mu ∂u/∂x
        -tau*u + q_tr + q_v + sum(h_s j_s),      # Total energy
        q_v + sum(e_{v,s} j_s)                   # Vibrational energy
    ]

    Note: The sign convention follows the conservation form where
    the viscous flux acts opposite to the gradient direction.

    Args:
        U: Conserved variables with halo [n_cells_with_halo, n_variables]
        equation_manager: Contains species data and collision integrals

    Returns:
        F_v: Viscous flux at interfaces [n_interfaces, n_variables]
    """
    species_table = equation_manager.species
    dx = equation_manager.numerics_config.dx
    n_species = species_table.n_species
    n_cells_with_halo = U.shape[0]
    n_interfaces = n_cells_with_halo - 1
    n_variables = U.shape[1]

    # If transport is disabled, return zero flux (inviscid case)
    if equation_manager.transport_model is None:
        return jnp.zeros((n_interfaces, n_variables))

    # Extract primitives from conserved variables
    Y_s, rho, T, T_v, p = equation_manager_utils.extract_primitives_from_U(
        U, equation_manager
    )

    # Compute mass fractions
    rho_s = U[:, :n_species]
    c_s = rho_s / rho[:, None]

    # Extract velocity
    u = U[:, n_species] / rho

    mu, eta_t, eta_r, eta_v, D_s = transport_models.compute_transport_properties(
        T, T_v, p, Y_s, rho, equation_manager
    )

    # Compute gradients at interfaces
    du_dx = compute_gradients_at_interfaces(u, dx)
    dT_dx = compute_gradients_at_interfaces(T, dx)
    dTv_dx = compute_gradients_at_interfaces(T_v, dx)
    # dc_s_dx = compute_gradients_at_interfaces_multispecies(c_s, dx)
    dc_s_dx = compute_gradients_at_interfaces_multispecies(c_s, dx)

    # Apply physical cap to diffusion coefficients
    clip_cfg = equation_manager.numerics_config.clipping
    D_s = jnp.clip(D_s, clip_cfg.D_s_min, clip_cfg.D_s_max)

    # Interpolate transport properties to interfaces
    mu_face = compute_interface_values(mu)
    eta_t_face = compute_interface_values(eta_t)
    eta_r_face = compute_interface_values(eta_r)
    eta_v_face = compute_interface_values(eta_v)
    rho_face = compute_interface_values(rho)
    u_face = compute_interface_values(u)
    D_s_face = compute_interface_values_multispecies(D_s)
    T_face = compute_interface_values(T)
    T_v_face = compute_interface_values(T_v)
    # c_s_face = compute_interface_values_multispecies(c_s)

    # Limit diffusion by diffusive CFL at interfaces
    D_cap = dx**2 / (2.0 * equation_manager.numerics_config.dt)
    D_s_face = jnp.clip(D_s_face, 0.0, D_cap)

    # Compute species diffusion flux at interfaces
    # j_s = compute_species_diffusion_flux(rho_face, D_s_face, dc_s_dx)
    j_s = rho_face[:, None] * D_s_face * dc_s_dx

    # Compute viscous stress at interfaces
    # tau = compute_viscous_stress_1d(du_dx, mu_face)
    tau = 4 / 3 * mu_face * du_dx

    # Compute heat fluxes at interfaces
    # q_tr = -(η_t + η_r) ∂T/∂x     (translational + rotational)
    # q_v = -η_v ∂T_v/∂x           (vibrational)

    # q_tr, q_v = compute_heat_flux(eta_t_face, eta_r_face, eta_v_face, dT_dx, dTv_dx)
    q_tr = -(eta_t_face + eta_r_face) * dT_dx  # translational + rotational
    q_v = -eta_v_face * dTv_dx  # vibrational

    # Compute species enthalpies at interface temperature
    h_s = thermodynamic_relations.compute_equilibrium_enthalpy(
        T_face, species_table
    )  # [n_species, n_interfaces]
    h_s = h_s.T  # [n_interfaces, n_species]

    # Compute vibrational enthalpy (approximately equal to vibrational energy for ideal gas)
    e_v_s = thermodynamic_relations.compute_e_ve(
        T_v_face, species_table
    )  # [n_species, n_interfaces]
    e_v_s = e_v_s.T  # [n_interfaces, n_species]

    # Assemble viscous flux
    F_v = jnp.zeros((n_interfaces, n_variables))

    # Species flux: F_{v,s} = -j_s = rho D_s dc_s/dx
    F_v = F_v.at[:, :n_species].set(-j_s)

    # Momentum flux: F_{v,rhou} = -tau
    F_v = F_v.at[:, n_species].set(-tau)

    # Total energy flux: F_{v,rhoE} = -tau*u + q_tr + q_v - sum(h_s j_s)
    energy_diffusion = -1 * jnp.sum(h_s * j_s, axis=-1)
    F_v = F_v.at[:, n_species + 1].set(-tau * u_face + q_tr + q_v + energy_diffusion)

    # Vibrational energy flux: F_{v,rhoEv} = q_v - sum(e_{v,s} j_s)
    vib_energy_diffusion = -1 * jnp.sum(e_v_s * j_s, axis=-1)
    F_v = F_v.at[:, n_species + 2].set(q_v + vib_energy_diffusion)

    return F_v


def compute_diffusive_cfl(
    rho: Float[Array, " n_cells"],
    mu: Float[Array, " n_cells"],
    eta: Float[Array, " n_cells"],
    D_s: Float[Array, "n_cells n_species"],
    cv: Float[Array, " n_cells"],
    dx: float,
) -> Float[Array, ""]:
    """Compute maximum stable timestep for diffusive terms.

    The diffusive CFL condition requires:
        dt <= dx² / (2 * max(ν, α, D))

    where:
        ν = μ/ρ is kinematic viscosity
        α = η/(ρ·cv) is thermal diffusivity
        D = max species diffusion coefficient

    Args:
        rho: Density [kg/m³]
        mu: Dynamic viscosity [Pa·s]
        eta: Thermal conductivity [W/(m·K)]
        D_s: Species diffusion coefficients [m²/s]
        cv: Specific heat at constant volume [J/(kg·K)]
        dx: Grid spacing [m]

    Returns:
        dt_diff: Maximum stable diffusive timestep [s]
    """
    # Kinematic viscosity
    nu = mu / rho

    # Thermal diffusivity
    alpha = eta / (rho * cv)

    # Maximum species diffusion coefficient
    D_max = jnp.max(D_s, axis=-1)

    # Maximum diffusivity across all modes
    diff_max = jnp.maximum(jnp.maximum(nu, alpha), D_max)

    # Minimum across all cells
    diff_max_global = jnp.max(diff_max)

    # CFL condition
    dt_diff = 0.5 * dx**2 / (2.0 * diff_max_global + 1e-30)

    return dt_diff
