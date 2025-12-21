from jaxtyping import Array, Float
import jax.numpy as jnp

from compressible_1d import equation_manager_types


def extract_primitives_from_U(
    U: Float[Array, "n_cells n_variables"],
    equation_manager: equation_manager_types.EquationManager,
) -> tuple[
    Float[Array, "n_cells n_species"],
    Float[Array, " n_cells"],
    Float[Array, " n_cells"],
    Float[Array, " n_cells"],
    Float[Array, " n_cells"],
]:
    """Extract primitive variables from conserved state vector.

    State vector: U = [rho_1, rho_2, ..., rho_ns, rho_u, rho_E, rho_Ev]
    This conversion is aligned with Gnoffo et al. (1989) Conservation Equations and 
    Physical Models for Hypersonic Air Flows in Thermal and Chemical Nonequilibrium.

    Args:
        U: Conserved variables
        equation_manager: Equation manager containing species data

    Returns:
        Y_s: Mole fractions
        rho: Total density [kg/m^3]
        T: Translational temperature [K]
        Tv: Vibrational temperature [K]
        p: Pressure [Pa]
    """
    n_species = equation_manager.species.n_species()

    # Extract partial densities
    rho_s = U[:, :n_species]  # [n_cells, n_species]
    rho_u = U[:, n_species]
    rho_E = U[:, n_species + 1]
    rho_Ev = U[:, n_species + 2]

    # Total density (Gnoffo 1989, Eq. 6)
    rho = jnp.sum(rho_s, axis=-1)

    # Velocity
    u = rho_u / rho

    # Kinetic energy
    E_kin = 0.5 * u**2

    # Vibrational energy per unit mass
    E_v = rho_Ev / rho

    # Total energy per unit mass
    E_total = rho_E / rho

    

    # Compute pressure
    # M_mix = thermodynamics.compute_mixture_molecular_mass(Y, species_list)
    # p = rho * (R / M_mix) * T

    from compressible_1d import constants

    M_s = equation_manager.species.M_s 

    # Mole fractions (Gnoffo 1989, Eq. 7)
    Y_s = (rho_s / M_s[None, :]) / jnp.sum(rho_s / M_s[None, :], axis=-1, keepdims=True) 

    # Calculate T_pressure according to Gnoffo 1989, Eq. 8, 9a and 9b
    # For heavy particles: T_pressure = T
    # For electrons: T_pressure = T_V
    # TODO: T and Tv need to be computed first (currently commented out above)
    # Placeholder until temperature calculations are implemented:
    # T_pressure = jnp.broadcast_to(T[:, None], (rho_s.shape[0], n_species))  # Shape: (n_cells, n_species)

    # Set electron temperature to Tv if electrons are present
    # electron_idx = equation_manager.species.electron_index
    # if electron_idx is not None:
    #     T_pressure = T_pressure.at[:, electron_idx].set(Tv)

    # Compute partial pressures (Gnoffo 1989, Eq. 8, 9a, 9b)
    # p_s = rho_s * R / M_s * T_pressure
    # Note: M_s is in kg/kmol, R_universal is in J/(mol*K), need factor of 1e3
    # p_s = rho_s * constants.R_universal * 1e3 / M_s[None, :] * T_pressure
    # p = jnp.sum(p_s, axis=-1)  # Shape: (n_cells,)

    # Apply clipping to primitive variables (no if-check for performance)
    # cfg = equation_manager.numerics_config.clipping
    # rho = jnp.clip(rho, cfg.rho_min, cfg.rho_max)
    # Y = jnp.clip(Y, cfg.Y_min, cfg.Y_max)
    # T = jnp.clip(T, cfg.T_min, cfg.T_max)
    # Tv = jnp.clip(Tv, cfg.Tv_min, cfg.Tv_max)
    # p = jnp.clip(p, cfg.p_min, cfg.p_max)

    # TODO: Remove these placeholders once T, Tv, p calculations are implemented
    T = jnp.zeros_like(rho)
    Tv = jnp.zeros_like(rho)
    p = jnp.zeros_like(rho)

    return Y_s, rho, T, Tv, p
