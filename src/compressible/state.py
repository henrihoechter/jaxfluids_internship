"""Unified state representation for the compressible solver.

State vector: U = [rho_1, ..., rho_ns, rho*u, rho*v, rho*E, rho*E_v]
              shape: (n_cells, n_species + 4)

For 1D problems, rho*v = 0 everywhere by construction.  All solver kernels
operate on this n+4 layout; no dimension-specific branching is needed.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from compressible_core import constants, thermodynamic_relations
from compressible.equation_manager_types import EquationManager


class Primitives(NamedTuple):
    """Primitive variables extracted from the conserved state.

    For 1D simulations, v is always zero.
    """

    Y_s: Array  # Mole fractions          (n_cells, n_species)
    rho: Array  # Total density            (n_cells,)
    u: Array  # x-velocity               (n_cells,)
    v: Array  # y-velocity (0 for 1D)    (n_cells,)
    T: Array  # Translational temperature (n_cells,)
    Tv: Array  # Vibrational temperature   (n_cells,)
    p: Array  # Pressure                  (n_cells,)


@jax.named_call
def extract_primitives_from_U(
    U: Float[Array, "n_cells n_variables"],
    equation_manager: EquationManager,
) -> Primitives:
    """Extract primitive variables from conserved state.

    State vector layout: [rho_s..., rho*u, rho*v, rho*E, rho*E_v]
    n_variables = n_species + 4 (rho*v = 0 for 1D)

    Args:
        U: Conserved state [n_cells, n_variables]
        equation_manager: Contains species table and clipping config.

    Returns:
        Primitives namedtuple with Y_s, rho, u, v, T, Tv, p.
    """
    n_species = equation_manager.species.n_species

    rho_s = U[:, :n_species]
    rho_u = U[:, n_species]
    rho_v = U[:, n_species + 1]
    rho_E = U[:, n_species + 2]
    rho_Ev = U[:, n_species + 3]

    rho = jnp.sum(rho_s, axis=-1)
    u = rho_u / rho
    v = rho_v / rho

    E_kin = 0.5 * (u**2 + v**2)
    E_v = rho_Ev / rho
    E_total = rho_E / rho
    e = E_total - E_kin

    M_s = equation_manager.species.M_s
    c_s = rho_s / rho[:, None]

    T_V_initial = jnp.full_like(rho, 298.16)
    T_V = (
        thermodynamic_relations.solve_vibrational_temperature_from_vibroelectric_energy(
            e_V_target=E_v,
            c_s=c_s.T,
            T_V_initial=T_V_initial,
            species_table=equation_manager.species,
            max_iterations=20,
            rtol=1e-6,
            atol=1.0,
        )
    )

    T_dummy = jnp.ones(1)
    cv_tr_all = thermodynamic_relations.compute_cv_tr(T_dummy, equation_manager.species)
    cv_tr_broadcast = jnp.broadcast_to(cv_tr_all[:, 0, None], (n_species, rho.shape[0]))

    e_s0 = thermodynamic_relations.compute_reference_internal_energy(
        equation_manager.species.h_s0,
        equation_manager.species.molar_masses,
        T_ref=298.16,
    )

    T = thermodynamic_relations.solve_T_from_internal_energy(
        e=e,
        e_V=E_v,
        c_s=c_s.T,
        cv_tr=cv_tr_broadcast,
        e_s0=e_s0,
        T_ref=298.16,
    )

    Y_s = (rho_s / M_s[None, :]) / jnp.sum(rho_s / M_s[None, :], axis=-1, keepdims=True)

    T_pressure = jnp.broadcast_to(T[:, None], (rho_s.shape[0], n_species))
    electron_idx = equation_manager.species.electron_index
    if electron_idx is not None:
        T_pressure = T_pressure.at[:, electron_idx].set(T_V)

    p_s = rho_s * constants.R_universal / M_s[None, :] * T_pressure
    p = jnp.sum(p_s, axis=-1)

    clip = equation_manager.numerics_config.clipping
    rho = jnp.clip(rho, clip.rho_min, clip.rho_max)
    Y_s = jnp.clip(Y_s, clip.Y_min, clip.Y_max)
    T = jnp.clip(T, clip.T_min, clip.T_max)
    T_V = jnp.clip(T_V, clip.Tv_min, clip.Tv_max)
    p = jnp.clip(p, clip.p_min, clip.p_max)

    return Primitives(Y_s=Y_s, rho=rho, u=u, v=v, T=T, Tv=T_V, p=p)


def extract_primitives(
    U: Float[Array, "n_cells n_variables"],
    equation_manager: EquationManager,
) -> Primitives:
    """Alias for extract_primitives_from_U."""
    return extract_primitives_from_U(U, equation_manager)


def compute_U_from_primitives(
    Y_s: Float[Array, "n_cells n_species"],
    rho: Float[Array, " n_cells"],
    u: Float[Array, " n_cells"],
    v: Float[Array, " n_cells"],
    T_tr: Float[Array, " n_cells"],
    T_V: Float[Array, " n_cells"],
    equation_manager: EquationManager,
) -> Float[Array, "n_cells n_variables"]:
    """Compute conserved state from primitives.

    Returns U = [rho_s..., rho*u, rho*v, rho*E, rho*E_v] of shape
    (n_cells, n_species + 4).  For 1D, pass v=0.
    """
    n_species = equation_manager.species.n_species
    M_s = equation_manager.species.M_s
    n_cells = rho.shape[0]

    Y_M = Y_s * M_s[None, :]
    c_s = Y_M / jnp.sum(Y_M, axis=-1, keepdims=True)
    rho_s = c_s * rho[:, None]

    rho_u = rho * u
    rho_v = rho * v
    E_kin = 0.5 * (u**2 + v**2)

    e_v_species = thermodynamic_relations.compute_e_ve(T_V, equation_manager.species)
    E_v = jnp.sum(c_s.T * e_v_species, axis=0)

    cv_tr_all = thermodynamic_relations.compute_cv_tr(T_tr, equation_manager.species)
    e_s0 = thermodynamic_relations.compute_reference_internal_energy(
        equation_manager.species.h_s0,
        equation_manager.species.molar_masses,
        T_ref=298.16,
    )

    e_0 = jnp.sum(c_s * e_s0[None, :], axis=-1)
    cv_tr_mix = jnp.sum(c_s.T * cv_tr_all, axis=0)
    T_ref = 298.16
    e = e_0 + E_v + cv_tr_mix * (T_tr - T_ref)
    E_total = e + E_kin

    rho_E = rho * E_total
    rho_Ev = rho * E_v

    U = jnp.zeros((n_cells, n_species + 4))
    U = U.at[:, :n_species].set(rho_s)
    U = U.at[:, n_species].set(rho_u)
    U = U.at[:, n_species + 1].set(rho_v)
    U = U.at[:, n_species + 2].set(rho_E)
    U = U.at[:, n_species + 3].set(rho_Ev)

    return U
