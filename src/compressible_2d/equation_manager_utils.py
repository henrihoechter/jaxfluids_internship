from typing import NamedTuple

from jaxtyping import Array, Float
import jax
import jax.numpy as jnp

from compressible_core import chemistry_types
from .boundary_conditions_utils import build_boundary_arrays
from .equation_manager_types import BoundaryConditionConfig2D, EquationManager2D
from .mesh_gmsh import Mesh2D
from compressible_core import constants
from compressible_core import thermodynamic_relations


class Primitives2D(NamedTuple):
    Y_s: Array
    rho: Array
    u: Array
    v: Array
    T: Array
    Tv: Array
    p: Array


def extract_primitives(
    U: Float[Array, "n_cells n_variables"],
    equation_manager: EquationManager2D,
) -> Primitives2D:
    return Primitives2D(*extract_primitives_from_U(U, equation_manager))


@jax.named_call
def extract_primitives_from_U(
    U: Float[Array, "n_cells n_variables"],
    equation_manager: EquationManager2D,
) -> tuple[
    Float[Array, "n_cells n_species"],
    Float[Array, " n_cells"],
    Float[Array, " n_cells"],
    Float[Array, " n_cells"],
    Float[Array, " n_cells"],
    Float[Array, " n_cells"],
    Float[Array, " n_cells"],
]:
    """Extract primitive variables from conserved state vector.

    State vector: U = [rho_1, ..., rho_ns, rho_u, rho_v, rho_E, rho_Ev]
    Returns Y_s, rho, u, v, T, Tv, p.
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

    clip_config = equation_manager.numerics_config.clipping
    rho = jnp.clip(rho, clip_config.rho_min, clip_config.rho_max)
    Y_s = jnp.clip(Y_s, clip_config.Y_min, clip_config.Y_max)
    T = jnp.clip(T, clip_config.T_min, clip_config.T_max)
    T_V = jnp.clip(T_V, clip_config.Tv_min, clip_config.Tv_max)
    p = jnp.clip(p, clip_config.p_min, clip_config.p_max)

    return Y_s, rho, u, v, T, T_V, p


def compute_U_from_primitives(
    Y_s: Float[Array, "n_cells n_species"],
    rho: Float[Array, " n_cells"],
    u: Float[Array, " n_cells"],
    v: Float[Array, " n_cells"],
    T_tr: Float[Array, " n_cells"],
    T_V: Float[Array, " n_cells"],
    equation_manager: EquationManager2D,
) -> Float[Array, "n_cells n_variables"]:
    """Compute conserved state vector from primitive variables.

    Returns U = [rho_1, ..., rho_ns, rho_u, rho_v, rho_E, rho_Ev].
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


def build_equation_manager(
    mesh: Mesh2D,
    *,
    species: chemistry_types.SpeciesTable,
    collision_integrals: chemistry_types.CollisionIntegralTable | None,
    reactions: chemistry_types.ReactionTable | None,
    numerics_config,
    boundary_config: BoundaryConditionConfig2D,
    transport_model,
    casseau_transport=None,
) -> EquationManager2D:
    boundary_arrays = build_boundary_arrays(mesh, boundary_config, species)
    return EquationManager2D(
        species=species,
        collision_integrals=collision_integrals,
        reactions=reactions,
        numerics_config=numerics_config,
        boundary_arrays=boundary_arrays,
        transport_model=transport_model,
        casseau_transport=casseau_transport,
    )
