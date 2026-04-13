"""Boundary-condition state builders for the compressible solver."""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from . import constants, thermodynamic_relations
from .boundary_conditions_types import (
    BC_INFLOW,
    BC_REFLECTIVE,
    BC_WALL,
    BC_WALL_EULER,
    BC_WALL_SLIP,
)
from .equation_manager_types import BoundaryConditionArrays, EquationManager
from .mesh import Mesh
from . import state as state_module


def compute_slip_wall_ghost(
    U_L: Float[Array, "n_faces n_variables"],
    n_hat: Float[Array, "n_faces 2"],
    equation_manager: EquationManager,
    Tw: Float[Array, "n_faces"],
    Y_wall: Float[Array, "n_faces n_species"],
    wall_u: Float[Array, "n_faces"],
    wall_v: Float[Array, "n_faces"],
    wall_dist: Float[Array, "n_faces"],
    sigma: Float[Array, "n_faces"],
    alpha: Float[Array, "n_faces"],
    dT_ds: Float[Array, "n_faces"],
) -> Float[Array, "n_faces n_variables"]:
    """Build Casseau-aligned slip-wall ghost states.

    Args:
        U_L: Interior states sampled on the left side of each boundary face.
        n_hat: Outward unit normals for the boundary faces.
        equation_manager: Solver configuration providing thermodynamic and
            transport closures.
        Tw: Wall translational temperature for each face [K].
        Y_wall: Wall species composition applied to the ghost state.
        wall_u: Wall x-velocity for each face [m/s].
        wall_v: Wall y-velocity for each face [m/s].
        wall_dist: Distance between the wall face and the adjacent cell centroid
            used by the jump relations [m].
        sigma: Tangential momentum accommodation coefficient.
        alpha: Thermal accommodation coefficient.
        dT_ds: Tangential wall-temperature gradient along each face [K/m].

    Returns:
        Ghost-state array for slip-wall boundary faces in conserved variables.
    """
    prim = state_module.extract_primitives_from_U(U_L, equation_manager)
    Y_L, rho_L, u_L, v_L, T_L, Tv_L, p_L = prim

    n_x = n_hat[:, 0]
    n_y = n_hat[:, 1]
    t_x = -n_y
    t_y = n_x

    u_n_L = u_L * n_x + v_L * n_y
    u_t_L = u_L * t_x + v_L * t_y

    cp_tr = thermodynamic_relations.compute_cp_tr(T_L, equation_manager.species)
    cv_tr = thermodynamic_relations.compute_cv_tr(T_L, equation_manager.species)
    cp_mix = jnp.sum(Y_L * cp_tr.T, axis=1)
    cv_tr_mix = jnp.sum(Y_L * cv_tr.T, axis=1)
    gamma = cp_mix / (cv_tr_mix + 1e-14)

    if equation_manager.transport_model is None:
        mu = jnp.zeros_like(T_L)
        eta_t = jnp.zeros_like(T_L)
        eta_r = jnp.zeros_like(T_L)
        eta_v = jnp.zeros_like(T_L)
    else:
        mu, eta_t, eta_r, eta_v, _D_s = (
            equation_manager.transport_model.compute_transport_properties(
                T_L, Tv_L, p_L, Y_L, rho_L
            )
        )
    k_tr = eta_t + eta_r
    pr_tr = mu * cp_mix / jnp.clip(k_tr, 1e-30, None)

    M_s = equation_manager.species.molar_masses
    denom = jnp.sum(Y_L / M_s[None, :], axis=1)
    M_mix = 1.0 / jnp.clip(denom, 1e-30, None)
    R_spec = constants.R_universal / M_mix
    cbar = jnp.sqrt(jnp.clip(8.0 * R_spec * T_L / jnp.pi, 1e-30, None))
    lambda_mfp = (16.0 / 5.0) * mu / jnp.clip(rho_L * cbar, 1e-30, None)

    sigma = jnp.clip(sigma, 1e-6, None)
    alpha = jnp.clip(alpha, 1e-6, None)
    wall_dist = jnp.clip(wall_dist, 1e-12, None)

    beta_T = (
        ((2.0 - alpha) / alpha)
        * (2.0 * gamma / (gamma + 1.0))
        * lambda_mfp
        / jnp.clip(pr_tr, 1e-30, None)
    )
    A_T = beta_T / wall_dist
    T_gs = (2.0 * Tw + (A_T - 1.0) * T_L) / jnp.clip(1.0 + A_T, 1e-6, None)
    T_gs = jnp.clip(T_gs, 1.0, None)

    beta_u = (2.0 / jnp.pi) * ((2.0 - sigma) / sigma) * lambda_mfp
    beta_tc = 3.0 * mu / jnp.clip(4.0 * rho_L * T_L, 1e-30, None)
    A_u = beta_u / wall_dist
    u_t_wall = wall_u * t_x + wall_v * t_y
    u_n_wall = wall_u * n_x + wall_v * n_y
    u_t_gs = (2.0 * u_t_wall + (A_u - 1.0) * u_t_L + 2.0 * beta_tc * dT_ds) / jnp.clip(
        1.0 + A_u, 1e-6, None
    )

    u_n_g = 2.0 * u_n_wall - u_n_L
    u_g = u_n_g * n_x + u_t_gs * t_x
    v_g = u_n_g * n_y + u_t_gs * t_y

    return state_module.compute_U_from_primitives(
        Y_s=Y_wall,
        rho=rho_L,
        u=u_g,
        v=v_g,
        T_tr=T_gs,
        T_V=Tv_L,
        equation_manager=equation_manager,
    )


def _cell_gradient_scalar_from_cells(
    phi: Float[Array, "n_cells"],
    mesh: Mesh,
) -> Float[Array, "n_cells 2"]:
    """Approximate a cell-centred scalar gradient from neighbouring cells."""
    face_left = jnp.asarray(mesh.face_left)
    face_right = jnp.asarray(mesh.face_right)
    normals = jnp.asarray(mesh.face_normals)
    areas = jnp.asarray(mesh.face_areas)
    cell_areas = jnp.asarray(mesh.cell_areas)

    n_cells = cell_areas.shape[0]
    phi_L = phi[face_left]
    interior_mask = face_right >= 0
    safe_right = jnp.where(interior_mask, face_right, face_left)
    phi_R = jnp.where(interior_mask, phi[safe_right], phi_L)
    phi_face = 0.5 * (phi_L + phi_R)

    contrib = phi_face[:, None] * normals * areas[:, None]
    grad = jnp.zeros((n_cells, 2))
    grad = grad.at[face_left].add(contrib)
    grad = grad.at[safe_right].add(jnp.where(interior_mask[:, None], -contrib, 0.0))
    return grad / cell_areas[:, None]


def _ghost_inflow(
    bc: BoundaryConditionArrays,
    equation_manager: EquationManager,
) -> Float[Array, "n_faces n_variables"]:
    """Build inflow ghost states from boundary data."""
    return state_module.compute_U_from_primitives(
        Y_s=bc.inflow_Y,
        rho=bc.inflow_rho,
        u=bc.inflow_u,
        v=bc.inflow_v,
        T_tr=bc.inflow_T,
        T_V=bc.inflow_Tv,
        equation_manager=equation_manager,
    )


def _ghost_reflective(
    U_L: Float[Array, "n_faces n_variables"],
    n_hat: Float[Array, "n_faces 2"],
    equation_manager: EquationManager,
) -> Float[Array, "n_faces n_variables"]:
    """Reflect the normal momentum while preserving the other entries."""
    n_species = equation_manager.species.n_species
    n_x = n_hat[:, 0]
    n_y = n_hat[:, 1]
    rho_u = U_L[:, n_species]
    rho_v = U_L[:, n_species + 1]
    rho_m_dot_n = rho_u * n_x + rho_v * n_y

    U_R = U_L
    U_R = U_R.at[:, n_species].set(rho_u - 2.0 * rho_m_dot_n * n_x)
    U_R = U_R.at[:, n_species + 1].set(rho_v - 2.0 * rho_m_dot_n * n_y)
    return U_R


def _ghost_wall_euler(
    U_L: Float[Array, "n_faces n_variables"],
    n_hat: Float[Array, "n_faces 2"],
    equation_manager: EquationManager,
) -> Float[Array, "n_faces n_variables"]:
    """Build Euler wall ghost states by reflecting the normal velocity."""
    prim = state_module.extract_primitives_from_U(U_L, equation_manager)
    Y_L, rho_L, u_L, v_L, T_L, Tv_L, _ = prim
    n_x = n_hat[:, 0]
    n_y = n_hat[:, 1]
    t_x = -n_y
    t_y = n_x
    u_n = u_L * n_x + v_L * n_y
    u_t = u_L * t_x + v_L * t_y
    u_g = -u_n * n_x + u_t * t_x
    v_g = -u_n * n_y + u_t * t_y
    return state_module.compute_U_from_primitives(
        Y_s=Y_L,
        rho=rho_L,
        u=u_g,
        v=v_g,
        T_tr=T_L,
        T_V=Tv_L,
        equation_manager=equation_manager,
    )


def _ghost_wall(
    U_L: Float[Array, "n_faces n_variables"],
    bc: BoundaryConditionArrays,
    bc_id: Int[Array, "n_faces"],
    equation_manager: EquationManager,
) -> Float[Array, "n_faces n_variables"]:
    """Build no-slip wall ghost states."""
    prim = state_module.extract_primitives_from_U(U_L, equation_manager)
    Y_L, rho_L, u_L, v_L, T_L, Tv_L, _ = prim

    wall_mask = bc_id == BC_WALL
    wall_count = jnp.sum(wall_mask)
    wall_count_safe = jnp.maximum(wall_count, 1.0)
    Tw_wall_mean = jnp.sum(T_L * wall_mask) / wall_count_safe
    Tw_default = jnp.where(wall_count > 0, Tw_wall_mean, jnp.mean(T_L))

    Tw = jnp.where(bc.wall_has_Tw, bc.wall_Tw, Tw_default)
    Tvw = jnp.where(bc.wall_has_Tvw, bc.wall_Tvw, Tw)
    Y = jnp.where(bc.wall_has_Y[:, None], bc.wall_Y, Y_L)

    u_g = -u_L
    v_g = -v_L
    T_ghost = 2.0 * Tw - T_L
    Tv_ghost = 2.0 * Tvw - Tv_L
    rho_ghost = rho_L * T_L / jnp.clip(T_ghost, 1.0, None)

    return state_module.compute_U_from_primitives(
        Y_s=Y,
        rho=rho_ghost,
        u=u_g,
        v=v_g,
        T_tr=T_ghost,
        T_V=Tv_ghost,
        equation_manager=equation_manager,
    )


def _ghost_wall_slip(
    U: Float[Array, "n_cells n_variables"],
    U_L: Float[Array, "n_faces n_variables"],
    bc: BoundaryConditionArrays,
    bc_id: Int[Array, "n_faces"],
    n_hat: Float[Array, "n_faces 2"],
    mesh: Mesh,
    equation_manager: EquationManager,
) -> Float[Array, "n_faces n_variables"]:
    """Build slip-wall ghost states with wall data defaults."""
    prim = state_module.extract_primitives_from_U(U_L, equation_manager)
    Y_L, rho_L, u_L, v_L, T_L, Tv_L, _ = prim

    wall_mask = bc_id == BC_WALL_SLIP
    wall_count = jnp.sum(wall_mask)
    wall_count_safe = jnp.maximum(wall_count, 1.0)
    Tw_wall_mean = jnp.sum(T_L * wall_mask) / wall_count_safe
    Tw_default = jnp.where(wall_count > 0, Tw_wall_mean, jnp.mean(T_L))

    Tw = jnp.where(bc.wall_has_Tw, bc.wall_Tw, Tw_default)
    Y = jnp.where(bc.wall_has_Y[:, None], bc.wall_Y, Y_L)
    cell_primitives = state_module.extract_primitives_from_U(U, equation_manager)
    grad_T_cell = _cell_gradient_scalar_from_cells(cell_primitives.T, mesh)
    face_left = jnp.asarray(mesh.face_left)
    t_x = -n_hat[:, 1]
    t_y = n_hat[:, 0]
    dT_ds = grad_T_cell[face_left, 0] * t_x + grad_T_cell[face_left, 1] * t_y

    return compute_slip_wall_ghost(
        U_L=U_L,
        n_hat=n_hat,
        equation_manager=equation_manager,
        Tw=Tw,
        Y_wall=Y,
        wall_u=bc.wall_u,
        wall_v=bc.wall_v,
        wall_dist=bc.wall_dist,
        sigma=bc.wall_sigma,
        alpha=bc.wall_alpha,
        dT_ds=dT_ds,
    )


@jax.named_call
def compute_face_states(
    U: Float[Array, "n_cells n_variables"],
    mesh: Mesh,
    equation_manager: EquationManager,
) -> tuple[Float[Array, "n_faces n_variables"], Float[Array, "n_faces n_variables"]]:
    """Build the left and right states at every face.

    Args:
        U: Cell-centered conserved state array.
        mesh: Mesh providing face connectivity and normals.
        equation_manager: Solver configuration with boundary-condition arrays.

    Returns:
        Tuple `(U_L, U_R)` containing the left and right conserved states for
        every face, including ghost states on boundaries.
    """
    face_left = jnp.asarray(mesh.face_left)
    face_right = jnp.asarray(mesh.face_right)

    U_L = U[face_left]
    # For boundary faces (face_right == -1): temporarily use U_L as placeholder
    safe_right = jnp.where(face_right >= 0, face_right, 0)
    U_R = jnp.where(face_right[:, None] >= 0, U[safe_right], U_L)

    bc = equation_manager.boundary_arrays
    if bc is None:
        raise ValueError(
            "equation_manager.boundary_arrays must be set. "
            "Use boundary_conditions_utils.build_boundary_arrays_1d() or "
            "build_boundary_arrays_2d()."
        )

    bc_id = bc.bc_id
    n_hat = jnp.asarray(mesh.face_normals)

    # Compute all ghost states (all branches traced in JIT)
    U_R_inflow = _ghost_inflow(bc, equation_manager)
    U_R_reflective = _ghost_reflective(U_L, n_hat, equation_manager)
    U_R_wall = _ghost_wall(U_L, bc, bc_id, equation_manager)
    U_R_wall_slip = _ghost_wall_slip(U, U_L, bc, bc_id, n_hat, mesh, equation_manager)
    U_R_wall_euler = _ghost_wall_euler(U_L, n_hat, equation_manager)

    # Dispatch via jnp.where (JAX-safe: all masks are traced boolean arrays)
    U_R = jnp.where((bc_id == BC_INFLOW)[:, None], U_R_inflow, U_R)
    U_R = jnp.where((bc_id == BC_REFLECTIVE)[:, None], U_R_reflective, U_R)
    U_R = jnp.where((bc_id == BC_WALL)[:, None], U_R_wall, U_R)
    U_R = jnp.where((bc_id == BC_WALL_SLIP)[:, None], U_R_wall_slip, U_R)
    U_R = jnp.where((bc_id == BC_WALL_EULER)[:, None], U_R_wall_euler, U_R)

    return U_L, U_R
