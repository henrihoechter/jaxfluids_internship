"""Boundary conditions for 2D axisymmetric solver."""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from compressible_core import constants, thermodynamic_relations

from .boundary_conditions_types import (
    BC_AXISYMMETRIC,
    BC_INFLOW,
    BC_WALL,
    BC_WALL_SLIP,
    BC_WALL_EULER,
)
from .equation_manager_types import BoundaryConditionArrays2D, EquationManager2D
from .mesh_gmsh import Mesh2D
from . import equation_manager_utils


def compute_slip_wall_ghost(
    U_L: Float[Array, "n_faces n_variables"],
    n_hat: Float[Array, "n_faces 2"],
    equation_manager: EquationManager2D,
    Tw: Float[Array, "n_faces"],
    Tvw: Float[Array, "n_faces"],
    Y_wall: Float[Array, "n_faces n_species"],
    wall_u: Float[Array, "n_faces"],
    wall_v: Float[Array, "n_faces"],
    wall_dist: Float[Array, "n_faces"],
    sigma_t: Float[Array, "n_faces"],
    sigma_v: Float[Array, "n_faces"],
) -> Float[Array, "n_faces n_variables"]:
    """Compute ghost state for a slip/jump wall using Maxwell/Smoluchowski forms."""
    Y_L, rho_L, u_L, v_L, T_L, Tv_L, p_L = (
        equation_manager_utils.extract_primitives_from_U(U_L, equation_manager)
    )

    n_x = n_hat[:, 0]
    n_y = n_hat[:, 1]
    t_x = -n_y
    t_y = n_x

    u_n_L = u_L * n_x + v_L * n_y
    u_t_L = u_L * t_x + v_L * t_y

    # Mixture cp, cv_tr, gamma
    cp_tr = thermodynamic_relations.compute_cp_tr(T_L, equation_manager.species)
    cv_tr = thermodynamic_relations.compute_cv_tr(T_L, equation_manager.species)
    cp_mix = jnp.sum(Y_L * cp_tr.T, axis=1)
    cv_tr_mix = jnp.sum(Y_L * cv_tr.T, axis=1)
    gamma = cp_mix / (cv_tr_mix + 1e-14)

    # Transport properties for Prandtl number and mean free path
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
    pr = mu * cp_mix / jnp.clip(k_tr, 1e-30, None)

    # Mean free path from kinetic theory (Chapman-Enskog hard-sphere):
    # lambda = (16/5) * mu / (rho * cbar)
    M_s = equation_manager.species.molar_masses
    denom = jnp.sum(Y_L / M_s[None, :], axis=1)
    M_mix = 1.0 / jnp.clip(denom, 1e-30, None)
    R_spec = constants.R_universal / M_mix
    cbar = jnp.sqrt(jnp.clip(8.0 * R_spec * T_L / jnp.pi, 1e-30, None))
    lambda_mfp = (16.0 / 5.0) * mu / jnp.clip(rho_L * cbar, 1e-30, None)

    # Smoluchowski temperature jump (Casseau form) — translational-rotational
    sigma_t = jnp.clip(sigma_t, 1e-6, None)
    jump_coeff = (2.0 - sigma_t) / sigma_t
    Kn = lambda_mfp / jnp.clip(wall_dist, 1e-12, None)
    A_T = jump_coeff * (2.0 * gamma / (gamma + 1.0)) * (Kn / jnp.clip(pr, 1e-30, None))

    T_gs = (2.0 * Tw + (A_T - 1.0) * T_L) / jnp.clip(1.0 + A_T, 1e-6, None)
    T_gs = jnp.clip(T_gs, 1.0, None)

    # Smoluchowski temperature jump — vibrational (uses eta_v and cv_ve)
    cv_v = thermodynamic_relations.compute_cv_ve(Tv_L, equation_manager.species)
    cv_v_mix = jnp.sum(Y_L * cv_v.T, axis=1)
    pr_v = mu * cv_v_mix / jnp.clip(eta_v, 1e-30, None)
    A_Tv = (
        jump_coeff * (2.0 * gamma / (gamma + 1.0)) * (Kn / jnp.clip(pr_v, 1e-30, None))
    )

    Tv_gs = (2.0 * Tvw + (A_Tv - 1.0) * Tv_L) / jnp.clip(1.0 + A_Tv, 1e-6, None)
    Tv_gs = jnp.clip(Tv_gs, 1.0, None)

    # Maxwell slip (sigma_v=1 for Casseau)
    sigma_v = jnp.clip(sigma_v, 1e-6, None)
    slip_coeff = (2.0 - sigma_v) / sigma_v
    A_u = slip_coeff * lambda_mfp / jnp.clip(wall_dist, 1e-12, None)
    A_u = jnp.clip(A_u, -0.95, 0.95)

    u_t_wall = wall_u * t_x + wall_v * t_y
    u_t_g = (u_t_wall + A_u * u_t_L) / jnp.clip(1.0 + A_u, 1e-6, None)
    u_t_gs = 2.0 * u_t_g - u_t_L

    # Impermeable wall
    u_n_g = -u_n_L

    u_g = u_n_g * n_x + u_t_gs * t_x
    v_g = u_n_g * n_y + u_t_gs * t_y

    return equation_manager_utils.compute_U_from_primitives(
        Y_s=Y_wall,
        rho=rho_L,
        u=u_g,
        v=v_g,
        T_tr=T_gs,
        T_V=Tv_gs,
        equation_manager=equation_manager,
    )


def _ghost_inflow(
    boundary_arrays: BoundaryConditionArrays2D,
    equation_manager: EquationManager2D,
) -> Float[Array, "n_faces n_variables"]:
    return equation_manager_utils.compute_U_from_primitives(
        Y_s=boundary_arrays.inflow_Y,
        rho=boundary_arrays.inflow_rho,
        u=boundary_arrays.inflow_u,
        v=boundary_arrays.inflow_v,
        T_tr=boundary_arrays.inflow_T,
        T_V=boundary_arrays.inflow_Tv,
        equation_manager=equation_manager,
    )


def _ghost_axisymmetric(
    U_L: Float[Array, "n_faces n_variables"],
    n_hat: Float[Array, "n_faces 2"],
    equation_manager: EquationManager2D,
) -> Float[Array, "n_faces n_variables"]:
    Y_L, rho_L, u_L, v_L, T_L, Tv_L, _ = (
        equation_manager_utils.extract_primitives_from_U(U_L, equation_manager)
    )

    n_x = n_hat[:, 0]
    n_y = n_hat[:, 1]
    t_x = -n_y
    t_y = n_x

    u_n = u_L * n_x + v_L * n_y
    u_t = u_L * t_x + v_L * t_y

    u_n_g = -u_n
    u_t_g = u_t
    u_g = u_n_g * n_x + u_t_g * t_x
    v_g = u_n_g * n_y + u_t_g * t_y

    return equation_manager_utils.compute_U_from_primitives(
        Y_s=Y_L,
        rho=rho_L,
        u=u_g,
        v=v_g,
        T_tr=T_L,
        T_V=Tv_L,
        equation_manager=equation_manager,
    )


def _ghost_wall_euler(
    U_L: Float[Array, "n_faces n_variables"],
    n_hat: Float[Array, "n_faces 2"],
    equation_manager: EquationManager2D,
) -> Float[Array, "n_faces n_variables"]:
    """Euler (inviscid) slip wall: reflect only the normal velocity component."""
    Y_L, rho_L, u_L, v_L, T_L, Tv_L, _ = (
        equation_manager_utils.extract_primitives_from_U(U_L, equation_manager)
    )
    n_x = n_hat[:, 0]
    n_y = n_hat[:, 1]
    t_x = -n_y
    t_y = n_x
    u_n = u_L * n_x + v_L * n_y
    u_t = u_L * t_x + v_L * t_y
    u_g = -u_n * n_x + u_t * t_x
    v_g = -u_n * n_y + u_t * t_y
    return equation_manager_utils.compute_U_from_primitives(
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
    boundary_arrays: BoundaryConditionArrays2D,
    bc_id: Array,
    equation_manager: EquationManager2D,
) -> Float[Array, "n_faces n_variables"]:
    Y_L, rho_L, u_L, v_L, T_L, Tv_L, _ = (
        equation_manager_utils.extract_primitives_from_U(U_L, equation_manager)
    )

    wall_mask = bc_id == BC_WALL
    wall_count = jnp.sum(wall_mask)
    wall_count_safe = jnp.maximum(wall_count, 1.0)
    Tw_wall_mean = jnp.sum(T_L * wall_mask) / wall_count_safe
    Tw_default = jnp.where(wall_count > 0, Tw_wall_mean, jnp.mean(T_L))

    Tw = jnp.where(boundary_arrays.wall_has_Tw, boundary_arrays.wall_Tw, Tw_default)
    Tvw = jnp.where(boundary_arrays.wall_has_Tvw, boundary_arrays.wall_Tvw, Tw)
    Y = jnp.where(boundary_arrays.wall_has_Y[:, None], boundary_arrays.wall_Y, Y_L)

    # No-slip: reflect velocity so that the face-averaged velocity is zero at the wall.
    u_g = -u_L
    v_g = -v_L

    # Dirichlet temperature: set ghost so T_face = (T_L + T_ghost)/2 = Tw exactly.
    # => T_ghost = 2*Tw - T_L
    T_ghost = 2.0 * Tw - T_L
    Tv_ghost = 2.0 * Tvw - Tv_L

    # Pressure-matching density: keep p_ghost = p_L so the Riemann solver sees no
    # spurious pressure jump at the wall (p = rho*R*T for ideal gas).
    # => rho_ghost = rho_L * T_L / T_ghost
    rho_ghost = rho_L * T_L / jnp.clip(T_ghost, 1.0, None)

    return equation_manager_utils.compute_U_from_primitives(
        Y_s=Y,
        rho=rho_ghost,
        u=u_g,
        v=v_g,
        T_tr=T_ghost,
        T_V=Tv_ghost,
        equation_manager=equation_manager,
    )


def _ghost_wall_slip(
    U_L: Float[Array, "n_faces n_variables"],
    boundary_arrays: BoundaryConditionArrays2D,
    bc_id: Array,
    n_hat: Float[Array, "n_faces 2"],
    equation_manager: EquationManager2D,
) -> Float[Array, "n_faces n_variables"]:
    Y_L, rho_L, u_L, v_L, T_L, Tv_L, _ = (
        equation_manager_utils.extract_primitives_from_U(U_L, equation_manager)
    )

    wall_mask = bc_id == BC_WALL_SLIP
    wall_count = jnp.sum(wall_mask)
    wall_count_safe = jnp.maximum(wall_count, 1.0)
    Tw_wall_mean = jnp.sum(T_L * wall_mask) / wall_count_safe
    Tw_default = jnp.where(wall_count > 0, Tw_wall_mean, jnp.mean(T_L))

    Tw = jnp.where(boundary_arrays.wall_has_Tw, boundary_arrays.wall_Tw, Tw_default)
    Tvw = jnp.where(boundary_arrays.wall_has_Tvw, boundary_arrays.wall_Tvw, Tw)
    Y = jnp.where(boundary_arrays.wall_has_Y[:, None], boundary_arrays.wall_Y, Y_L)

    return compute_slip_wall_ghost(
        U_L=U_L,
        n_hat=n_hat,
        equation_manager=equation_manager,
        Tw=Tw,
        Tvw=Tvw,
        Y_wall=Y,
        wall_u=boundary_arrays.wall_u,
        wall_v=boundary_arrays.wall_v,
        wall_dist=boundary_arrays.wall_dist,
        sigma_t=boundary_arrays.wall_sigma_t,
        sigma_v=boundary_arrays.wall_sigma_v,
    )


@jax.named_call
def compute_face_states(
    U: Float[Array, "n_cells n_variables"],
    mesh: Mesh2D,
    equation_manager: EquationManager2D,
) -> tuple[Float[Array, "n_faces n_variables"], Float[Array, "n_faces n_variables"]]:
    face_left = jnp.asarray(mesh.face_left)
    face_right = jnp.asarray(mesh.face_right)
    U_L = U[face_left]
    U_R = jnp.where(face_right[:, None] >= 0, U[face_right], U_L)

    bc = equation_manager.boundary_arrays
    if bc is None:
        raise ValueError(
            "boundary_arrays is required for JIT-safe execution. "
            "Build it with compressible_2d.build_boundary_arrays(...)."
        )

    bc_id = bc.bc_id
    n_hat = jnp.asarray(mesh.face_normals)

    U_R_inflow = _ghost_inflow(bc, equation_manager)
    U_R_axis = _ghost_axisymmetric(U_L, n_hat, equation_manager)
    U_R_wall = _ghost_wall(U_L, bc, bc_id, equation_manager)
    U_R_wall_slip = _ghost_wall_slip(U_L, bc, bc_id, n_hat, equation_manager)
    U_R_wall_euler = _ghost_wall_euler(U_L, n_hat, equation_manager)

    mask_inflow = bc_id == BC_INFLOW
    mask_axis = bc_id == BC_AXISYMMETRIC
    mask_wall = bc_id == BC_WALL
    mask_wall_slip = bc_id == BC_WALL_SLIP
    mask_wall_euler = bc_id == BC_WALL_EULER

    U_R = jnp.where(mask_inflow[:, None], U_R_inflow, U_R)
    U_R = jnp.where(mask_axis[:, None], U_R_axis, U_R)
    U_R = jnp.where(mask_wall[:, None], U_R_wall, U_R)
    U_R = jnp.where(mask_wall_slip[:, None], U_R_wall_slip, U_R)
    U_R = jnp.where(mask_wall_euler[:, None], U_R_wall_euler, U_R)
    return U_L, U_R


def compute_ghost_state(
    U_L: Float[Array, "n_faces n_variables"],
    n_hat: Float[Array, "n_faces 2"],
    bc_type: str,
    bc_params: dict,
    equation_manager: EquationManager2D,
) -> Float[Array, "n_faces n_variables"]:
    """Compute ghost state for a boundary face."""
    Y_L, rho_L, u_L, v_L, T_L, Tv_L, _p_L = (
        equation_manager_utils.extract_primitives_from_U(U_L, equation_manager)
    )

    n_x = n_hat[:, 0]
    n_y = n_hat[:, 1]
    t_x = -n_y
    t_y = n_x

    u_n = u_L * n_x + v_L * n_y
    u_t = u_L * t_x + v_L * t_y

    if bc_type == "outflow":
        return U_L

    if bc_type == "inflow":
        inflow = bc_params
        rho = jnp.full_like(rho_L, float(inflow["rho"]))
        u = jnp.full_like(rho_L, float(inflow["u"]))
        v = jnp.full_like(rho_L, float(inflow["v"]))
        T = jnp.full_like(rho_L, float(inflow["T"]))
        Tv = jnp.full_like(rho_L, float(inflow.get("Tv", inflow["T"])))
        Y = jnp.asarray(inflow["Y"], dtype=U_L.dtype)
        if Y.ndim == 1:
            Y = jnp.broadcast_to(Y[None, :], (rho_L.shape[0], Y.shape[0]))
        return equation_manager_utils.compute_U_from_primitives(
            Y_s=Y,
            rho=rho,
            u=u,
            v=v,
            T_tr=T,
            T_V=Tv,
            equation_manager=equation_manager,
        )

    if bc_type == "axisymmetric":
        # Reflect normal velocity, keep tangential
        u_n_g = -u_n
        u_t_g = u_t
        u_g = u_n_g * n_x + u_t_g * t_x
        v_g = u_n_g * n_y + u_t_g * t_y
        return equation_manager_utils.compute_U_from_primitives(
            Y_s=Y_L,
            rho=rho_L,
            u=u_g,
            v=v_g,
            T_tr=T_L,
            T_V=Tv_L,
            equation_manager=equation_manager,
        )

    if bc_type == "wall":
        # No-slip isothermal wall: Dirichlet temperature via ghost cell.
        # Ghost is chosen so T_face = (T_L + T_ghost)/2 = Tw => T_ghost = 2*Tw - T_L.
        Tw = float(bc_params.get("Tw", T_L.mean()))
        Tvw = float(bc_params.get("Tvw", Tw))
        u_g = -u_L
        v_g = -v_L
        Y_wall = bc_params.get("Y_wall", None)
        if Y_wall is None:
            Y = Y_L
        else:
            Y = jnp.asarray(Y_wall, dtype=U_L.dtype)
            if Y.ndim == 1:
                Y = jnp.broadcast_to(Y[None, :], (rho_L.shape[0], Y.shape[0]))
        Tw_arr = jnp.full_like(rho_L, Tw)
        Tvw_arr = jnp.full_like(rho_L, Tvw)
        return equation_manager_utils.compute_U_from_primitives(
            Y_s=Y,
            rho=rho_L,
            u=u_g,
            v=v_g,
            T_tr=2.0 * Tw_arr - T_L,
            T_V=2.0 * Tvw_arr - Tv_L,
            equation_manager=equation_manager,
        )

    if bc_type == "wall_slip":
        Tw = float(bc_params.get("Tw", T_L.mean()))
        Tvw = float(bc_params.get("Tvw", Tw))
        u_wall = float(bc_params.get("u_wall", 0.0))
        v_wall = float(bc_params.get("v_wall", 0.0))
        sigma_t = float(bc_params.get("sigma_t", 1.0))
        sigma_v = float(bc_params.get("sigma_v", 1.0))
        wall_dist = float(bc_params.get("wall_dist", 1.0))
        Y_wall = bc_params.get("Y_wall", None)
        if Y_wall is None:
            Y = Y_L
        else:
            Y = jnp.asarray(Y_wall, dtype=U_L.dtype)
            if Y.ndim == 1:
                Y = jnp.broadcast_to(Y[None, :], (rho_L.shape[0], Y.shape[0]))

        return compute_slip_wall_ghost(
            U_L=U_L,
            n_hat=n_hat,
            equation_manager=equation_manager,
            Tw=jnp.full_like(rho_L, Tw),
            Tvw=jnp.full_like(rho_L, Tvw),
            Y_wall=Y,
            wall_u=jnp.full_like(rho_L, u_wall),
            wall_v=jnp.full_like(rho_L, v_wall),
            wall_dist=jnp.full_like(rho_L, wall_dist),
            sigma_t=jnp.full_like(rho_L, sigma_t),
            sigma_v=jnp.full_like(rho_L, sigma_v),
        )

    if bc_type == "wall_euler":
        # Euler (inviscid) slip wall: reflect only the normal velocity, keep tangential.
        u_n_g = -u_n
        u_t_g = u_t
        u_g = u_n_g * n_x + u_t_g * t_x
        v_g = u_n_g * n_y + u_t_g * t_y
        return equation_manager_utils.compute_U_from_primitives(
            Y_s=Y_L,
            rho=rho_L,
            u=u_g,
            v=v_g,
            T_tr=T_L,
            T_V=Tv_L,
            equation_manager=equation_manager,
        )

    raise ValueError(f"Unknown boundary condition type: {bc_type}")
