"""Boundary conditions for 2D axisymmetric solver."""

import jax.numpy as jnp
from jaxtyping import Array, Float

from compressible_core import constants, thermodynamic_relations, transport_models

from .equation_manager_types import EquationManager2D
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
    cp = thermodynamic_relations.compute_cp(T_L, equation_manager.species)
    cv_tr = thermodynamic_relations.compute_cv_tr(T_L, equation_manager.species)
    cp_mix = jnp.sum(Y_L * cp.T, axis=1)
    cv_tr_mix = jnp.sum(Y_L * cv_tr.T, axis=1)
    gamma = cp_mix / (cv_tr_mix + 1e-14)

    # Transport properties for Prandtl number and mean free path
    mu, eta_t, eta_r, _eta_v, _D_s = transport_models.compute_transport_properties(
        T_L, Tv_L, p_L, Y_L, rho_L, equation_manager
    )
    k_tr = eta_t + eta_r
    pr = mu * cp_mix / jnp.clip(k_tr, 1e-30, None)

    # Mean free path from kinetic theory: lambda = 3*mu / (rho * cbar)
    M_s = equation_manager.species.molar_masses
    denom = jnp.sum(Y_L / M_s[None, :], axis=1)
    M_mix = 1.0 / jnp.clip(denom, 1e-30, None)
    R_spec = constants.R_universal / M_mix
    cbar = jnp.sqrt(jnp.clip(8.0 * R_spec * T_L / jnp.pi, 1e-30, None))
    lambda_mfp = 3.0 * mu / jnp.clip(rho_L * cbar, 1e-30, None)

    # Smoluchowski temperature jump (sigma_t=1 for Casseau)
    sigma_t = jnp.clip(sigma_t, 1e-6, None)
    jump_coeff = (2.0 - sigma_t) / sigma_t
    A_T = (
        jump_coeff
        * (2.0 * gamma / (gamma + 1.0))
        * (lambda_mfp / jnp.clip(pr, 1e-30, None))
    )
    A_T = A_T / jnp.clip(wall_dist, 1e-12, None)
    A_T = jnp.clip(A_T, -0.95, 0.95)

    T_g = (Tw + A_T * T_L) / jnp.clip(1.0 + A_T, 1e-6, None)
    T_g_v = (Tvw + A_T * Tv_L) / jnp.clip(1.0 + A_T, 1e-6, None)
    T_gs = 2.0 * T_g - T_L
    Tv_gs = 2.0 * T_g_v - Tv_L

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
        # No-slip isothermal wall by default
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
        return equation_manager_utils.compute_U_from_primitives(
            Y_s=Y,
            rho=rho_L,
            u=u_g,
            v=v_g,
            T_tr=jnp.full_like(rho_L, Tw),
            T_V=jnp.full_like(rho_L, Tvw),
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

    raise ValueError(f"Unknown boundary condition type: {bc_type}")
