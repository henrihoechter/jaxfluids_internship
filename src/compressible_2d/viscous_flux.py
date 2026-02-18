"""Viscous flux computation for 2D axisymmetric two-temperature NSF."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from .mesh_gmsh import Mesh2D
from .equation_manager_types import EquationManager2D
from . import equation_manager_utils
from compressible_core import thermodynamic_relations


def extract_primitives(
    U: Float[Array, "n_faces n_variables"],
    equation_manager: EquationManager2D,
):
    return equation_manager_utils.extract_primitives(U, equation_manager)


def _face_values(phi_L: jnp.ndarray, phi_R: jnp.ndarray) -> jnp.ndarray:
    return 0.5 * (phi_L + phi_R)


def _compute_cell_gradients_scalar(
    phi_L: jnp.ndarray,
    phi_R: jnp.ndarray,
    mesh: Mesh2D,
) -> jnp.ndarray:
    """Green-Gauss gradients for scalar field."""
    phi_face = _face_values(phi_L, phi_R)
    normals = jnp.asarray(mesh.face_normals)
    areas = jnp.asarray(mesh.face_areas)
    face_left = jnp.asarray(mesh.face_left)
    face_right = jnp.asarray(mesh.face_right)

    contrib = phi_face[:, None] * normals * areas[:, None]
    n_cells = mesh.cell_areas.shape[0]
    grad = jnp.zeros((n_cells, 2))
    grad = grad.at[face_left].add(contrib)
    right_mask = face_right >= 0
    safe_right = jnp.where(right_mask, face_right, 0)
    contrib_right = -contrib
    contrib_right = jnp.where(right_mask[:, None], contrib_right, 0.0)
    grad = grad.at[safe_right].add(contrib_right)

    grad = grad / jnp.asarray(mesh.cell_areas)[:, None]
    return grad


def _compute_cell_gradients_vector(
    phi_L: jnp.ndarray,
    phi_R: jnp.ndarray,
    mesh: Mesh2D,
) -> jnp.ndarray:
    """Green-Gauss gradients for vector field (n_cells, n_species)."""
    phi_face = _face_values(phi_L, phi_R)  # (n_faces, n_species)
    normals = jnp.asarray(mesh.face_normals)
    areas = jnp.asarray(mesh.face_areas)
    face_left = jnp.asarray(mesh.face_left)
    face_right = jnp.asarray(mesh.face_right)

    contrib = phi_face[:, :, None] * normals[:, None, :] * areas[:, None, None]
    n_cells = mesh.cell_areas.shape[0]
    n_species = phi_face.shape[1]
    grad = jnp.zeros((n_cells, n_species, 2))
    grad = grad.at[face_left].add(contrib)
    right_mask = face_right >= 0
    safe_right = jnp.where(right_mask, face_right, 0)
    contrib_right = -contrib
    contrib_right = jnp.where(right_mask[:, None, None], contrib_right, 0.0)
    grad = grad.at[safe_right].add(contrib_right)

    grad = grad / jnp.asarray(mesh.cell_areas)[:, None, None]
    return grad


def _face_gradients(
    grad_cell: jnp.ndarray,
    mesh: Mesh2D,
) -> jnp.ndarray:
    face_left = jnp.asarray(mesh.face_left)
    face_right = jnp.asarray(mesh.face_right)
    grad_L = grad_cell[face_left]
    # Reshape condition to broadcast correctly for both 2D (n_cells, 2)
    # and 3D (n_cells, n_species, 2) gradient arrays.
    cond_shape = (-1,) + (1,) * (grad_L.ndim - 1)
    mask = (face_right >= 0).reshape(cond_shape)
    grad_R = jnp.where(mask, grad_cell[face_right], grad_L)
    return 0.5 * (grad_L + grad_R)


def compute_viscous_flux_faces(
    U: Float[Array, "n_cells n_variables"],
    U_L: Float[Array, "n_faces n_variables"],
    U_R: Float[Array, "n_faces n_variables"],
    mesh: Mesh2D,
    equation_manager: EquationManager2D,
    cell_primitives: equation_manager_utils.Primitives2D | None = None,
    face_primitives_L: equation_manager_utils.Primitives2D | None = None,
    face_primitives_R: equation_manager_utils.Primitives2D | None = None,
) -> Float[Array, "n_faces n_variables"]:
    n_species = equation_manager.species.n_species
    if equation_manager.transport_model is None:
        return jnp.zeros_like(U_L)

    face_left = jnp.asarray(mesh.face_left)
    face_right = jnp.asarray(mesh.face_right)

    # Extract primitives from cell-centered U (allow precomputed)
    if cell_primitives is None:
        cell_primitives = equation_manager_utils.extract_primitives(U, equation_manager)
    Y, rho, u, v, T, Tv, p = cell_primitives

    # Transport properties at cells
    mu, eta_t, eta_r, eta_v, D_s = (
        equation_manager.transport_model.compute_transport_properties(
            T, Tv, p, Y, rho
        )
    )

    # Face primitives (include boundary ghost states via U_R)
    if face_primitives_L is None:
        face_primitives_L = equation_manager_utils.extract_primitives(
            U_L, equation_manager
        )
    if face_primitives_R is None:
        face_primitives_R = equation_manager_utils.extract_primitives(
            U_R, equation_manager
        )

    Y_Lf, rho_Lf, u_Lf, v_Lf, T_Lf, Tv_Lf, _ = face_primitives_L
    Y_Rf, rho_Rf, u_Rf, v_Rf, T_Rf, Tv_Rf, _ = face_primitives_R

    # Cell gradients from face values
    u_L = u_Lf
    u_R = u_Rf
    v_L = v_Lf
    v_R = v_Rf
    T_L = T_Lf
    T_R = T_Rf
    Tv_L = Tv_Lf
    Tv_R = Tv_Rf

    rho_s = U[:, :n_species]
    c_s = rho_s / rho[:, None]
    c_L = c_s[face_left]
    c_R = jnp.where(face_right[:, None] >= 0, c_s[face_right], c_L)

    grad_u_cell = _compute_cell_gradients_scalar(u_L, u_R, mesh)
    grad_v_cell = _compute_cell_gradients_scalar(v_L, v_R, mesh)
    grad_T_cell = _compute_cell_gradients_scalar(T_L, T_R, mesh)
    grad_Tv_cell = _compute_cell_gradients_scalar(Tv_L, Tv_R, mesh)
    grad_c_cell = _compute_cell_gradients_vector(c_L, c_R, mesh)

    grad_u = _face_gradients(grad_u_cell, mesh)
    grad_v = _face_gradients(grad_v_cell, mesh)
    grad_T = _face_gradients(grad_T_cell, mesh)
    grad_Tv = _face_gradients(grad_Tv_cell, mesh)
    grad_c = _face_gradients(grad_c_cell, mesh)

    # Face-averaged values
    mu_face = _face_values(
        mu[face_left], jnp.where(face_right >= 0, mu[face_right], mu[face_left])
    )
    eta_t_face = _face_values(
        eta_t[face_left],
        jnp.where(face_right >= 0, eta_t[face_right], eta_t[face_left]),
    )
    eta_r_face = _face_values(
        eta_r[face_left],
        jnp.where(face_right >= 0, eta_r[face_right], eta_r[face_left]),
    )
    eta_v_face = _face_values(
        eta_v[face_left],
        jnp.where(face_right >= 0, eta_v[face_right], eta_v[face_left]),
    )
    rho_face = _face_values(rho_Lf, rho_Rf)
    u_face = _face_values(u_Lf, u_Rf)
    v_face = _face_values(v_Lf, v_Rf)
    D_s_face = _face_values(
        D_s[face_left],
        jnp.where(face_right[:, None] >= 0, D_s[face_right], D_s[face_left]),
    )
    T_face = _face_values(T_Lf, T_Rf)
    Tv_face = _face_values(Tv_Lf, Tv_Rf)

    # Apply diffusion cap
    clip_cfg = equation_manager.numerics_config.clipping
    D_s_face = jnp.clip(D_s_face, clip_cfg.D_s_min, clip_cfg.D_s_max)

    # Species diffusion flux vector
    j_s = -rho_face[:, None, None] * D_s_face[:, :, None] * grad_c

    normals = jnp.asarray(mesh.face_normals)
    n_x = normals[:, 0]
    n_y = normals[:, 1]

    j_s_n = j_s[:, :, 0] * n_x[:, None] + j_s[:, :, 1] * n_y[:, None]

    # Stress tensor components
    du_dx = grad_u[:, 0]
    du_dy = grad_u[:, 1]
    dv_dx = grad_v[:, 0]
    dv_dy = grad_v[:, 1]
    div_u = du_dx + dv_dy

    lam = -2.0 / 3.0 * mu_face
    tau_xx = 2.0 * mu_face * du_dx + lam * div_u
    tau_yy = 2.0 * mu_face * dv_dy + lam * div_u
    tau_xy = mu_face * (du_dy + dv_dx)

    # Heat flux vectors
    q_tr = -(eta_t_face + eta_r_face)[:, None] * grad_T
    q_v = -eta_v_face[:, None] * grad_Tv

    q_tr_n = q_tr[:, 0] * n_x + q_tr[:, 1] * n_y
    q_v_n = q_v[:, 0] * n_x + q_v[:, 1] * n_y

    # Enthalpy and vibrational energy
    h_s = thermodynamic_relations.compute_equilibrium_enthalpy(
        T_face, equation_manager.species
    ).T
    e_v_s = thermodynamic_relations.compute_e_ve(Tv_face, equation_manager.species).T

    energy_diffusion = -jnp.sum(h_s * j_s_n, axis=-1)
    vib_energy_diffusion = -jnp.sum(e_v_s * j_s_n, axis=-1)

    tau_dot_n_x = tau_xx * n_x + tau_xy * n_y
    tau_dot_n_y = tau_xy * n_x + tau_yy * n_y

    tau_u = tau_xx * u_face + tau_xy * v_face
    tau_v = tau_xy * u_face + tau_yy * v_face
    tau_u_dot_n = tau_u * n_x + tau_v * n_y

    n_faces = normals.shape[0]
    F_v = jnp.zeros((n_faces, n_species + 4))
    F_v = F_v.at[:, :n_species].set(-j_s_n)
    F_v = F_v.at[:, n_species].set(-tau_dot_n_x)
    F_v = F_v.at[:, n_species + 1].set(-tau_dot_n_y)
    F_v = F_v.at[:, n_species + 2].set(-tau_u_dot_n + q_tr_n + q_v_n + energy_diffusion)
    F_v = F_v.at[:, n_species + 3].set(q_v_n + vib_energy_diffusion)

    return F_v
