"""Viscous-flux helpers for the compressible solver."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from .mesh import Mesh
from .equation_manager_types import EquationManager
from . import state as state_module
from . import thermodynamic_relations


def _face_avg(
    phi_L: Float[Array, "n_faces ..."], phi_R: Float[Array, "n_faces ..."]
) -> Float[Array, "n_faces ..."]:
    """Average left and right face data."""
    return 0.5 * (phi_L + phi_R)


def _cell_gradient_scalar(
    phi_L: jnp.ndarray,
    phi_R: jnp.ndarray,
    mesh: Mesh,
) -> jnp.ndarray:
    """Green-Gauss cell gradient for a scalar field."""
    phi_face = _face_avg(phi_L, phi_R)
    normals = jnp.asarray(mesh.face_normals)
    areas = jnp.asarray(mesh.face_areas)
    face_left = jnp.asarray(mesh.face_left)
    face_right = jnp.asarray(mesh.face_right)
    n_cells = mesh.cell_areas.shape[0]

    contrib = phi_face[:, None] * normals * areas[:, None]
    grad = jnp.zeros((n_cells, 2))
    grad = grad.at[face_left].add(contrib)
    right_mask = face_right >= 0
    safe_r = jnp.where(right_mask, face_right, 0)
    grad = grad.at[safe_r].add(jnp.where(right_mask[:, None], -contrib, 0.0))
    return grad / jnp.asarray(mesh.cell_areas)[:, None]


def _cell_gradient_vector(
    phi_L: jnp.ndarray,
    phi_R: jnp.ndarray,
    mesh: Mesh,
) -> jnp.ndarray:
    """Green-Gauss cell gradient for a vector field (n_cells, n_species)."""
    phi_face = _face_avg(phi_L, phi_R)
    normals = jnp.asarray(mesh.face_normals)
    areas = jnp.asarray(mesh.face_areas)
    face_left = jnp.asarray(mesh.face_left)
    face_right = jnp.asarray(mesh.face_right)
    n_cells = mesh.cell_areas.shape[0]
    n_s = phi_face.shape[1]

    contrib = phi_face[:, :, None] * normals[:, None, :] * areas[:, None, None]
    grad = jnp.zeros((n_cells, n_s, 2))
    grad = grad.at[face_left].add(contrib)
    right_mask = face_right >= 0
    safe_r = jnp.where(right_mask, face_right, 0)
    grad = grad.at[safe_r].add(jnp.where(right_mask[:, None, None], -contrib, 0.0))
    return grad / jnp.asarray(mesh.cell_areas)[:, None, None]


def _face_gradient(grad_cell: jnp.ndarray, mesh: Mesh) -> jnp.ndarray:
    """Average of adjacent cell gradients; use left gradient at boundaries."""
    face_left = jnp.asarray(mesh.face_left)
    face_right = jnp.asarray(mesh.face_right)
    grad_L = grad_cell[face_left]
    extra_dims = (1,) * (grad_L.ndim - 1)
    mask = (face_right >= 0).reshape((-1,) + extra_dims)
    safe_r = jnp.where(face_right >= 0, face_right, face_left)
    grad_R = jnp.where(mask, grad_cell[safe_r], grad_L)
    return _face_avg(grad_L, grad_R)


def compute_viscous_flux_faces(
    U: Float[Array, "n_cells n_variables"],
    U_L: Float[Array, "n_faces n_variables"],
    U_R: Float[Array, "n_faces n_variables"],
    mesh: Mesh,
    equation_manager: EquationManager,
    cell_primitives: state_module.Primitives | None = None,
    face_primitives_L: state_module.Primitives | None = None,
    face_primitives_R: state_module.Primitives | None = None,
) -> Float[Array, "n_faces n_variables"]:
    """Compute viscous flux at every face.

    Returns zeros if transport_model is None (inviscid).

    Args:
        U: Cell-centered conserved state.
        U_L: Left face states (with ghost BCs applied).
        U_R: Right face states (with ghost BCs applied).
        mesh: Unified Mesh (1D or 2D).
        equation_manager: Physics and numerics configuration.
        cell_primitives: Pre-extracted cell primitives (computed if None).
        face_primitives_L: Pre-extracted left face primitives (computed if None).
        face_primitives_R: Pre-extracted right face primitives (computed if None).

    Returns:
        Viscous flux in Cartesian coordinates.
    """
    n_species = equation_manager.species.n_species
    if equation_manager.transport_model is None:
        return jnp.zeros_like(U_L)

    face_left = jnp.asarray(mesh.face_left)
    face_right = jnp.asarray(mesh.face_right)

    if cell_primitives is None:
        cell_primitives = state_module.extract_primitives_from_U(U, equation_manager)
    Y = cell_primitives.Y_s
    rho = cell_primitives.rho
    T = cell_primitives.T
    Tv = cell_primitives.Tv
    p = cell_primitives.p

    mu, eta_t, eta_r, eta_v, D_s = (
        equation_manager.transport_model.compute_transport_properties(T, Tv, p, Y, rho)
    )

    if face_primitives_L is None:
        face_primitives_L = state_module.extract_primitives_from_U(
            U_L, equation_manager
        )
    if face_primitives_R is None:
        face_primitives_R = state_module.extract_primitives_from_U(
            U_R, equation_manager
        )

    u_Lf = face_primitives_L.u
    v_Lf = face_primitives_L.v
    T_Lf = face_primitives_L.T
    Tv_Lf = face_primitives_L.Tv
    rho_Lf = face_primitives_L.rho

    u_Rf = face_primitives_R.u
    v_Rf = face_primitives_R.v
    T_Rf = face_primitives_R.T
    Tv_Rf = face_primitives_R.Tv
    rho_Rf = face_primitives_R.rho

    rho_s = U[:, :n_species]
    c_s = rho_s / rho[:, None]
    interior_mask = face_right >= 0
    safe_r = jnp.where(interior_mask, face_right, face_left)
    c_L = c_s[face_left]
    c_R_boundary = U_R[:, :n_species] / jnp.clip(rho_Rf[:, None], 1e-30, None)
    c_R = jnp.where(interior_mask[:, None], c_s[safe_r], c_R_boundary)

    # Direct face gradients: grad_phi[f] = (phi_R - phi_L) * (x_R - x_L) / |x_R - x_L|^2
    #
    # This is equivalent to compressible_1d's (phi[i+1] - phi[i]) / dx at each interface
    # and avoids the spurious gradient propagation of the two-step Green-Gauss approach
    # (cell gradient -> face average), which incorrectly spreads a large gradient at a
    # strong discontinuity into neighbouring faces between uniform cells.
    cell_cx = jnp.asarray(mesh.cell_centroids[:, 0])
    cell_cy = jnp.asarray(mesh.cell_centroids[:, 1])
    face_cx = jnp.asarray(mesh.face_centroids[:, 0])
    face_cy = jnp.asarray(mesh.face_centroids[:, 1])

    # L-to-R displacement: centroid difference for interior faces; for boundary faces
    # the ghost centroid is the mirror of the interior centroid across the face.
    dx_lr = jnp.where(
        interior_mask,
        cell_cx[safe_r] - cell_cx[face_left],
        2.0 * (face_cx - cell_cx[face_left]),
    )
    dy_lr = jnp.where(
        interior_mask,
        cell_cy[safe_r] - cell_cy[face_left],
        2.0 * (face_cy - cell_cy[face_left]),
    )
    inv_h2 = 1.0 / jnp.clip(dx_lr**2 + dy_lr**2, 1e-30)
    lr_over_h2 = jnp.stack([dx_lr * inv_h2, dy_lr * inv_h2], axis=1)  # (n_faces, 2)

    # Use cell-centred interior states for interior faces and the ghost state on
    # boundary faces so wall BCs directly affect shear and heat fluxes.
    T_cell_c = cell_primitives.T
    Tv_cell_c = cell_primitives.Tv
    u_cell_c = cell_primitives.u
    v_cell_c = cell_primitives.v

    T_L_g = T_cell_c[face_left]
    Tv_L_g = Tv_cell_c[face_left]
    u_L_g = u_cell_c[face_left]
    v_L_g = v_cell_c[face_left]
    T_R_g = jnp.where(interior_mask, T_cell_c[safe_r], T_Rf)
    Tv_R_g = jnp.where(interior_mask, Tv_cell_c[safe_r], Tv_Rf)
    u_R_g = jnp.where(interior_mask, u_cell_c[safe_r], u_Rf)
    v_R_g = jnp.where(interior_mask, v_cell_c[safe_r], v_Rf)

    grad_u = (u_R_g - u_L_g)[:, None] * lr_over_h2
    grad_v = (v_R_g - v_L_g)[:, None] * lr_over_h2
    grad_T = (T_R_g - T_L_g)[:, None] * lr_over_h2
    grad_Tv = (Tv_R_g - Tv_L_g)[:, None] * lr_over_h2
    grad_c = (c_R - c_L)[:, :, None] * lr_over_h2[:, None, :]

    def _cell_to_face(arr):
        arr_L = arr[face_left]
        arr_R = jnp.where(interior_mask, arr[safe_r], arr_L)
        return _face_avg(arr_L, arr_R)

    mu_f = _cell_to_face(mu)
    eta_t_f = _cell_to_face(eta_t)
    eta_r_f = _cell_to_face(eta_r)
    eta_v_f = _cell_to_face(eta_v)
    D_s_f = _face_avg(
        D_s[face_left],
        jnp.where(interior_mask[:, None], D_s[safe_r], D_s[face_left]),
    )

    rho_f = _face_avg(rho_Lf, rho_Rf)
    u_f = _face_avg(u_Lf, u_Rf)
    v_f = _face_avg(v_Lf, v_Rf)
    T_f = _face_avg(T_Lf, T_Rf)
    Tv_f = _face_avg(Tv_Lf, Tv_Rf)

    clip = equation_manager.numerics_config.clipping
    D_s_f = jnp.clip(D_s_f, clip.D_s_min, clip.D_s_max)

    # Species diffusion flux (vector per species)
    j_s = -rho_f[:, None, None] * D_s_f[:, :, None] * grad_c

    normals = jnp.asarray(mesh.face_normals)
    n_x = normals[:, 0]
    n_y = normals[:, 1]

    j_s_n = j_s[:, :, 0] * n_x[:, None] + j_s[:, :, 1] * n_y[:, None]

    # Stress tensor
    du_dx = grad_u[:, 0]
    du_dy = grad_u[:, 1]
    dv_dx = grad_v[:, 0]
    dv_dy = grad_v[:, 1]
    div_u = du_dx + dv_dy

    lam = -2.0 / 3.0 * mu_f
    tau_xx = 2.0 * mu_f * du_dx + lam * div_u
    tau_yy = 2.0 * mu_f * dv_dy + lam * div_u
    tau_xy = mu_f * (du_dy + dv_dx)

    # Heat fluxes
    q_tr = -(eta_t_f + eta_r_f)[:, None] * grad_T
    q_v = -eta_v_f[:, None] * grad_Tv

    q_tr_n = q_tr[:, 0] * n_x + q_tr[:, 1] * n_y
    q_v_n = q_v[:, 0] * n_x + q_v[:, 1] * n_y

    h_s = thermodynamic_relations.compute_equilibrium_enthalpy(
        T_f, equation_manager.species
    ).T
    e_v_s = thermodynamic_relations.compute_e_ve(Tv_f, equation_manager.species).T

    energy_diffusion = -jnp.sum(h_s * j_s_n, axis=-1)
    vib_energy_diffusion = -jnp.sum(e_v_s * j_s_n, axis=-1)

    tau_dot_n_x = tau_xx * n_x + tau_xy * n_y
    tau_dot_n_y = tau_xy * n_x + tau_yy * n_y
    tau_u_dot_n = (tau_xx * u_f + tau_xy * v_f) * n_x + (
        tau_xy * u_f + tau_yy * v_f
    ) * n_y

    n_faces = normals.shape[0]
    F_v = jnp.zeros((n_faces, n_species + 4))
    F_v = F_v.at[:, :n_species].set(-j_s_n)
    F_v = F_v.at[:, n_species].set(-tau_dot_n_x)
    F_v = F_v.at[:, n_species + 1].set(-tau_dot_n_y)
    F_v = F_v.at[:, n_species + 2].set(-tau_u_dot_n + q_tr_n + q_v_n + energy_diffusion)
    F_v = F_v.at[:, n_species + 3].set(q_v_n + vib_energy_diffusion)
    return F_v
