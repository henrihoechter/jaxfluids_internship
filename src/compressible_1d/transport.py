"""Transport property calculations for two-temperature model.

This module implements transport property calculations following NASA TP-2867
(Gnoffo et al., 1989). The approach uses collision integrals tabulated at
reference temperatures and interpolated for arbitrary temperatures.

Key equations from TP-2867:
- Eq. 67: Collision integral interpolation
- Eq. 69-70: Modified collision integrals Δ^(1) and Δ^(2)
- Eq. 72: Mixture viscosity
- Eq. 73: Translational thermal conductivity
- Eq. 75: Rotational thermal conductivity
- Eq. 77: Vibrational thermal conductivity
- Eq. 79: Binary diffusion coefficient
- Eq. 81: Effective mixture diffusion
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import json
from pathlib import Path

import jax.numpy as jnp
from jaxtyping import Array, Float

from compressible_1d import constants

if TYPE_CHECKING:
    from compressible_1d.chemistry_types import CollisionIntegralTable


# Physical constants
K_BOLTZMANN = 1.380649e-23  # Boltzmann constant [J/K]
N_AVOGADRO = 6.02214076e23  # Avogadro's number [1/mol]
R_UNIVERSAL = constants.R_universal  # Universal gas constant [J/(mol·K)]

# Reference temperatures for collision integral interpolation
T_REF_LOW = 2000.0  # [K]
T_REF_HIGH = 4000.0  # [K]
LN_T_REF_LOW = jnp.log(T_REF_LOW)
LN_T_REF_HIGH = jnp.log(T_REF_HIGH)


def interpolate_collision_integral(
    T: Float[Array, "..."],
    omega_2000K: Float[Array, " n_pairs"],
    omega_4000K: Float[Array, " n_pairs"],
) -> Float[Array, "... n_pairs"]:
    """Interpolate collision integral at temperature T.

    Uses linear interpolation in ln(T) space (Eq. 67 from TP-2867):
        log10(pi_Omega(T)) = log10(pi_Omega(2000)) + slope × [ln(T) - ln(2000)]
    where:
        slope = [log10(pi_Omega(4000)) - log10(pi_Omega(2000))] / [ln(4000) - ln(2000)]

    Args:
        T: Temperature [K], shape (...)
        omega_2000K: log10(pi_Omega) at 2000K, shape (n_pairs,)
        omega_4000K: log10(pi_Omega) at 4000K, shape (n_pairs,)

    Returns:
        pi_Omega(T) in cm², shape (..., n_pairs)
    """
    # Compute slope for interpolation
    slope = (omega_4000K - omega_2000K) / (LN_T_REF_HIGH - LN_T_REF_LOW)

    # Interpolate in log space
    ln_T = jnp.log(jnp.clip(T, 300.0, 50000.0))

    # Broadcast for arbitrary T shape
    # omega_2000K: (n_pairs,), slope: (n_pairs,), ln_T: (...)
    log10_omega = omega_2000K + slope * (ln_T[..., None] - LN_T_REF_LOW)

    # Convert from log10 to linear scale
    pi_omega = jnp.power(10.0, log10_omega)

    return pi_omega  # [cm²]


def compute_modified_collision_integral_1(
    T: Float[Array, "..."],
    M_s: Float[Array, " n_species"],
    M_r: Float[Array, " n_species"],
    pi_omega_11: Float[Array, "... n_pairs"],
    pair_indices_sr: Float[Array, "n_species n_species"],
) -> Float[Array, "... n_species n_species"]:
    """Compute modified collision integral Delta1_sr(T).

    Eq. 69 from TP-2867:
        Delta1_sr(T) = (8/3) × [2M_s·M_r / (pi·R·T·(M_s+M_r))]^(1/2) × pi_Omega_11_sr

    Args:
        T: Temperature [K], shape (...)
        M_s: Molar masses [kg/kmol], shape (n_species,)
        M_r: Molar masses [kg/kmol], shape (n_species,)
        pi_omega_11: pi_Omega^(1,1) in cm², shape (..., n_pairs)
        pair_indices_sr: Indices into pi_omega_11 for each (s,r) pair, shape (n_species, n_species)

    Returns:
        Delta1_sr(T) in appropriate units, shape (..., n_species, n_species)
    """
    n_species = M_s.shape[0]

    # Convert molar masses from kg/kmol to kg/mol
    M_s_mol = M_s / 1000.0  # [kg/mol]
    M_r_mol = M_r / 1000.0  # [kg/mol]

    # Compute mass factor for each pair
    # [2M_s·M_r / (pi·R·(M_s+M_r))]^(1/2)
    M_s_grid, M_r_grid = jnp.meshgrid(M_s_mol, M_r_mol, indexing="ij")
    mass_factor = jnp.sqrt(
        2.0
        * M_s_grid
        * M_r_grid
        / (jnp.pi * constants.R_universal * (M_s_grid + M_r_grid))
    )
    # mass_factor shape: (n_species, n_species)

    # Get collision integrals for each pair
    # pair_indices_sr shape: (n_species, n_species), values are indices into pi_omega_11
    pair_indices_int = pair_indices_sr.astype(jnp.int32)

    # pi_omega_11 shape: (..., n_pairs)
    # We need to gather along the last axis using pair_indices
    # Result shape should be (..., n_species, n_species)
    pi_omega_sr = pi_omega_11[..., pair_indices_int]  # (..., n_species, n_species)

    # Convert pi_Omega from cm² to m²
    pi_omega_sr_m2 = pi_omega_sr * 1e-4  # [m²]

    # Temperature factor: T^(-1/2)
    T_factor = 1.0 / jnp.sqrt(T[..., None, None])  # (..., 1, 1) for broadcasting

    # Compute Delta1_sr
    delta_1 = (8.0 / 3.0) * mass_factor * T_factor * pi_omega_sr_m2

    return delta_1


def compute_modified_collision_integral_2(
    T: Float[Array, "..."],
    M_s: Float[Array, " n_species"],
    M_r: Float[Array, " n_species"],
    pi_omega_22: Float[Array, "... n_pairs"],
    pair_indices_sr: Float[Array, "n_species n_species"],
) -> Float[Array, "... n_species n_species"]:
    """Compute modified collision integral Δ^(2)_sr(T).

    Eq. 70 from TP-2867:
        Δ^(2)_sr(T) = (16/5) × [2M_s·M_r / (πRT(M_s+M_r))]^(1/2) × πΩ^(2,2)_sr

    Args:
        T: Temperature [K], shape (...)
        M_s: Molar masses [kg/kmol], shape (n_species,)
        M_r: Molar masses [kg/kmol], shape (n_species,)
        pi_omega_22: πΩ^(2,2) in cm², shape (..., n_pairs)
        pair_indices_sr: Indices into pi_omega_22 for each (s,r) pair, shape (n_species, n_species)

    Returns:
        Δ^(2)_sr(T) in appropriate units, shape (..., n_species, n_species)
    """
    n_species = M_s.shape[0]

    # Convert molar masses from kg/kmol to kg/mol
    M_s_mol = M_s / 1000.0  # [kg/mol]
    M_r_mol = M_r / 1000.0  # [kg/mol]

    # Compute mass factor for each pair
    M_s_grid, M_r_grid = jnp.meshgrid(M_s_mol, M_r_mol, indexing="ij")
    mass_factor = jnp.sqrt(
        2.0 * M_s_grid * M_r_grid / (jnp.pi * R_UNIVERSAL * (M_s_grid + M_r_grid))
    )

    # Get collision integrals for each pair
    pair_indices_int = pair_indices_sr.astype(jnp.int32)
    pi_omega_sr = pi_omega_22[..., pair_indices_int]

    # Convert πΩ from cm² to m²
    pi_omega_sr_m2 = pi_omega_sr * 1e-4  # [m²]

    # Temperature factor
    T_factor = 1.0 / jnp.sqrt(T[..., None, None])

    # Compute Δ^(2)_sr
    delta_2 = (16.0 / 5.0) * mass_factor * T_factor * pi_omega_sr_m2

    return delta_2


def build_pair_index_matrix(
    species_names: tuple[str, ...],
    collision_integrals: CollisionIntegralTable,
) -> Float[Array, "n_species n_species"]:
    """Build matrix mapping (s,r) species pairs to collision integral indices.

    Args:
        species_names: Tuple of species names
        collision_integrals: CollisionIntegralTable with species pairs

    Returns:
        Matrix of indices, shape (n_species, n_species)
    """
    n_species = len(species_names)
    indices = jnp.zeros((n_species, n_species), dtype=jnp.int32)

    for i, s in enumerate(species_names):
        for j, r in enumerate(species_names):
            try:
                idx = collision_integrals.get_pair_index(s, r)
                indices = indices.at[i, j].set(idx)
            except ValueError:
                # If pair not found, use diagonal self-collision as fallback
                try:
                    idx = collision_integrals.get_pair_index(s, s)
                    indices = indices.at[i, j].set(idx)
                except ValueError:
                    # Last resort: use index 0
                    indices = indices.at[i, j].set(0)

    return indices


def compute_mixture_viscosity(
    T: Float[Array, " n_cells"],
    gamma_s: Float[Array, "n_cells n_species"],
    M_s: Float[Array, " n_species"],
    delta_2_sr: Float[Array, "n_cells n_species n_species"],
) -> Float[Array, " n_cells"]:
    """Compute mixture dynamic viscosity.

    Eq. 72 from TP-2867 (simplified for heavy particles only):
        mu = sum_s (m_s * gamma_s) / [sum_r gamma_r * Delta2_sr(T)]

    where m_s is the molecular mass and gamma_s is the molar concentration.

    Args:
        T: Temperature [K], shape (n_cells,)
        gamma_s: Molar concentrations gamma_s = rho_s/(rho * M_s), shape (n_cells, n_species)
        M_s: Molar masses [kg/kmol], shape (n_species,)
        delta_2_sr: Modified collision integral Delta2, shape (n_cells, n_species, n_species)

    Returns:
        mu: Dynamic viscosity [Pa·s], shape (n_cells,)
    """
    # Molecular mass [kg/molecule]
    m_s = M_s / (1000.0 * constants.N_A)  # [kg/molecule]

    # Numerator: sum_s (m_s * gamma_s)
    numerator = jnp.sum(m_s * gamma_s, axis=-1)  # (n_cells,)

    # Denominator for each species s: sum_r gamma_r * Delta2_sr
    # delta_2_sr: (n_cells, n_species, n_species)
    # gamma_s: (n_cells, n_species)
    # We want: sum over r of gamma_r * delta_2_sr for each s
    denominator_per_s = jnp.einsum(
        "cr,csr->cs", gamma_s, delta_2_sr
    )  # (n_cells, n_species)

    # Sum contributions from all species
    # For each s, contribution is (m_s * gamma_s) / denominator_per_s
    # Total viscosity is sum of these contributions
    viscosity = jnp.sum(
        m_s * gamma_s / jnp.clip(denominator_per_s, 1e-30, None), axis=-1
    )

    return viscosity


def compute_translational_thermal_conductivity(
    T: Float[Array, " n_cells"],
    gamma_s: Float[Array, "n_cells n_species"],
    M_s: Float[Array, " n_species"],
    delta_2_sr: Float[Array, "n_cells n_species n_species"],
) -> Float[Array, " n_cells"]:
    """Compute translational thermal conductivity eta_t.

    Eq. 73 from TP-2867 (simplified for heavy particles only):
        eta_t = (15k/4) · sum_s gamma_s / [sum_r a_sr·gamma_r·Delta2_sr(T)]

    where a_sr is defined by Eq. 74.

    Args:
        T: Temperature [K], shape (n_cells,)
        gamma_s: Molar concentrations, shape (n_cells, n_species)
        M_s: Molar masses [kg/kmol], shape (n_species,)
        delta_2_sr: Modified collision integral Delta2, shape (n_cells, n_species, n_species)

    Returns:
        eta_t: Translational thermal conductivity [W/(m·K)], shape (n_cells,)
    """
    n_species = M_s.shape[0]

    # Compute a_sr factor (Eq. 74)
    # a_sr = 1 + [1 - (m_s/m_r)][0.45 - 2.54(m_s/m_r)] / [1 + (m_s/m_r)]²
    M_s_grid, M_r_grid = jnp.meshgrid(M_s, M_s, indexing="ij")
    m_ratio = M_s_grid / M_r_grid

    a_sr = 1.0 + (1.0 - m_ratio) * (0.45 - 2.54 * m_ratio) / jnp.square(1.0 + m_ratio)
    # a_sr shape: (n_species, n_species)

    # Coefficient
    coeff = 15.0 * constants.k / 4.0

    # Denominator for each species s: sum_r a_sr·gamma_r·Delta2_sr
    # a_sr: (n_species, n_species)
    # delta_2_sr: (n_cells, n_species, n_species)
    # gamma_s: (n_cells, n_species)
    weighted_delta = a_sr[None, :, :] * delta_2_sr  # (n_cells, n_species, n_species)
    denominator_per_s = jnp.einsum(
        "cr,csr->cs", gamma_s, weighted_delta
    )  # (n_cells, n_species)

    # Sum contributions
    eta_t = coeff * jnp.sum(gamma_s / jnp.clip(denominator_per_s, 1e-30, None), axis=-1)

    return eta_t


def compute_rotational_thermal_conductivity(
    T: Float[Array, " n_cells"],
    gamma_s: Float[Array, "n_cells n_species"],
    is_molecule: Float[Array, " n_species"],
    delta_1_sr: Float[Array, "n_cells n_species n_species"],
) -> Float[Array, " n_cells"]:
    """Compute rotational thermal conductivity eta_r.

    Eq. 75 from TP-2867:
        eta_r = k · sum_{s=mol} gamma_s / [sum_r gamma_r·Delta1_sr(T)]

    Only molecular species contribute to rotational conductivity.

    Args:
        T: Temperature [K], shape (n_cells,)
        gamma_s: Molar concentrations, shape (n_cells, n_species)
        is_molecule: Boolean mask (True for molecules), shape (n_species,)
        delta_1_sr: Modified collision integral Delta1, shape (n_cells, n_species, n_species)

    Returns:
        eta_r: Rotational thermal conductivity [W/(m·K)], shape (n_cells,)
    """
    # Denominator for each species s: sum_r gamma_r·Delta1_sr
    denominator_per_s = jnp.einsum(
        "cr,csr->cs", gamma_s, delta_1_sr
    )  # (n_cells, n_species)

    # Mask to only include molecular species
    mol_mask = is_molecule[None, :]  # (1, n_species)

    # Sum contributions from molecular species only
    eta_r = constants.k * jnp.sum(
        mol_mask * gamma_s / jnp.clip(denominator_per_s, 1e-30, None), axis=-1
    )

    return eta_r


def compute_vibrational_thermal_conductivity(
    T_v: Float[Array, " n_cells"],
    gamma_s: Float[Array, "n_cells n_species"],
    is_molecule: Float[Array, " n_species"],
    delta_1_sr: Float[Array, "n_cells n_species n_species"],
) -> Float[Array, " n_cells"]:
    """Compute vibrational thermal conductivity η_v.

    Eq. 77 from TP-2867:
        η_v = η_r (same form as rotational)

    Args:
        T_v: Vibrational temperature [K], shape (n_cells,)
        gamma_s: Molar concentrations, shape (n_cells, n_species)
        is_molecule: Boolean mask (True for molecules), shape (n_species,)
        delta_1_sr: Modified collision integral Δ^(1) at T_v, shape (n_cells, n_species, n_species)

    Returns:
        η_v: Vibrational thermal conductivity [W/(m·K)], shape (n_cells,)
    """
    # Same formula as rotational conductivity
    return compute_rotational_thermal_conductivity(
        T_v, gamma_s, is_molecule, delta_1_sr
    )


def compute_binary_diffusion_coefficient(
    T: Float[Array, " n_cells"],
    p: Float[Array, " n_cells"],
    delta_1_sr: Float[Array, "n_cells n_species n_species"],
) -> Float[Array, "n_cells n_species n_species"]:
    """Compute binary diffusion coefficients D_sr.

    Eq. 79 from TP-2867:
        D_sr = kT / (p·Delta1_sr(T))

    Args:
        T: Temperature [K], shape (n_cells,)
        p: Pressure [Pa], shape (n_cells,)
        delta_1_sr: Modified collision integral Delta1, shape (n_cells, n_species, n_species)

    Returns:
        D_sr: Binary diffusion coefficients [m²/s], shape (n_cells, n_species, n_species)
    """
    D_sr = (
        constants.k
        * T[:, None, None]
        / (p[:, None, None] * jnp.clip(delta_1_sr, 1e-30, None))
    )

    return D_sr


def compute_effective_diffusion_coefficient(
    gamma_s: Float[Array, "n_cells n_species"],
    M_s: Float[Array, " n_species"],
    D_sr: Float[Array, "n_cells n_species n_species"],
) -> Float[Array, "n_cells n_species"]:
    """Compute effective mixture diffusion coefficients D_s.

    Eq. 81 from TP-2867:
        D_s = gamma_t^2 * M_s * (1 - M_s * gamma_s) / sum_{r!=s} (gamma_r / D_sr)

    where gamma_t = sum_s gamma_s is the total molar concentration.

    Args:
        gamma_s: Molar concentrations, shape (n_cells, n_species)
        M_s: Molar masses [kg/kmol], shape (n_species,)
        D_sr: Binary diffusion coefficients [m²/s], shape (n_cells, n_species, n_species)

    Returns:
        D_s: Effective diffusion coefficients [m²/s], shape (n_cells, n_species)
    """
    n_cells, n_species = gamma_s.shape

    # Total molar concentration
    gamma_t = jnp.sum(gamma_s, axis=-1, keepdims=True)  # (n_cells, 1)

    # Convert molar masses to kg/mol for consistency
    M_s_mol = M_s / 1000.0  # [kg/mol]
    # Numerator: gamma_t^2·M_s·(1 - M_s·gamma_s)
    numerator = (
        jnp.square(gamma_t) * M_s_mol * (1.0 - M_s_mol * gamma_s)
    )  # (n_cells, n_species)

    # Denominator: Σ_{r≠s} (gamma_r/D_sr)
    # We need to sum gamma_r/D_sr over r, excluding r=s
    # D_sr: (n_cells, n_species, n_species)
    # gamma_s: (n_cells, n_species)

    # Create mask for r≠s
    eye = jnp.eye(n_species)
    off_diag_mask = 1.0 - eye  # (n_species, n_species)

    # γ_r / D_sr for all r,s
    gamma_over_D = gamma_s[:, None, :] / jnp.clip(
        D_sr, 1e-30, None
    )  # (n_cells, n_species, n_species)

    # Apply mask and sum over r (last axis)
    denominator = jnp.sum(off_diag_mask * gamma_over_D, axis=-1)  # (n_cells, n_species)

    # Effective diffusion coefficient
    D_s = numerator / jnp.clip(denominator, 1e-30, None)

    return D_s


def load_collision_integrals_from_json(filepath: str | Path) -> dict:
    """Load collision integral data from JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        Dictionary with collision integral data
    """
    with open(filepath, "r") as f:
        data = json.load(f)
    return data


def create_collision_integral_table_from_json(filepath: str | Path):
    """Create CollisionIntegralTable from JSON file.

    Args:
        filepath: Path to collision integrals JSON file

    Returns:
        CollisionIntegralTable instance
    """
    from compressible_1d.chemistry_types import CollisionIntegralTable

    data = load_collision_integrals_from_json(filepath)
    pairs = data["pairs"]

    species_pairs = tuple((p["s"], p["r"]) for p in pairs)
    omega_11_2000K = jnp.array([p["omega_11_2000"] for p in pairs])
    omega_11_4000K = jnp.array([p["omega_11_4000"] for p in pairs])
    omega_22_2000K = jnp.array([p["omega_22_2000"] for p in pairs])
    omega_22_4000K = jnp.array([p["omega_22_4000"] for p in pairs])

    return CollisionIntegralTable(
        species_pairs=species_pairs,
        omega_11_2000K=omega_11_2000K,
        omega_11_4000K=omega_11_4000K,
        omega_22_2000K=omega_22_2000K,
        omega_22_4000K=omega_22_4000K,
    )
