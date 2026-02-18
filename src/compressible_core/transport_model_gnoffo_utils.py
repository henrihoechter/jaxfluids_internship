"""I/O helpers for Gnoffo (TP-2867) transport data."""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp

from compressible_core.chemistry_types import CollisionIntegralTable


def load_collision_integrals_from_json(filepath: str | Path) -> dict:
    """Load collision integral data from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def create_collision_integral_table_from_json(
    filepath: str | Path,
) -> CollisionIntegralTable:
    """Create CollisionIntegralTable from JSON file."""
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
