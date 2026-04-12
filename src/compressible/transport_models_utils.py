"""Helpers for loading transport data and building transport models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import jax.numpy as jnp

from . import chemistry_types
from .transport_models_types import (
    CasseauTransportTable,
    CollisionIntegralTable,
    TransportModel,
    TransportModelConfig,
)


def load_collision_integrals_from_json(filepath: str | Path) -> dict:
    """Load raw collision-integral data from JSON."""
    with Path(filepath).open("r", encoding="utf-8") as file:
        return json.load(file)


def create_collision_integral_table_from_json(
    filepath: str | Path,
) -> CollisionIntegralTable:
    """Build a collision-integral table from JSON data."""
    data = load_collision_integrals_from_json(filepath)
    pairs = data["pairs"]

    species_pairs = tuple((pair["s"], pair["r"]) for pair in pairs)
    omega_11_2000K = jnp.array([pair["omega_11_2000"] for pair in pairs])
    omega_11_4000K = jnp.array([pair["omega_11_4000"] for pair in pairs])
    omega_22_2000K = jnp.array([pair["omega_22_2000"] for pair in pairs])
    omega_22_4000K = jnp.array([pair["omega_22_4000"] for pair in pairs])

    return CollisionIntegralTable(
        species_pairs=species_pairs,
        omega_11_2000K=omega_11_2000K,
        omega_11_4000K=omega_11_4000K,
        omega_22_2000K=omega_22_2000K,
        omega_22_4000K=omega_22_4000K,
    )


def load_casseau_transport_table(
    json_path: str | Path, species_names: Sequence[str]
) -> CasseauTransportTable:
    """Load Casseau transport coefficients for the requested species."""
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    entries = {entry["name"]: entry for entry in data["species"]}

    missing = [name for name in species_names if name not in entries]
    if missing:
        raise ValueError(f"Casseau transport data missing for species: {missing}")

    d_ref = []
    omega = []
    blottner_a = []
    blottner_b = []
    blottner_c = []

    for name in species_names:
        entry = entries[name]
        d_ref.append(float(entry["d_ref"]))
        omega.append(float(entry["omega"]))
        blottner_a.append(float(entry["blottner_A"]))
        blottner_b.append(float(entry["blottner_B"]))
        blottner_c.append(float(entry["blottner_C"]))

    return CasseauTransportTable(
        species_names=tuple(species_names),
        d_ref=jnp.array(d_ref),
        omega=jnp.array(omega),
        blottner_A=jnp.array(blottner_a),
        blottner_B=jnp.array(blottner_b),
        blottner_C=jnp.array(blottner_c),
        T_ref=float(data.get("T_ref", 273.0)),
    )


def build_transport_model_from_config(
    config: TransportModelConfig | None,
    *,
    species_table: chemistry_types.SpeciesTable,
) -> TransportModel:
    """Build the configured transport model."""
    if config is None:
        config = TransportModelConfig()

    collision_integrals = None
    if config.collision_integrals_path is not None:
        collision_integrals = create_collision_integral_table_from_json(
            config.collision_integrals_path
        )

    from . import transport_models

    model = config.model.lower()
    if model == "gnoffo":
        return transport_models.build_gnoffo_transport_model(
            species_table=species_table,
            collision_integrals=collision_integrals,
            include_diffusion=config.include_diffusion,
        )

    if model == "casseau":
        if config.casseau_data_path is None:
            raise ValueError("Casseau transport model requires casseau_data_path.")
        casseau_transport = load_casseau_transport_table(
            config.casseau_data_path, species_table.names
        )
        return transport_models.build_casseau_transport_model(
            species_table=species_table,
            casseau_transport=casseau_transport,
            collision_integrals=collision_integrals,
            include_diffusion=config.include_diffusion,
        )

    raise ValueError(f"Unknown transport model '{config.model}'.")
