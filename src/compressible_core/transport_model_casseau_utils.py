"""I/O helpers for Casseau transport data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import jax.numpy as jnp

from compressible_core.transport_casseau_types import CasseauTransportTable


def load_casseau_transport_table(
    json_path: str | Path, species_names: Sequence[str]
) -> CasseauTransportTable:
    """Load Casseau transport coefficients for selected species."""
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    entries = {entry["name"]: entry for entry in data["species"]}

    missing = [name for name in species_names if name not in entries]
    if missing:
        raise ValueError(f"Casseau transport data missing for species: {missing}")

    d_ref = []
    omega = []
    A = []
    B = []
    C = []
    for name in species_names:
        entry = entries[name]
        d_ref.append(float(entry["d_ref"]))
        omega.append(float(entry["omega"]))
        A.append(float(entry["blottner_A"]))
        B.append(float(entry["blottner_B"]))
        C.append(float(entry["blottner_C"]))

    return CasseauTransportTable(
        species_names=tuple(species_names),
        d_ref=jnp.array(d_ref),
        omega=jnp.array(omega),
        blottner_A=jnp.array(A),
        blottner_B=jnp.array(B),
        blottner_C=jnp.array(C),
        T_ref=float(data.get("T_ref", 273.0)),
    )
