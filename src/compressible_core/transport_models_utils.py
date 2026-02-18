from __future__ import annotations

import functools

from compressible_core import chemistry_types
from compressible_core.transport_casseau_types import CasseauTransportTable
from compressible_core.transport_models_types import TransportModel, TransportModelConfig


def build_transport_model_from_config(
    config: TransportModelConfig | None,
    *,
    species_table: chemistry_types.SpeciesTable,
    collision_integrals: chemistry_types.CollisionIntegralTable | None,
    casseau_transport: CasseauTransportTable | None = None,
) -> TransportModel:
    """Build a transport model from a configuration object."""
    if config is None:
        config = TransportModelConfig()

    from compressible_core import transport_models

    model = config.model.lower()
    if model == "gnoffo":
        compute_transport_properties = functools.partial(
            transport_models.compute_transport_properties_gnoffo,
            species_table=species_table,
            collision_integrals=collision_integrals,
            include_diffusion=config.include_diffusion,
        )
        return TransportModel(compute_transport_properties=compute_transport_properties)

    if model == "casseau":
        if casseau_transport is None:
            raise ValueError("Casseau transport selected but no data provided.")
        compute_transport_properties = functools.partial(
            transport_models.compute_transport_properties_casseau,
            species_table=species_table,
            casseau_transport=casseau_transport,
            collision_integrals=collision_integrals,
            include_diffusion=config.include_diffusion,
        )
        return TransportModel(compute_transport_properties=compute_transport_properties)

    raise ValueError(f"Unknown transport model '{config.model}'.")
