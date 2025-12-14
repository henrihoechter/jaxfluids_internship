"""Two-temperature thermochemical nonequilibrium model for hypersonic flows.

This package implements the two-temperature (T, Tv) model based on NASA TP-2867
(Gnoffo et al., 1989) for simulating hypersonic flows with thermal and chemical
nonequilibrium.

Key components:
- config: Configuration classes for model setup
- thermodynamics: Species properties and thermodynamic calculations
- transport: Diffusion and thermal conductivity models
- kinetics: Chemical kinetics (Park's model)
- relaxation: Vibrational relaxation (Millikan-White with Park's correction)
- source_terms: Chemistry and energy exchange source term integration
- solver_adapter: Integration with finite volume solver
"""

from compressible_1d.two_temperature.config import (
    TwoTemperatureModelConfig,
    SpeciesData,
    Reaction,
)

__all__ = [
    "TwoTemperatureModelConfig",
    "SpeciesData",
    "Reaction",
]
