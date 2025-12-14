"""Configuration classes for two-temperature model.

This module defines the configuration structures for the two-temperature
thermochemical nonequilibrium model, including species data, reaction data,
and overall model configuration.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class SpeciesData:
    """Thermodynamic and transport data for a chemical species.

    Attributes:
        name: Species name (e.g., "N2", "N")
        molecular_mass: Molecular mass [kg/mol]
        formation_enthalpy: Formation enthalpy at 298K [J/kg]
        theta_v: Characteristic vibrational temperature [K] (0 for atoms)
        theta_rot: Characteristic rotational temperature [K] (0 for atoms)
        theta_d: Characteristic dissociation temperature [K]
        collision_diameter: Collision diameter [m]
        nasa_low: NASA polynomial coefficients (low temp range, 7 coefficients)
        nasa_high: NASA polynomial coefficients (high temp range, 7 coefficients)
        temp_mid: Transition temperature between polynomial ranges [K]
    """

    name: str
    molecular_mass: float  # kg/mol
    formation_enthalpy: float  # J/kg
    theta_v: float = 0.0  # K
    theta_rot: float = 0.0  # K
    theta_d: float = 0.0  # K
    collision_diameter: float = 0.0  # m
    nasa_low: tuple[float, ...] = field(default_factory=lambda: (0.0,) * 7)
    nasa_high: tuple[float, ...] = field(default_factory=lambda: (0.0,) * 7)
    temp_mid: float = 1000.0  # K


@dataclass
class Reaction:
    """Chemical reaction data for kinetic modeling.

    For a reaction: Σ ν'_i A_i → Σ ν''_i A_i

    Attributes:
        name: Reaction identifier
        reactants: List of reactant species names
        products: List of product species names
        stoich_reactants: Stoichiometric coefficients for reactants
        stoich_products: Stoichiometric coefficients for products
        A: Pre-exponential factor [m^3/mol/s] or appropriate units
        n: Temperature exponent
        theta_d: Activation temperature [K]
        third_body_efficiencies: Dictionary of species:efficiency for third body
    """

    name: str
    reactants: list[str]
    products: list[str]
    stoich_reactants: list[float]
    stoich_products: list[float]
    A: float  # Pre-exponential factor
    n: float  # Temperature exponent
    theta_d: float  # Activation temperature [K]
    third_body_efficiencies: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class TwoTemperatureModelConfig:
    """Configuration for two-temperature thermochemical nonequilibrium model.

    This configuration class controls all aspects of the two-temperature model
    including species selection, chemical kinetics, transport properties, and
    numerical settings.

    Attributes:
        species_names: List of species in the simulation (e.g., ["N2", "N"])

        Chemical kinetics configuration:
        kinetics_model: Type of kinetic model ("park", "arrhenius")
        rate_data_source: Specific rate coefficient dataset ("park1993", etc.)
        controlling_temperature_mode: How to compute T_control for dissociation
            Options: "geometric_mean", "translational", "harmonic_mean"

        Transport properties configuration:
        diffusion_model: Diffusion model type ("mixture_averaged", "constant")
        constant_diffusion_coeff: Value if using constant diffusion [m^2/s]

        Vibrational relaxation configuration:
        relaxation_model: Vibrational relaxation model ("millikan_white_park")
        dissociation_model: Dissociation coupling ("preferential", "none")

        Numerical settings:
        chemistry_substeps: Number of subcycles for chemistry integration
        source_term_splitting: Operator splitting scheme ("strang", "godunov")

        Physical constants:
        gamma_mode: How to compute gamma ("mixture", "constant")
        constant_gamma: Value if using constant gamma
    """

    # Species configuration
    species_names: tuple[str, ...] = field(default_factory=lambda: ("N2", "N"))

    # Chemical kinetics
    kinetics_model: str = "park"
    rate_data_source: str = "park1993"
    controlling_temperature_mode: str = "geometric_mean"

    # Transport properties
    diffusion_model: str = "mixture_averaged"
    constant_diffusion_coeff: Optional[float] = None

    # Vibrational relaxation
    relaxation_model: str = "millikan_white_park"
    dissociation_model: str = "preferential"

    # Numerical settings
    chemistry_substeps: int = 10
    source_term_splitting: str = "strang"

    # Physical constants
    gamma_mode: str = "mixture"
    constant_gamma: Optional[float] = 1.4

    # Universal gas constant [J/(mol·K)]
    R_universal: float = 8.314462618

    def __post_init__(self):
        """Validate configuration after initialization."""
        valid_kinetics = ["park", "arrhenius"]
        if self.kinetics_model not in valid_kinetics:
            raise ValueError(
                f"kinetics_model must be one of {valid_kinetics}, "
                f"got {self.kinetics_model}"
            )

        valid_control_temp = ["geometric_mean", "translational", "harmonic_mean"]
        if self.controlling_temperature_mode not in valid_control_temp:
            raise ValueError(
                f"controlling_temperature_mode must be one of {valid_control_temp}, "
                f"got {self.controlling_temperature_mode}"
            )

        valid_diffusion = ["mixture_averaged", "constant"]
        if self.diffusion_model not in valid_diffusion:
            raise ValueError(
                f"diffusion_model must be one of {valid_diffusion}, "
                f"got {self.diffusion_model}"
            )

        if self.diffusion_model == "constant" and self.constant_diffusion_coeff is None:
            raise ValueError(
                "constant_diffusion_coeff must be specified when "
                "diffusion_model='constant'"
            )

        valid_relaxation = ["millikan_white_park", "none"]
        if self.relaxation_model not in valid_relaxation:
            raise ValueError(
                f"relaxation_model must be one of {valid_relaxation}, "
                f"got {self.relaxation_model}"
            )

        valid_dissociation = ["preferential", "none"]
        if self.dissociation_model not in valid_dissociation:
            raise ValueError(
                f"dissociation_model must be one of {valid_dissociation}, "
                f"got {self.dissociation_model}"
            )

        valid_splitting = ["strang", "godunov"]
        if self.source_term_splitting not in valid_splitting:
            raise ValueError(
                f"source_term_splitting must be one of {valid_splitting}, "
                f"got {self.source_term_splitting}"
            )

        valid_gamma_mode = ["mixture", "constant"]
        if self.gamma_mode not in valid_gamma_mode:
            raise ValueError(
                f"gamma_mode must be one of {valid_gamma_mode}, "
                f"got {self.gamma_mode}"
            )

        if self.gamma_mode == "constant" and self.constant_gamma is None:
            raise ValueError(
                "constant_gamma must be specified when gamma_mode='constant'"
            )

        if self.chemistry_substeps < 1:
            raise ValueError(
                f"chemistry_substeps must be >= 1, got {self.chemistry_substeps}"
            )

    @property
    def n_species(self) -> int:
        """Number of species in the simulation."""
        return len(self.species_names)

    @property
    def n_conserved(self) -> int:
        """Number of conserved variables.

        Returns n_species + 2 (momentum, total energy) + 1 (vibrational energy).
        """
        return self.n_species + 3
