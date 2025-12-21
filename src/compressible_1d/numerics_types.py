from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True, slots=True)
class ClippingConfig:
    """Clipping limits for primitive and conserved variables.

    Always instantiated to avoid runtime checks. Set min=-inf and max=inf
    to effectively disable clipping for specific variables.
    """
    # Primitive variables
    rho_min: float = 1e-10
    rho_max: float = 1e10
    p_min: float = 1.0
    p_max: float = 1e10
    T_min: float = 100.0
    T_max: float = 50000.0
    Tv_min: float = 100.0
    Tv_max: float = 50000.0
    Y_min: float = 0.0  # Mass fractions
    Y_max: float = 1.0

    # Conserved variables
    rho_s_min: float = 1e-15  # Partial densities
    rho_s_max: float = 1e10
    rho_u_min: float = -1e10  # Momentum
    rho_u_max: float = 1e10
    rho_E_min: float = 1e3  # Total energy per volume
    rho_E_max: float = 1e12
    rho_Ev_min: float = 0.0  # Vibrational energy per volume
    rho_Ev_max: float = 1e12


class NumericsConfig:
    dt: float
    dx: float
    integrator_scheme: Literal["forward-euler", "rk2"]
    spatial_scheme: Literal["first_order", "muscl"]
    flux_scheme: Literal["lax_friedrichs", "hllc"]
    n_halo_cells: int  # per side
    clipping: ClippingConfig = field(default_factory=ClippingConfig)
