from dataclasses import dataclass, field
from typing import Literal
import jax


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class ClippingConfig2D:
    """Clipping limits for primitive and conserved variables."""

    # Primitive variables
    rho_min: float = 1e-10
    rho_max: float = 1e10
    p_min: float = 1.0
    p_max: float = 1e10
    T_min: float = 100.0
    T_max: float = 50000.0
    Tv_min: float = 100.0
    Tv_max: float = 50000.0
    Y_min: float = 0.0
    Y_max: float = 1.0

    # Conserved variables
    rho_s_min: float = 1e-15
    rho_s_max: float = 1e10
    rho_u_min: float = -1e10
    rho_u_max: float = 1e10
    rho_v_min: float = -1e10
    rho_v_max: float = 1e10
    rho_E_min: float = 1e3
    rho_E_max: float = 1e12
    rho_Ev_min: float = 0.0
    rho_Ev_max: float = 1e12

    # Transport properties
    D_s_min: float = 0.0
    D_s_max: float = 1e2


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class NumericsConfig2D:
    dt: float | None = field(metadata=dict(static=True))
    cfl: float = field(default=0.4, metadata=dict(static=True))
    dt_mode: Literal["fixed", "cfl"] = field(
        default="fixed", metadata=dict(static=True)
    )
    integrator_scheme: Literal["forward-euler", "rk2"] = field(
        default="rk2", metadata=dict(static=True)
    )
    spatial_scheme: Literal["first_order"] = field(
        default="first_order", metadata=dict(static=True)
    )
    flux_scheme: Literal["hllc"] = field(default="hllc", metadata=dict(static=True))
    axisymmetric: bool = field(default=True, metadata=dict(static=True))
    clipping: ClippingConfig2D = field(default_factory=ClippingConfig2D)
