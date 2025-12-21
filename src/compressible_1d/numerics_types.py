from typing import Literal


class NumericsConfig:
    dt: float
    dx: float
    integrator_scheme: Literal["forward-euler", "rk2"]
    spatial_scheme: Literal["first_order", "muscl"]
    flux_scheme: Literal["lax_friedrichs", "hllc"]
    n_halo_cells: int  # per side
