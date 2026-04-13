# jaxfluids_internship

JAX-based compressible flow solver with support for thermochemical nonequilibrium, targeting hypersonic reentry flows. Built on top of [JaxFluids](https://github.com/tumaer/JAXFLUIDS).

## Setup

### Requirements

- Python 3.11

### Installation

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Project Structure

```
src/compressible/              # core solver package
  solver.py                    # flux computation and time integration
  mesh.py                      # unstructured mesh handling
  state.py                     # state vector and primitive/conserved conversions
  equation_manager_types.py    # equation set configuration (species, BC, numerics)
  boundary_conditions*.py      # boundary condition types and application
  thermodynamic_relations.py   # cp, cv, enthalpy, mixture properties
  energy_models*.py            # thermal energy models (single-T, two-T)
  chemistry*.py                # reaction rates, equilibrium constants, source terms
  transport_model_gnoffo*.py   # Gnoffo viscosity/conductivity model
  transport_model_casseau*.py  # Casseau multi-temperature transport model
  source_terms.py              # chemistry and vibrational relaxation source terms
  viscous_flux.py              # viscous flux assembly
  diagnose.py                  # NaN diagnostics

data/                          # JSON parameter files
  species.json                 # species thermodynamic properties
  air_5_gnoffo.json            # 5-species air model (Gnoffo)
  air_5_casseau_transport.json # 5-species air transport (Casseau)
  park_1990_reactions.json     # Park (1990) reaction set
  park_1993_reactions.json     # Park (1993) reaction set
  scanlon_table2_reactions.json
  collision_integrals_tp2867.json
  casseau_qk_reactions.json
  bluntedCone.msh              # 2D blunt cone mesh

experiments/                   # verification and application cases
  heatbath_0d_casseau/         # 0D thermal relaxation — Casseau verification
  heatbath_0d_williams/        # 0D thermal relaxation — Williams verification
  shock_tube_1d_toro/          # 1D Euler shock tube — Toro verification
  hypersonic_shock_tube_1d/    # 1D shock tube with chemistry — Williams verification

  blunt_cone_2d/               # 2D hypersonic blunt cone reentry vehicle
  flat_plate_2d/               # 2D flat plate boundary layer
  debug/                       # curve fit and reaction dataset notebooks
```

## Running an Example

```bash
python3 ./experiments/example.py
```

## Verification Notebooks

Each case directory contains a Jupyter notebook used to set up and validate the case against reference data:

| Notebook | Case | Reference |
|---|---|---|
| [heatbath_0d_casseau.ipynb](experiments/heatbath_0d_casseau/heatbath_0d_casseau.ipynb) | 0D thermal relaxation | Casseau |
| [heatbath_0d_williams.ipynb](experiments/heatbath_0d_williams/heatbath_0d_williams.ipynb) | 0D thermal relaxation | Williams |
| [shock_tube_1d.ipynb](experiments/shock_tube_1d_toro/shock_tube_1d.ipynb) | 1D Euler shock tube | Toro |
| [shock_tube_1d.ipynb](experiments/hypersonic_shock_tube_1d/shock_tube_1d.ipynb) | 1D hypersonic shock tube | Williams |
| [plot_reentry_vehicle.ipynb](experiments/blunt_cone_2d/plot_reentry_vehicle.ipynb) | 2D hypersonic (inert) blunt cone | Casseau |

## References

- **Gnoffo, P.A., Gupta, R.N., Shinn, J.L.** — *Conservation Equations and Physical Models for Hypersonic Air Flows in Thermal and Chemical Nonequilibrium*, NASA Technical Publication NASA-TP-2867, NASA Langley Research Center, Hampton, Virginia, February 1989.

- **Park, C.** — *Nonequilibrium Hypersonic Aerothermodynamics*, Wiley-Interscience, New York, 1990.

- **Casseau, V.** — *An Open-Source CFD Solver for Planetary Entry*, Ph.D. Thesis, Department of Mechanical and Aerospace Engineering, University of Strathclyde, Glasgow, UK, 2017.

- **Williams, C., Di Renzo, M., and Urzay, J.** — *Two-temperature extension of the HTR solver for hypersonic turbulent flows in thermochemical nonequilibrium*, Annual Research Briefs 2021, Center for Turbulence Research, Stanford University, pp. 95–104, 2021.

- **Scalabrin, L.C.** — *Numerical Simulation of Weakly Ionized Hypersonic Flow over Reentry Capsules*, Ph.D. Thesis, University of Michigan, Ann Arbor, MI, 2007.
