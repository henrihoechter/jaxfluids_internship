#!/usr/bin/env bash
set -euo pipefail
cd /home/hhoechter/jaxfluids_internship
source .venv/bin/activate
PYTHONPATH=/home/hhoechter/jaxfluids_internship/src /home/hhoechter/jaxfluids_internship/.venv/bin/python /home/hhoechter/jaxfluids_internship/experiments/blunt_cone_2d/run_blunt_cone.py \
    --mesh=/home/hhoechter/jaxfluids_internship/data/bluntedCone.msh \
    --t-final=1e-5 \
    --dt=1e-11 \
    --dt-mode=fixed \
    --save-interval=200 \
    --tag-inflow=1 --tag-outflow=2 --tag-wall=3 --tag-axis=7 \
    --transport=casseau \
    --species=N2 \
    --output=solution_frozen_N2.npz
