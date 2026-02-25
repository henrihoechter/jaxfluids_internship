#!/usr/bin/env bash
# launch_jobs.sh — launch blunt-cone run jobs via tmux
# Usage: bash launch_jobs.sh
# Each job runs in its own tmux window so you can detach and reattach freely.
set -euo pipefail

REPO=/home/hhoechter/jaxfluids_internship
SCRIPT=$REPO/experiments/blunt_cone_2d/run_blunt_cone.py
MESH=$REPO/data/bluntedCone.msh
DATA=$REPO/data
VENV=$REPO/.venv/bin/python

# ── shared numerics ────────────────────────────────────────────────────────────
T_FINAL=1e-5
DT=1e-11
DT_MODE=fixed
SAVE_INTERVAL=200

# ── mesh tags ─────────────────────────────────────────────────────────────────
TAG_INFLOW=1
TAG_OUTFLOW=2
TAG_WALL=3
TAG_AXIS=7          # remapped from gmsh tag 7 -> tag 4 internally

# ── common args ───────────────────────────────────────────────────────────────
COMMON="
  --mesh=$MESH
  --t-final=$T_FINAL
  --dt=$DT
  --dt-mode=$DT_MODE
  --save-interval=$SAVE_INTERVAL
  --tag-inflow=$TAG_INFLOW
  --tag-outflow=$TAG_OUTFLOW
  --tag-wall=$TAG_WALL
  --tag-axis=$TAG_AXIS
  --transport=casseau
"

SESSION=bluntcone

# create a new tmux session (detached) or reuse existing
tmux new-session -d -s $SESSION 2>/dev/null || true

# ── job 1: frozen N2, no reactions ────────────────────────────────────────────
tmux new-window -t $SESSION -n "frozen_N2"
tmux send-keys -t $SESSION:frozen_N2 "
cd $REPO && source .venv/bin/activate && \\
PYTHONPATH=$REPO/src $VENV $SCRIPT \\
  $COMMON \\
  --species=N2 \\
  --output=solution_frozen_N2.npz \\
  2>&1 | tee experiments/blunt_cone_2d/frozen_N2.log
" Enter

# ── job 2: N2 + N with dissociation reactions ─────────────────────────────────
tmux new-window -t $SESSION -n "reacting_N2_N"
tmux send-keys -t $SESSION:reacting_N2_N "
cd $REPO && source .venv/bin/activate && \\
PYTHONPATH=$REPO/src $VENV $SCRIPT \\
  $COMMON \\
  --species=N2,N \\
  --reactions=$DATA/casseau_qk_reactions.json \\
  --collision-integrals=$DATA/collision_integrals_tp2867.json \\
  --output=solution_reacting_N2_N.npz \\
  2>&1 | tee experiments/blunt_cone_2d/reacting_N2_N.log
" Enter

echo "Jobs launched in tmux session '$SESSION'."
echo ""
echo "  tmux attach -t $SESSION          # attach"
echo "  Ctrl-B, n / p                    # next / prev window"
echo "  Ctrl-B, d                        # detach"
echo ""
echo "  tail -f $REPO/experiments/blunt_cone_2d/frozen_N2.log"
echo "  tail -f $REPO/experiments/blunt_cone_2d/reacting_N2_N.log"
