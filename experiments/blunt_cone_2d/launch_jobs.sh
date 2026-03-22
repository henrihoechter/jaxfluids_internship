#!/usr/bin/env bash
# launch_jobs.sh — write per-job run scripts and launch them in tmux
set -euo pipefail

REPO=/home/hhoechter/jaxfluids_internship
SCRIPT=$REPO/experiments/blunt_cone_2d/run_blunt_cone.py
MESH=$REPO/data/bluntedCone.msh
DATA=$REPO/data
PYTHON=$REPO/.venv/bin/python
OUTDIR=$REPO/experiments/blunt_cone_2d

# ── shared numerics ────────────────────────────────────────────────────────────
T_FINAL=1e-5
DT=1e-11
DT_MODE=fixed
SAVE_INTERVAL=200
SESSION=bluntcone

# ── write per-job run scripts ─────────────────────────────────────────────────
cat > "$OUTDIR/run_frozen_N2.sh" << EOF
#!/usr/bin/env bash
set -euo pipefail
cd $REPO
source .venv/bin/activate
PYTHONPATH=$REPO/src $PYTHON $SCRIPT \\
    --mesh=$MESH \\
    --t-final=$T_FINAL \\
    --dt=$DT \\
    --dt-mode=$DT_MODE \\
    --save-interval=$SAVE_INTERVAL \\
    --tag-inflow=1 --tag-outflow=2 --tag-wall=3 --tag-axis=7 \\
    --transport=casseau \\
    --species=N2 \\
    --output=solution_frozen_N2.npz
EOF

cat > "$OUTDIR/run_reacting_N2_N.sh" << EOF
#!/usr/bin/env bash
set -euo pipefail
cd $REPO
source .venv/bin/activate
PYTHONPATH=$REPO/src $PYTHON $SCRIPT \\
    --mesh=$MESH \\
    --t-final=$T_FINAL \\
    --dt=$DT \\
    --dt-mode=$DT_MODE \\
    --save-interval=$SAVE_INTERVAL \\
    --tag-inflow=1 --tag-outflow=2 --tag-wall=3 --tag-axis=7 \\
    --transport=casseau \\
    --species=N2,N \\
    --reactions=$DATA/casseau_qk_reactions.json \\
    --collision-integrals=$DATA/collision_integrals_tp2867.json \\
    --output=solution_reacting_N2_N.npz
EOF

chmod +x "$OUTDIR/run_frozen_N2.sh" "$OUTDIR/run_reacting_N2_N.sh"

# ── launch in tmux ────────────────────────────────────────────────────────────
tmux new-session -d -s $SESSION 2>/dev/null || true

tmux new-window -t $SESSION -n frozen_N2
tmux send-keys -t $SESSION:frozen_N2 \
    "bash $OUTDIR/run_frozen_N2.sh 2>&1 | tee $OUTDIR/frozen_N2.log" Enter

tmux new-window -t $SESSION -n reacting_N2_N
tmux send-keys -t $SESSION:reacting_N2_N \
    "bash $OUTDIR/run_reacting_N2_N.sh 2>&1 | tee $OUTDIR/reacting_N2_N.log" Enter

echo "Jobs launched in tmux session '$SESSION'."
echo ""
echo "  tmux attach -t $SESSION        # attach"
echo "  Ctrl-B, n / p                  # next / prev window"
echo "  Ctrl-B, d                      # detach"
echo ""
echo "  tail -f $OUTDIR/frozen_N2.log"
echo "  tail -f $OUTDIR/reacting_N2_N.log"
