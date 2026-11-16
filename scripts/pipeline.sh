#!/bin/bash
# Run the rest of Project 1 unattended: wait for training, evaluate, interpret, publish.
#
#     nohup setsid scripts/pipeline.sh > logs/pipeline.out 2>&1 < /dev/null &
#     tail -f logs/pipeline.out
#
# Stages (each is skipped when its output already exists, so the script can simply be restarted):
#   1. wait for the ROSCO IPC gain sweep on the training winds; rerun it as a job if it is missing
#   2. wait for the KAN and MLP training jobs, pushing the training curves every few hours
#   3. evaluate PI, ROSCO IPC (best gain), KAN-PPO and MLP-PPO on the evaluation winds
#   4. interpret the KAN policy and extract the four-gain law
#   5. evaluate the four-gain law in closed loop
#   6. write results/, the paper figures and tables, paper/overleaf.zip; commit and push
#
# TAG names the set of runs (default v1); KAN_JOB / MLP_JOB are the Slurm job ids of the training
# runs to wait for. A second set runs alongside with, e.g.,
#     TAG=v2 KAN_JOB=<id> MLP_JOB=<id> nohup setsid scripts/pipeline.sh > logs/pipeline_v2.out 2>&1 < /dev/null &

set -uo pipefail
cd "$(dirname "$0")/.."
source /apps/miniforge/24.11.3/etc/profile.d/conda.sh
conda activate windctrl
export OMP_NUM_THREADS=1

TAG=${TAG:-v1}                   # names the set of runs: kan_$TAG, mlp_$TAG, eval_$TAG, eval_law_$TAG
KAN_JOB=${KAN_JOB:-206385}
MLP_JOB=${MLP_JOB:-206386}
KAN_RUN=kan_$TAG
MLP_RUN=mlp_$TAG
PUSH_EVERY=$((4 * 3600))          # seconds between progress pushes while training runs

log() { echo "[$(date '+%F %T')] [$TAG] $*"; }

running() { squeue -h -j "$1" 2>/dev/null | grep -q .; }

publish() {
    # Refresh results/, this tag's paper files and zip, then commit and push whatever changed.
    # The lock keeps two pipelines (different tags) from using the git index at the same time.
    exec 9> logs/publish.lock
    flock 9
    python -m windctrl.report --tag "$TAG" > "logs/report_$TAG.out" 2>&1 || { log "report failed, see logs/report_$TAG.out"; flock -u 9; return 1; }
    git add -A results paper config docs
    if git diff --cached --quiet; then
        log "publish: nothing new"
    else
        git commit -q -m "$1 ($TAG)

Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>" && log "committed: $1"
    fi
    git push -q origin main && log "pushed" || log "push failed (will retry at the next publish)"
    flock -u 9
}

evaluate_job() {
    # $1 = number of CPUs, $2 = arguments of windctrl.evaluate. Blocks until the job has finished.
    log "submitting evaluation: $2"
    sbatch --wait -c "$1" --export=ALL,ARGS="$2" scripts/evaluate.sbatch
    log "evaluation job ended with status $?"
}

best_ipc_gain() {
    # Gain with the lowest mean blade DEL among the sweep entries that had no aborted run.
    python - <<'EOF'
import csv, collections, math
rows = list(csv.DictReader(open("outputs/ipc_sweep/results.csv")))
acc, bad = collections.defaultdict(list), set()
for r in rows:
    if not r["spec"].startswith("rosco_ipc"):
        continue
    if r["failed"] == "True" or not r.get("blade_DEL"):
        bad.add(r["spec"])
    else:
        acc[r["spec"]].append(float(r["blade_DEL"]))
ok = {s: sum(v) / len(v) for s, v in acc.items() if s not in bad}
print(min(ok, key=ok.get) if ok else "")
EOF
}

# ---- 1. ROSCO IPC gain sweep ---------------------------------------------------------------------
log "stage 1: ROSCO IPC gain sweep"
while pgrep -u "$USER" -f launch_ipc_sweep > /dev/null; do sleep 60; done
if [ ! -f outputs/ipc_sweep/results.csv ]; then
    log "sweep results missing; running the sweep as a job"
    evaluate_job 40 "--name ipc_sweep --winds train --seeds 101 --episode-seconds 300 --controllers pi rosco_ipc:3e-10 rosco_ipc:1e-9 rosco_ipc:3e-9 rosco_ipc:1e-8"
fi
if [ -f outputs/ipc_sweep/results.csv ]; then
    BEST=$(best_ipc_gain)
    log "best ROSCO IPC gain: ${BEST:-none (every gain had an aborted run)}"
else
    BEST=""
    log "sweep did not produce results; continuing without a ROSCO IPC baseline"
fi
publish "Add ROSCO IPC gain sweep and current training curves"

# ---- 2. training ---------------------------------------------------------------------------------
log "stage 2: waiting for training jobs $KAN_JOB (KAN) and $MLP_JOB (MLP)"
last=$(date +%s)
while running "$KAN_JOB" || running "$MLP_JOB"; do
    sleep 300
    if [ $(( $(date +%s) - last )) -ge "$PUSH_EVERY" ]; then
        publish "Update training curves"
        last=$(date +%s)
    fi
done
log "training jobs finished: $(sacct -n -X -j "$KAN_JOB,$MLP_JOB" -o JobID,State,Elapsed | tr -s ' ' | tr '\n' ';')"
if [ ! -f outputs/$KAN_RUN/checkpoint.pt ]; then
    log "no KAN checkpoint; stopping"
    exit 1
fi
publish "Add final training curves and checkpoints"

# ---- 3. evaluation -------------------------------------------------------------------------------
log "stage 3: evaluation on unseen winds"
CONTROLLERS="pi"
KAN_INDEX=1
if [ -n "$BEST" ]; then CONTROLLERS="$CONTROLLERS $BEST"; KAN_INDEX=2; fi
CONTROLLERS="$CONTROLLERS kan:$KAN_RUN"
[ -f outputs/$MLP_RUN/checkpoint.pt ] && CONTROLLERS="$CONTROLLERS mlp:$MLP_RUN"
if [ ! -f outputs/eval_$TAG/results.csv ]; then
    evaluate_job 56 "--name eval_$TAG --controllers $CONTROLLERS"
fi
[ -f outputs/eval_$TAG/results.csv ] || { log "evaluation produced no results; stopping"; publish "Update results"; exit 1; }
publish "Add closed-loop evaluation on unseen winds"

# ---- 4. interpretation ---------------------------------------------------------------------------
log "stage 4: interpretation of the KAN policy"
if [ ! -f outputs/$KAN_RUN/interpret/law.json ]; then
    python -m windctrl.interpret --run "$KAN_RUN" --states "outputs/eval_$TAG/states_c${KAN_INDEX}.npz" > "logs/interpret_$TAG.out" 2>&1 \
        || log "interpretation failed, see logs/interpret_$TAG.out"
fi

# ---- 5. four-gain law in closed loop -------------------------------------------------------------
if [ -f outputs/$KAN_RUN/interpret/law.json ] && [ ! -f outputs/eval_law_$TAG/results.csv ]; then
    log "stage 5: four-gain law in closed loop"
    evaluate_job 16 "--name eval_law_$TAG --controllers law:$KAN_RUN"
fi

# ---- 6. publish ----------------------------------------------------------------------------------
publish "Add interpretation, four-gain law evaluation and updated paper project"
log "PIPELINE DONE"
