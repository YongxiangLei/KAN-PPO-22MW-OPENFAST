#!/bin/bash
#SBATCH -A prj0000000262
#SBATCH --job-name openfast
#SBATCH --partition cpup
#SBATCH --nodes 1
#SBATCH --cpus-per-task 240
#SBATCH --time 2:00:00
#SBATCH --output=logs/%x-%j.out
#
# Run every case in runs/$SWEEP/cases.txt, one OpenFAST process per physical core.
# Submit from ~/projects/WindCtrl:
#     sbatch --export=ALL,SWEEP=steady scripts/run_cases.sbatch
# 240 CPUs = 120 physical cores, the most one cpup node offers (256 threads, 16 reserved).
# For small sweeps request fewer, e.g. `sbatch -c 48 ...`, to queue faster and cost less.
#
# OpenFAST is essentially single-threaded, so the speed-up comes from running many
# cases at once rather than from giving one case more cores.

set -euo pipefail
SWEEP=${SWEEP:-steady}
cd "$SLURM_SUBMIT_DIR"
CASES=runs/$SWEEP/cases.txt

source /apps/miniforge/24.11.3/etc/profile.d/conda.sh
conda activate windctrl

# cpup nodes expose 2 hardware threads per core; use one process per physical core.
NPROC=$(( SLURM_CPUS_PER_TASK / 2 ))
export OMP_NUM_THREADS=1

echo "node=$(hostname) cpus=$SLURM_CPUS_PER_TASK parallel=$NPROC cases=$(wc -l < "$CASES")"
echo "openfast: $(which openfast)"
START=$SECONDS

# Each case runs in its own directory and logs to <case>.log; a failure does not stop the rest.
xargs -a "$CASES" -P "$NPROC" -I{} bash -c '
    fst={}; dir=$(dirname "$fst"); name=$(basename "$fst" .fst)
    cd "$dir"
    # Turbulent cases: generate the wind field first (skipped if it already exists).
    if [ -f "${name}_wind.inp" ] && [ ! -f "${name}_wind.bts" ]; then
        turbsim "${name}_wind.inp" > "${name}_wind.log" 2>&1 || { echo "FAIL $name (turbsim)"; exit 0; }
    fi
    if openfast "$name.fst" > "$name.log" 2>&1; then echo "OK   $name"; else echo "FAIL $name"; fi
' || true

echo "wall time: $(( SECONDS - START )) s"
echo "finished: $(grep -l "OpenFAST terminated normally" runs/$SWEEP/*/*.log 2>/dev/null | wc -l) / $(wc -l < "$CASES")"
