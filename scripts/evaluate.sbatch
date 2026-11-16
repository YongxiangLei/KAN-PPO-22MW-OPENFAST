#!/bin/bash
#SBATCH -A prj0000000262
#SBATCH --job-name ipc-eval
#SBATCH --partition cpup
#SBATCH --nodes 1
#SBATCH --cpus-per-task 64
#SBATCH --time 2:00:00
#SBATCH --output=logs/%x-%j.out
#
# Evaluate controllers on the same winds. Submit from the repository root, e.g.
#     sbatch --export=ALL,ARGS="--name eval_v1 --controllers pi rosco_ipc:1e-9 kan:kan_v1 mlp:mlp_v1" scripts/evaluate.sbatch
#
# One OpenFAST run per (controller, wind) pair, all in parallel: 4 controllers x 6 winds = 24 runs.
# Request about two CPUs per run (`sbatch -c N`). A 600-s run takes about 35 minutes.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source /apps/miniforge/24.11.3/etc/profile.d/conda.sh
conda activate windctrl
export OMP_NUM_THREADS=1

echo "node=$(hostname) args=${ARGS:-}"
python -m windctrl.evaluate ${ARGS:-}
