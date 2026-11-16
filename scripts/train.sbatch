#!/bin/bash
#SBATCH -A prj0000000262
#SBATCH --job-name ipc-ppo
#SBATCH --partition cpup
#SBATCH --nodes 1
#SBATCH --cpus-per-task 16
#SBATCH --time 40:00:00
#SBATCH --output=logs/%x-%j.out
#
# Train one policy. Submit from the repository root:
#     sbatch --export=ALL,POLICY=kan,RUN=kan_v1 scripts/train.sbatch
#     sbatch --export=ALL,POLICY=mlp,RUN=mlp_v1 scripts/train.sbatch
#
# An update is three 300-s OpenFAST episodes run side by side (~17 min of wall time), so the
# default 100 updates take about 29 hours. More cores do not make it faster; 16 CPUs leave room
# for the six calibration episodes of the first run.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source /apps/miniforge/24.11.3/etc/profile.d/conda.sh
conda activate windctrl
export OMP_NUM_THREADS=1

echo "node=$(hostname) policy=${POLICY:-kan} run=${RUN:-kan_ipc} extra=${EXTRA:-}"
python -m windctrl.train --policy "${POLICY:-kan}" --run "${RUN:-kan_ipc}" ${EXTRA:-}
