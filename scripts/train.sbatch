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
# With the defaults an update is three 300-s OpenFAST episodes side by side (about 23 min on the
# compute nodes), so 100 updates take about 38 hours, and more cores do not make that faster.
# To use more cores, pass more and shorter episodes through EXTRA and request two CPUs per episode:
#     sbatch -c 48 --export=ALL,POLICY=kan,RUN=kan_v2,EXTRA="--episodes-per-speed 4 --episode-seconds 75 --burn-in-seconds 20 --seeds 101 102 103 104 105 106 107 108" scripts/train.sbatch

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source /apps/miniforge/24.11.3/etc/profile.d/conda.sh
conda activate windctrl
export OMP_NUM_THREADS=1

echo "node=$(hostname) policy=${POLICY:-kan} run=${RUN:-kan_ipc} extra=${EXTRA:-}"
python -m windctrl.train --policy "${POLICY:-kan}" --run "${RUN:-kan_ipc}" ${EXTRA:-}
