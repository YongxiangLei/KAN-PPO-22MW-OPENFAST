#!/bin/bash
#SBATCH -A prj0000000262
#SBATCH --job-name ipc-filter
#SBATCH --partition cpup
#SBATCH --nodes 1
#SBATCH --cpus-per-task 32
#SBATCH --time 1:30:00
#SBATCH --output=logs/%x-%j.out
#
# Four-gain law with slower filters on its pitch offsets, on the evaluation winds (see
# scripts/filter_study.py). Five controllers x six winds = 30 OpenFAST runs in parallel; a 600-s
# run takes about 35 minutes. Submit from the repository root:
#     sbatch scripts/filter_study.sbatch

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source /apps/miniforge/24.11.3/etc/profile.d/conda.sh
conda activate windctrl
export OMP_NUM_THREADS=1

ARGS=${ARGS:---name eval_lawfilter_v2 --controllers pi law:kan_v2:tau=0.1 law:kan_v2:tau=0.3 law:kan_v2:tau=0.6 law:kan_v2:tau=1.0}
echo "node=$(hostname) args=$ARGS"
python scripts/filter_study.py $ARGS
