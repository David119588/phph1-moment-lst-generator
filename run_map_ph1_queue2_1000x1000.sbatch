#!/bin/bash
#SBATCH --job-name=map_ph1_q2
#SBATCH --partition=power-general-shared-pool
#SBATCH --array=0-999%100
#SBATCH --output=/scratch200/davidfine/phph1/logs/map_ph1_q2_%A_%a.out
#SBATCH --error=/scratch200/davidfine/phph1/logs/map_ph1_q2_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

set -euo pipefail

PROJECT_DIR=/scratch200/davidfine/phph1
OUTPUT_DIR=/scratch200/davidfine/map_ph1_queue2_data
BUTOOLS_DIR=/scratch200/davidfine/butools2/Python

mkdir -p "$OUTPUT_DIR"
mkdir -p "$PROJECT_DIR/logs"

cd "$PROJECT_DIR"
export BUTOOLS_PATH="$BUTOOLS_DIR"

echo "MAP/PH/1 second queue experiment"
echo "SLURM job id: ${SLURM_JOB_ID}"
echo "SLURM array task id: ${SLURM_ARRAY_TASK_ID}"
echo "Each task generates 1000 examples."
echo "Up to 100 tasks run in parallel."
echo "Global example IDs: $((SLURM_ARRAY_TASK_ID * 1000))..$((SLURM_ARRAY_TASK_ID * 1000 + 999))"
echo "Sampling service mean/utilization rho uniformly in [0.3, 0.99]."

python -u sample_map_ph1_sojourn.py \
  --output-dir "$OUTPUT_DIR" \
  --num-examples 1000 \
  --resume 1 \
  --service-mean-min 0.3 \
  --service-mean-max 0.99 \
  --service-size 20 \
  --plot 1
