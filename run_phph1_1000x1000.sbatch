#!/bin/bash
#SBATCH --job-name=phph1_1000x1000
#SBATCH --partition=power-general-shared-pool
#SBATCH --array=0-999%100
#SBATCH --output=/scratch200/davidfine/phph1/logs/phph1_%A_%a.out
#SBATCH --error=/scratch200/davidfine/phph1/logs/phph1_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

set -euo pipefail

PROJECT_DIR=/scratch200/davidfine/phph1
OUTPUT_DIR=/scratch200/davidfine/phph1_data
BUTOOLS_DIR=/scratch200/davidfine/butools2/Python

mkdir -p "$OUTPUT_DIR"
mkdir -p "$PROJECT_DIR/logs"

cd "$PROJECT_DIR"
export BUTOOLS_PATH="$BUTOOLS_DIR"

echo "SLURM job id: ${SLURM_JOB_ID}"
echo "SLURM array task id: ${SLURM_ARRAY_TASK_ID}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Each task generates 1000 examples."
echo "Array concurrency limit: up to 100 tasks can run in parallel."
echo "Global example IDs: $((SLURM_ARRAY_TASK_ID * 1000))..$((SLURM_ARRAY_TASK_ID * 1000 + 999))"
echo "Sampling utilization rho/service mean uniformly in [0.3, 0.99]."

python -u generate_phph1_sojourn_moments.py \
  --output-dir "$OUTPUT_DIR" \
  --num-examples 1000 \
  --resume 1 \
  --random-service-mean 1 \
  --service-mean-min 0.3 \
  --service-mean-max 0.99 \
  --payload-format dict
