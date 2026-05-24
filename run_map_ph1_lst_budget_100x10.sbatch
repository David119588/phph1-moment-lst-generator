#!/bin/bash
#SBATCH --job-name=map_ph1_l2
#SBATCH --partition=power-general-shared-pool
#SBATCH --array=0-99%35
#SBATCH --output=/scratch200/davidfine/phph1/logs/map_ph1_l2_%A_%a.out
#SBATCH --error=/scratch200/davidfine/phph1/logs/map_ph1_l2_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

set -euo pipefail

PROJECT_DIR=/scratch200/davidfine/phph1
ROOT_OUTPUT_DIR=/scratch200/davidfine/map_ph1_lst_moment_budget_1000
BUTOOLS_DIR=/scratch200/davidfine/butools2/Python

EXAMPLES_PER_TASK=10
TASK_ID=${SLURM_ARRAY_TASK_ID}
START_EXAMPLE=$((TASK_ID * EXAMPLES_PER_TASK))
CHUNK_DIR=${ROOT_OUTPUT_DIR}/chunk_${TASK_ID}

mkdir -p "$CHUNK_DIR"
mkdir -p "$PROJECT_DIR/logs"

cd "$PROJECT_DIR"
export BUTOOLS_PATH="$BUTOOLS_DIR"

echo "MAP/PH/1 LST moment-budget chunk"
echo "SLURM job id: ${SLURM_JOB_ID}"
echo "SLURM array task id: ${TASK_ID}"
echo "Chunk output dir: ${CHUNK_DIR}"
echo "Global example IDs: $((START_EXAMPLE + 1))..$((START_EXAMPLE + EXAMPLES_PER_TASK))"
echo "MAP size up to 100, PH service size up to 100, odd moments K=1,3,...,15"

python -u map_ph1_lst_moment_budget_1000.py \
  --num-examples "$EXAMPLES_PER_TASK" \
  --example-id-offset "$START_EXAMPLE" \
  --queue-id 1 \
  --seed "$((20260520 + TASK_ID))" \
  --output-dir "$CHUNK_DIR" \
  --max-map-size 100 \
  --min-map-size 2 \
  --service-size-min 2 \
  --service-size-max 100 \
  --rho-min 0.3 \
  --rho-max 0.99 \
  --service-scv-min 0.15 \
  --service-scv-max 20 \
  --candidate-count 80 \
  --n-s 180 \
  --s-max-factor 20 \
  --progress-every 1
