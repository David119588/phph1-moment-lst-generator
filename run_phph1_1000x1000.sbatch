#!/bin/bash
#SBATCH --job-name=phph1_1000x1000
#SBATCH --partition=power-general-shared-pool
#SBATCH --output=/scratch200/davidfine/phph1/logs/phph1_%j.out
#SBATCH --error=/scratch200/davidfine/phph1/logs/phph1_%j.err
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

if [[ -z "${MANUAL_JOB_INDEX:-}" ]]; then
  echo "ERROR: MANUAL_JOB_INDEX is required for this non-array script."
  echo "Submit with: sbatch --export=ALL,MANUAL_JOB_INDEX=0 run_phph1_1000x1000.sbatch"
  exit 1
fi

echo "SLURM job id: ${SLURM_JOB_ID}"
echo "Manual job index: ${MANUAL_JOB_INDEX}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Each task generates 1000 examples."
echo "Global example IDs: $((MANUAL_JOB_INDEX * 1000))..$((MANUAL_JOB_INDEX * 1000 + 999))"
echo "Sampling utilization rho/service mean uniformly in [0.3, 0.99]."

python -u generate_phph1_sojourn_moments.py \
  --output-dir "$OUTPUT_DIR" \
  --num-examples 1000 \
  --job-index "$MANUAL_JOB_INDEX" \
  --resume 1 \
  --random-service-mean 1 \
  --service-mean-min 0.3 \
  --service-mean-max 0.99 \
  --payload-format dict
