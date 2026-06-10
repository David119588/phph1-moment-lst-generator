#!/bin/bash
#SBATCH --job-name=tandem_ph
#SBATCH --partition=power-general-shared-pool
#SBATCH --array=0-999
#SBATCH --output=/scratch200/davidfine/tandem_ph/tandem_%A_%a.out
#SBATCH --error=/scratch200/davidfine/tandem_ph/tandem_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

cd /scratch200/davidfine/phph1

python -u simulate_tandem_ph_departures.py \
  --output-dir /scratch200/davidfine/tandem_departure_pkls \
  --manifest-file "manifest_${SLURM_ARRAY_TASK_ID}.csv" \
  --num-systems 1 \
  --example-start "${SLURM_ARRAY_TASK_ID}" \
  --num-departures 100000 \
  --warmup-departures 5000 \
  --seed $((20260610 + SLURM_ARRAY_TASK_ID)) \
  --resume 1
