#!/bin/bash
#SBATCH --job-name=mapfit_s1
#SBATCH --partition=power-general-shared-pool
#SBATCH --output=/scratch200/davidfine/map_from_tandem_fit/station1_%j.out
#SBATCH --error=/scratch200/davidfine/map_from_tandem_fit/station1_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

cd /scratch200/davidfine/phph1
mkdir -p /scratch200/davidfine/map_from_tandem_fit

export BUTOOLS_PATH=/scratch200/davidfine/butools2/Python

python -u fit_map_from_departure_trace_sample.py \
  --input /scratch200/davidfine/tandem_departure_pkls \
  --output-dir /scratch200/davidfine/map_from_tandem_fit/station1_order5_sample \
  --station 1 \
  --num-files 20 \
  --map-order 5 \
  --max-trace-length 95000 \
  --min-trace-length 5000 \
  --normalize-mean 1 \
  --max-iter 200 \
  --stop-cond 1e-7
