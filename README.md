# PH/PH/1 Moment and LST Dataset Generator

This project generates datasets for PH/PH/1 queues.

For each sampled PH/PH/1 queue, the code saves:

- arrival moments
- service moments
- sojourn-time moments
- utilization rho
- true sojourn-time LST on a fixed grid
- fitted ME LSTs using MEFromMoments
- L2 reconstruction errors
- validity checks for fitted LSTs

## Example run

```bash
python generate_phph1_dataset.py --num-examples 10 --arrival-size 20 --service-size 10 --k-list 1,3,5,7 --output-dir C:\phph1_dataset_medium --resume 0
```

## Generate PH/PH/1 log-moment PKLs

The single-example generator can also create a clean folder of random PH/PH/1
examples. By default it writes to `C:\phph1`, creates `1000` PKLs, samples
arrival/service PH sizes that add to `100`, samples service mean uniformly from
`0.3` to `0.99`, and covers SCV bands up to `20`.

```bash
python generate_phph1_sojourn_moments.py --clean-output 1
```

On a Linux server, use a Linux output path:

```bash
python generate_phph1_sojourn_moments.py --output-dir /scratch200/davidfine/phph1_data --clean-output 1
```

For SLURM, use an array job when you want many independent jobs. This example
submits `1000` SLURM tasks, and each task creates `1000` PKLs. The task ID is
used as an offset, so task `0` writes examples `0..999`, task `1` writes
examples `1000..1999`, and so on.

```bash
cat > run_phph1_array.sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=phph1
#SBATCH --partition=power-general-shared-pool
#SBATCH --array=0-999
#SBATCH --output=/scratch200/davidfine/phph1/phph1_%A_%a.out
#SBATCH --error=/scratch200/davidfine/phph1/phph1_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

cd /scratch200/davidfine/phph1
export BUTOOLS_PATH=/scratch200/davidfine/butools2/Python

python -u generate_phph1_sojourn_moments.py \
  --output-dir /scratch200/davidfine/phph1_data \
  --num-examples 1000 \
  --resume 1 \
  --random-service-mean 1 \
  --service-mean-min 0.3 \
  --service-mean-max 0.99
EOF

sbatch run_phph1_array.sbatch
```

Do not use `--clean-output 1` inside an array job, because every task shares the
same output folder.

Each PKL stores the arrival PH, service PH, sojourn ME representation, SCV
metadata, and log moment arrays.

## Sample MAP/PH/1 second-queue sojourn data

Use `sample_map_ph1_sojourn.py` when the second queue arrival process is a MAP.
The default sampler uses weakly correlated 2-state MAP arrivals with small
positive and small negative autocorrelation around zero, matching the measured
interdeparture autocorrelation range. It also samples diverse PH service times
with PH size up to `100` and SCV up to `20`, then computes the MAP/PH/1
sojourn-time ME representation with BuTools `MAPMAP1`.

Local/single-job example:

```bash
python sample_map_ph1_sojourn.py --output-dir C:\map_ph1_queue2 --num-examples 1000 --service-mean-min 0.3 --service-mean-max 0.99 --random-service-size 1 --service-size-min 2 --service-size-max 100 --service-scv-max 20 --map-corr-mode near_zero_mixed
```

SLURM array example:

```bash
sbatch run_map_ph1_queue2_1000x1000.sbatch
```

This submits `1000` array tasks with `1000` examples per task. The array limit
`%100` means up to `100` tasks run in parallel. Each example samples service PH
size uniformly from `2` to `100`, service SCV up to `20`, and utilization in
`[0.3, 0.99]`. The MAP sampler defaults to near-zero mixed autocorrelation.
Outputs are written to `/scratch200/davidfine/map_ph1_queue2_data`.

The script saves PKLs, per-job manifest CSV files, and these graphs:

- `map_ph1_histograms.png`
- `map_ph1_sojourn_area.png`
- `map_ph1_service_scv_vs_sojourn.png`

## MAP/PH/1 LST moment-budget experiment

This experiment answers how many moments are needed to reconstruct the
MAP/PH/1 sojourn-time LST in the second queue. It samples `1000` MAP/PH/1
queues, with MAP size up to `100` and PH service size up to `100`, then fits
from odd moments `K = 1, 3, ..., 15`.

On SLURM, run the parallel chunks:

```bash
cd /scratch200/davidfine/phph1
sbatch run_map_ph1_lst_budget_100x10.sbatch
```

After all array tasks finish, combine the chunks and draw the final mean-only
L2 graph:

```bash
python aggregate_map_ph1_lst_slurm.py --root-dir /scratch200/davidfine/map_ph1_lst_moment_budget_1000
```

Final outputs:

- `/scratch200/davidfine/map_ph1_lst_moment_budget_1000/map_ph1_lst_l2_summary_by_K.csv`
- `/scratch200/davidfine/map_ph1_lst_moment_budget_1000/map_ph1_lst_mean_l2_vs_odd_moments_up_to_15.png`

## Plot SCV and PH-size graphs

```bash
python plot_phph1_scv_histograms.py --output-dir C:\Users\osamb\PycharmProjects\PythonProject5
python plot_ph_size_histograms.py --output-dir C:\Users\osamb\PycharmProjects\PythonProject5
```

The plot scripts create:

- `scv_histograms.png`
- `skewness_vs_scv.png`
- `kurtosis_vs_scv.png`
- `ph_size_histograms.png`

## Build interdeparture moment/autocorrelation dataset

Use `build_interdeparture_dataset.py` to create a balanced pandas table from
the departure PKLs. The table contains half queue-0 examples and half queue-1
examples, with the first 10 log interdeparture moments and the first
autocorrelation.

First run, when `C:\DEPART_1` has not been extracted yet:

```bash
python build_interdeparture_dataset.py --extract --delete-empty --examples-per-queue 25000 --output-dir C:\Users\osamb\PycharmProjects\PythonProject5
```

Later runs, after both `C:\DEPART_0` and `C:\DEPART_1` already exist:

```bash
python build_interdeparture_dataset.py --delete-empty --examples-per-queue 25000 --output-dir C:\Users\osamb\PycharmProjects\PythonProject5
```

Outputs:

- `interdeparture_balanced_dataset.csv`
- `interdeparture_balanced_dataset.pkl`

To describe the empirical distribution area after the dataset is created, run:

```bash
python analyze_interdeparture_area.py
```

This creates the folder `interdeparture_area_analysis` with summary CSV files
and plots comparing the two queues:

- `interdeparture_area_summary_by_queue.csv`
- `interdeparture_area_correlation_matrix.csv`
- `autocorrelation_by_queue.png`
- `log_moment_histograms_by_queue.png`
- `moment_autocorrelation_area.png`
- `moment_area_m2_m3.png`

## Check outputs

```bash
python check_outputs.py --output-dir C:\phph1_dataset_medium
```

## Output files

The generator creates the following files inside the output directory:

- `examples_summary.csv`
- `fits_summary.csv`
- `failures.csv`
- `experiment_plan.csv`
- `ph_pools.pkl`
- `examples_pkl/`

## Main CSV files

### `examples_summary.csv`

One row per PH/PH/1 example.

It contains:

- arrival PH information
- service PH information
- utilization `rho`
- arrival moments
- service moments
- sojourn-time moments
- path to the full PKL file

### `fits_summary.csv`

One row per pair `(example_id, K)`.

It contains:

- number of fitted moments `K`
- fitted LST error
- L2 distance
- validity checks
- monotonicity checks

## Notes

The code requires BuTools.

If BuTools is not installed globally, set the `BUTOOLS_PATH` environment variable.
For example, if the BuTools `Python` folder is at `/scratch200/davidfine/butools2/Python`:

```bash
export BUTOOLS_PATH=/scratch200/davidfine/butools2/Python
python -c "import sys; sys.path.append('$BUTOOLS_PATH'); import butools; print('BuTools OK')"
```

Then run the generator from the cloned repo:

```bash
cd /scratch200/davidfine/phph1
python generate_phph1_sojourn_moments.py --output-dir /scratch200/davidfine/phph1_data --num-examples 1000 --resume 1
python plot_phph1_scv_histograms.py --input-dir /scratch200/davidfine/phph1_data --output-dir /scratch200/davidfine/phph1
python plot_ph_size_histograms.py --input-dir /scratch200/davidfine/phph1_data --output-dir /scratch200/davidfine/phph1
```
