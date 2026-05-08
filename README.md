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
arrival/service PH sizes that add to `100`, and covers SCV bands up to `20`.

```bash
python generate_phph1_sojourn_moments.py --clean-output 1
```

On a Linux server, use a Linux output path:

```bash
python generate_phph1_sojourn_moments.py --output-dir /scratch200/davidfine/phph1_data --clean-output 1
```

Each PKL stores the arrival PH, service PH, sojourn ME representation, SCV
metadata, and log moment arrays.

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
python generate_phph1_sojourn_moments.py --output-dir /scratch200/davidfine/phph1_data --clean-output 1
python plot_phph1_scv_histograms.py --input-dir /scratch200/davidfine/phph1_data --output-dir /scratch200/davidfine/phph1
python plot_ph_size_histograms.py --input-dir /scratch200/davidfine/phph1_data --output-dir /scratch200/davidfine/phph1
```
