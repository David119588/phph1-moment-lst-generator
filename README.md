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

If BuTools is not installed globally, set the `BUTOOLS_PATH` variable inside `generate_phph1_dataset.py`.