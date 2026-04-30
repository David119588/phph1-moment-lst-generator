import argparse
import pandas as pd
from pathlib import Path
from pandas.errors import EmptyDataError


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check PH/PH/1 dataset output files."
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=r"C:\phph1_dataset_medium",
        help="Output directory to check."
    )

    return parser.parse_args()


args = parse_args()

OUTPUT_DIR = Path(args.output_dir)

examples_path = OUTPUT_DIR / "examples_summary.csv"
fits_path = OUTPUT_DIR / "fits_summary.csv"
failures_path = OUTPUT_DIR / "failures.csv"


def safe_read_csv(path):
    if not path.exists():
        print(f"File does not exist: {path}")
        return pd.DataFrame()

    if path.stat().st_size == 0:
        print(f"File is empty: {path}")
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except EmptyDataError:
        print(f"CSV has no columns: {path}")
        return pd.DataFrame()


examples_df = safe_read_csv(examples_path)
fits_df = safe_read_csv(fits_path)
failures_df = safe_read_csv(failures_path)

print("=" * 70)
print("CHECK OUTPUTS")
print("=" * 70)
print("OUTPUT_DIR:", OUTPUT_DIR)
print("examples_df shape:", examples_df.shape)
print("fits_df shape:", fits_df.shape)
print("failures_df shape:", failures_df.shape)

print("\nExamples:")
if len(examples_df) > 0:
    cols = [
        "example_id", "rho", "arrival_family", "service_family",
        "arrival_scv", "service_scv", "mean_sojourn", "status"
    ]
    cols = [c for c in cols if c in examples_df.columns]
    print(examples_df[cols].head())
else:
    print("examples_df is empty")

print("\nFits:")
if len(fits_df) > 0:
    cols = [
        "example_id", "K", "l2_norm", "fit_status", "fit_valid",
        "is_in_0_1", "is_monotone"
    ]
    cols = [c for c in cols if c in fits_df.columns]
    print(fits_df[cols].head(30))

    print("\nL2 summary by K:")
    fits_ok = fits_df[fits_df["fit_status"].astype(str).str.lower() == "ok"].copy()
    if len(fits_ok) > 0:
        print(
            fits_ok
            .groupby("K")
            .agg(
                rows=("l2_norm", "count"),
                mean_l2=("l2_norm", "mean"),
                median_l2=("l2_norm", "median"),
                max_l2=("l2_norm", "max"),
                valid_rows=("fit_valid", "sum"),
            )
            .reset_index()
        )
else:
    print("fits_df is empty")

print("\nFailures:")
if len(failures_df) > 0:
    print(failures_df.tail())
else:
    print("No failures were recorded, or failures.csv is empty.")