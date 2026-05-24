from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


K_VALUES = tuple(range(1, 16, 2))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate MAP/PH/1 LST moment-budget SLURM chunks and plot mean L2."
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path("/scratch200/davidfine/map_ph1_lst_moment_budget_1000"),
        help="Root folder containing chunk_* output directories.",
    )
    return parser.parse_args()


def plot_mean_l2(summary, root_dir):
    out_file = root_dir / "map_ph1_lst_mean_l2_vs_odd_moments_up_to_15.png"

    fig, axis = plt.subplots(figsize=(9, 6))
    axis.plot(
        summary["K"],
        summary["mean_l2"],
        marker="o",
        linewidth=2.5,
        color="#2f76b7",
        label="Mean L2",
    )
    axis.set_title("Mean MAP/PH/1 LST reconstruction error vs number of moments", fontweight="bold")
    axis.set_xlabel("Number of moments K", fontweight="bold")
    axis.set_ylabel("Mean L2 distance", fontweight="bold")
    axis.set_xticks(list(K_VALUES))
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(out_file, dpi=250)
    plt.close(fig)
    return out_file


def main():
    args = parse_args()
    root_dir = args.root_dir
    fit_files = sorted(root_dir.glob("chunk_*/map_ph1_lst_fit_results.csv"))
    example_files = sorted(root_dir.glob("chunk_*/map_ph1_examples_summary.csv"))

    if not fit_files:
        raise FileNotFoundError(f"No chunk fit files found under {root_dir}")

    fits = pd.concat((pd.read_csv(path) for path in fit_files), ignore_index=True)
    fits["K"] = pd.to_numeric(fits["K"], errors="coerce")
    fits["l2"] = pd.to_numeric(fits["l2"], errors="coerce")
    ok_fits = fits[np.isfinite(fits["l2"])].copy()

    summary = (
        ok_fits.groupby("K")
        .agg(
            rows=("l2", "count"),
            mean_l2=("l2", "mean"),
            median_l2=("l2", "median"),
            max_l2=("l2", "max"),
        )
        .reset_index()
        .sort_values("K")
    )

    root_dir.mkdir(parents=True, exist_ok=True)
    fits_path = root_dir / "map_ph1_lst_fit_results_all_chunks.csv"
    summary_path = root_dir / "map_ph1_lst_l2_summary_by_K.csv"
    fits.to_csv(fits_path, index=False)
    summary.to_csv(summary_path, index=False)

    if example_files:
        examples = pd.concat((pd.read_csv(path) for path in example_files), ignore_index=True)
        examples.to_csv(root_dir / "map_ph1_examples_summary_all_chunks.csv", index=False)

    plot_path = plot_mean_l2(summary, root_dir)

    print("Read chunk fit files:", len(fit_files))
    print("Saved combined fits:", fits_path)
    print("Saved summary:", summary_path)
    print("Saved mean-only plot:", plot_path)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
