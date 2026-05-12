"""
Analyze the interdeparture distribution area for queue 0 and queue 1.

Input:
    interdeparture_balanced_dataset.pkl

Outputs:
    interdeparture_area_summary_by_queue.csv
    interdeparture_area_correlation_matrix.csv
    autocorrelation_by_queue.png
    log_moment_histograms_by_queue.png
    moment_autocorrelation_area.png
    moment_area_m2_m3.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATASET_PATH = Path("interdeparture_balanced_dataset.pkl")
OUTPUT_DIR = Path("interdeparture_area_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MOMENT_COLUMNS = [f"log_interdeparture_moment_{i}" for i in range(1, 11)]
FEATURE_COLUMNS = ["first_autocorrelation"] + MOMENT_COLUMNS


def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}\n"
            "First run build_interdeparture_dataset.py."
        )

    df = pd.read_pickle(DATASET_PATH)
    required = ["queue_id"] + FEATURE_COLUMNS
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    return df


def save_summary_tables(df):
    summary = df.groupby("queue_id")[FEATURE_COLUMNS].describe().T
    summary_path = OUTPUT_DIR / "interdeparture_area_summary_by_queue.csv"
    summary.to_csv(summary_path)

    corr = df[FEATURE_COLUMNS].corr()
    corr_path = OUTPUT_DIR / "interdeparture_area_correlation_matrix.csv"
    corr.to_csv(corr_path)

    return summary_path, corr_path


def plot_autocorrelation_histogram(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    colors = {0: "#28666e", 1: "#b3532f"}

    for queue_id, axis in zip([0, 1], axes):
        part = df[df["queue_id"] == queue_id]
        axis.hist(
            part["first_autocorrelation"],
            bins=60,
            color=colors[queue_id],
            edgecolor="white",
            alpha=0.9,
        )
        axis.set_title(f"Queue {queue_id}: first autocorrelation")
        axis.set_xlabel("first autocorrelation")
        axis.set_ylabel("count")

    fig.tight_layout()
    path = OUTPUT_DIR / "autocorrelation_by_queue.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_log_moment_histograms(df):
    columns = MOMENT_COLUMNS[:6]
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    axes = axes.ravel()

    for axis, column in zip(axes, columns):
        for queue_id, color in [(0, "#28666e"), (1, "#b3532f")]:
            part = df[df["queue_id"] == queue_id]
            axis.hist(
                part[column],
                bins=60,
                alpha=0.55,
                color=color,
                label=f"queue {queue_id}",
                density=True,
            )
        axis.set_title(column)
        axis.set_xlabel("log moment value")
        axis.set_ylabel("density")
        axis.legend()

    fig.tight_layout()
    path = OUTPUT_DIR / "log_moment_histograms_by_queue.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_moment_autocorrelation_area(df):
    fig, axis = plt.subplots(figsize=(9, 6))
    colors = {0: "#28666e", 1: "#b3532f"}

    for queue_id in [0, 1]:
        part = df[df["queue_id"] == queue_id]
        axis.scatter(
            part["log_interdeparture_moment_2"],
            part["first_autocorrelation"],
            s=5,
            alpha=0.22,
            color=colors[queue_id],
            label=f"queue {queue_id}",
        )

    axis.set_xlabel("log second interdeparture moment")
    axis.set_ylabel("first autocorrelation")
    axis.set_title("Distribution area: moment vs autocorrelation")
    axis.legend(markerscale=3)
    fig.tight_layout()

    path = OUTPUT_DIR / "moment_autocorrelation_area.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_moment_area_m2_m3(df):
    fig, axis = plt.subplots(figsize=(9, 6))
    colors = {0: "#28666e", 1: "#b3532f"}

    for queue_id in [0, 1]:
        part = df[df["queue_id"] == queue_id]
        axis.scatter(
            part["log_interdeparture_moment_2"],
            part["log_interdeparture_moment_3"],
            s=5,
            alpha=0.22,
            color=colors[queue_id],
            label=f"queue {queue_id}",
        )

    axis.set_xlabel("log second interdeparture moment")
    axis.set_ylabel("log third interdeparture moment")
    axis.set_title("Distribution area: second vs third log moment")
    axis.legend(markerscale=3)
    fig.tight_layout()

    path = OUTPUT_DIR / "moment_area_m2_m3.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def print_research_description(df):
    counts = df["queue_id"].value_counts().sort_index()
    autocorr_range = df.groupby("queue_id")["first_autocorrelation"].agg(
        ["min", "mean", "max"]
    )
    moment_ranges = df.groupby("queue_id")[MOMENT_COLUMNS[:3]].agg(["min", "mean", "max"])

    print("Rows by queue:")
    print(counts)
    print()
    print("First autocorrelation range by queue:")
    print(autocorr_range)
    print()
    print("First three log moment ranges by queue:")
    print(moment_ranges)
    print()
    print("Research wording:")
    print(
        "The empirical distribution area is represented by the first 10 log raw "
        "moments of the interdeparture times together with the first lag "
        "autocorrelation. The dataset is balanced across the two queues, so the "
        "region of queue 0 and queue 1 can be compared directly in the same "
        "feature space."
    )


def main():
    df = load_dataset()
    summary_path, corr_path = save_summary_tables(df)
    plot_paths = [
        plot_autocorrelation_histogram(df),
        plot_log_moment_histograms(df),
        plot_moment_autocorrelation_area(df),
        plot_moment_area_m2_m3(df),
    ]

    print_research_description(df)
    print()
    print(f"Saved summary: {summary_path}")
    print(f"Saved correlation matrix: {corr_path}")
    for path in plot_paths:
        print(f"Saved plot: {path}")


if __name__ == "__main__":
    main()
