"""
Analyze the interdeparture distribution area for queue 0 and queue 1.

Input:
    interdeparture_balanced_dataset.pkl

Outputs:
    interdeparture_area_summary_by_queue.csv
    interdeparture_area_correlation_matrix.csv
    histogram_probability_checks.csv
    autocorrelation_by_queue.png
    probability_histograms_by_queue.png
    density_histograms_by_queue.png
    log_moment_histograms_by_queue.png  # legacy density filename
    moment_autocorrelation_area.png
    moment_area_m2_m3.png
    moment_autocorrelation_probability_heatmap.png
    moment_m2_m3_probability_heatmap.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASET_PATH = Path("interdeparture_balanced_dataset.pkl")
OUTPUT_DIR = Path("interdeparture_area_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MOMENT_COLUMNS = [f"log_interdeparture_moment_{i}" for i in range(1, 11)]
FEATURE_COLUMNS = ["first_autocorrelation"] + MOMENT_COLUMNS
PLOTTED_MOMENT_COLUMNS = MOMENT_COLUMNS[1:7]
COLORS = {0: "#28666e", 1: "#b3532f"}


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


def histogram_checks_for_feature(values, bins):
    values = np.asarray(values, dtype=float)
    counts, edges = np.histogram(values, bins=bins)
    total = counts.sum()
    if total == 0:
        return np.nan, np.nan

    probabilities = counts / total
    probability_sum = probabilities.sum()

    densities, density_edges = np.histogram(values, bins=bins, density=True)
    widths = np.diff(density_edges)
    density_area = float(np.sum(densities * widths))
    return float(probability_sum), density_area


def save_histogram_checks(df, bins=60):
    rows = []
    for queue_id in sorted(df["queue_id"].unique()):
        part = df[df["queue_id"] == queue_id]
        for column in FEATURE_COLUMNS:
            probability_sum, density_area = histogram_checks_for_feature(
                part[column], bins=bins
            )
            rows.append(
                {
                    "queue_id": int(queue_id),
                    "feature": column,
                    "n": int(len(part)),
                    "probability_sum": probability_sum,
                    "density_area_estimate": density_area,
                }
            )

    path = OUTPUT_DIR / "histogram_probability_checks.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def plot_autocorrelation_histogram(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    for queue_id, axis in zip([0, 1], axes):
        part = df[df["queue_id"] == queue_id]
        axis.hist(
            part["first_autocorrelation"],
            bins=60,
            color=COLORS[queue_id],
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


def plot_probability_histograms(df):
    columns = PLOTTED_MOMENT_COLUMNS
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    axes = axes.ravel()

    for axis, column in zip(axes, columns):
        for queue_id, color in COLORS.items():
            part = df[df["queue_id"] == queue_id]
            weights = np.ones(len(part), dtype=float) / len(part)
            axis.hist(
                part[column],
                bins=60,
                weights=weights,
                alpha=0.55,
                color=color,
                label=f"queue {queue_id}",
            )
        axis.set_title(column)
        axis.set_xlabel("log moment value")
        axis.set_ylabel("probability per bin")
        axis.legend()

    fig.tight_layout()
    path = OUTPUT_DIR / "probability_histograms_by_queue.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_density_histograms(df):
    columns = PLOTTED_MOMENT_COLUMNS
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    axes = axes.ravel()

    for axis, column in zip(axes, columns):
        for queue_id, color in COLORS.items():
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
        axis.set_ylabel("PDF density (area = 1)")
        axis.legend()

    fig.tight_layout()
    path = OUTPUT_DIR / "density_histograms_by_queue.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)

    legacy_path = OUTPUT_DIR / "log_moment_histograms_by_queue.png"
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    axes = axes.ravel()
    for axis, column in zip(axes, columns):
        for queue_id, color in COLORS.items():
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
        axis.set_ylabel("PDF density (area = 1)")
        axis.legend()
    fig.tight_layout()
    fig.savefig(legacy_path, dpi=180)
    plt.close(fig)
    return path


def plot_moment_autocorrelation_area(df):
    fig, axis = plt.subplots(figsize=(9, 6))

    for queue_id in [0, 1]:
        part = df[df["queue_id"] == queue_id]
        axis.scatter(
            part["log_interdeparture_moment_2"],
            part["first_autocorrelation"],
            s=5,
            alpha=0.22,
            color=COLORS[queue_id],
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

    for queue_id in [0, 1]:
        part = df[df["queue_id"] == queue_id]
        axis.scatter(
            part["log_interdeparture_moment_2"],
            part["log_interdeparture_moment_3"],
            s=5,
            alpha=0.22,
            color=COLORS[queue_id],
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


def plot_probability_heatmaps(df, x_col, y_col, output_name, title):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)

    x_min, x_max = df[x_col].min(), df[x_col].max()
    y_min, y_max = df[y_col].min(), df[y_col].max()
    x_bins = np.linspace(x_min, x_max, 70)
    y_bins = np.linspace(y_min, y_max, 70)

    for queue_id, axis in zip([0, 1], axes):
        part = df[df["queue_id"] == queue_id]
        weights = np.ones(len(part), dtype=float) / len(part)
        mesh = axis.hist2d(
            part[x_col],
            part[y_col],
            bins=[x_bins, y_bins],
            weights=weights,
            cmap="viridis",
            cmin=1.0 / len(part),
        )
        axis.set_title(f"{title}: queue {queue_id}")
        axis.set_xlabel(x_col)
        axis.set_ylabel(y_col)
        colorbar = fig.colorbar(mesh[3], ax=axis)
        colorbar.set_label("probability per 2D bin")

    fig.tight_layout()
    path = OUTPUT_DIR / output_name
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
        "The empirical distribution area is represented by log raw moments 2-10 "
        "of the interdeparture times together with the first lag autocorrelation. "
        "The first log moment is summarized numerically but is not used as a main "
        "area plot because it is normalized near log(1)=0. Probability histograms "
        "show mass per bin; density histograms show PDF height whose area is 1."
    )


def main():
    df = load_dataset()
    summary_path, corr_path = save_summary_tables(df)
    checks_path = save_histogram_checks(df)
    plot_paths = [
        plot_autocorrelation_histogram(df),
        plot_probability_histograms(df),
        plot_density_histograms(df),
        plot_moment_autocorrelation_area(df),
        plot_moment_area_m2_m3(df),
        plot_probability_heatmaps(
            df,
            "log_interdeparture_moment_2",
            "first_autocorrelation",
            "moment_autocorrelation_probability_heatmap.png",
            "Moment vs autocorrelation probability",
        ),
        plot_probability_heatmaps(
            df,
            "log_interdeparture_moment_2",
            "log_interdeparture_moment_3",
            "moment_m2_m3_probability_heatmap.png",
            "Second vs third moment probability",
        ),
    ]

    print_research_description(df)
    print()
    print(f"Saved summary: {summary_path}")
    print(f"Saved correlation matrix: {corr_path}")
    print(f"Saved histogram checks: {checks_path}")
    for path in plot_paths:
        print(f"Saved plot: {path}")


if __name__ == "__main__":
    main()
