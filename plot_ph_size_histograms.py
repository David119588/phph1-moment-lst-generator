"""
Plot histograms of arrival/service PH sizes from saved PH/PH/1 examples.

This reads phph1_examples_manifest.csv, which is written by
generate_phph1_sojourn_moments.py.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Plot PH-size histograms.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=r"C:\phph1",
        help="Folder containing phph1_examples_manifest.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Folder where PNG/CSV outputs are saved.",
    )
    parser.add_argument("--bins", type=int, default=25)
    return parser.parse_args()


def load_rows(input_dir):
    manifest_path = Path(input_dir) / "phph1_examples_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    rows = []
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "example_id": row["example_id"],
                    "arrival_size": int(row["arrival_size"]),
                    "service_size": int(row["service_size"]),
                    "total_ph_size": int(row["total_ph_size"]),
                    "sojourn_order": int(row["sojourn_order"]),
                    "path": row["path"],
                }
            )
    return rows


def save_rows(rows, output_dir):
    path = Path(output_dir) / "ph_size_values.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot_histograms(rows, bins, output_dir):
    arrival = [row["arrival_size"] for row in rows]
    service = [row["service_size"] for row in rows]
    sojourn = [row["sojourn_order"] for row in rows]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].hist(arrival, bins=bins, color="#28666e", edgecolor="white", alpha=0.9)
    axes[0].set_title(f"Arrival PH size ({len(rows)} examples)")
    axes[0].set_xlabel("arrival PH size")
    axes[0].set_ylabel("count")

    axes[1].hist(service, bins=bins, color="#b3532f", edgecolor="white", alpha=0.9)
    axes[1].set_title(f"Service PH size ({len(rows)} examples)")
    axes[1].set_xlabel("service PH size")

    axes[2].hist(sojourn, bins=bins, color="#585858", edgecolor="white", alpha=0.9)
    axes[2].set_title(f"Sojourn ME order ({len(rows)} examples)")
    axes[2].set_xlabel("arrival_size * service_size")

    fig.tight_layout()
    path = Path(output_dir) / "ph_size_histograms.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.input_dir)
    csv_path = save_rows(rows, output_dir)
    png_path = plot_histograms(rows, args.bins, output_dir)

    print("Number of examples:", len(rows))
    print("Arrival PH size range:", min(r["arrival_size"] for r in rows), "to", max(r["arrival_size"] for r in rows))
    print("Service PH size range:", min(r["service_size"] for r in rows), "to", max(r["service_size"] for r in rows))
    print("Saved values:", csv_path)
    print("Saved PH size histograms:", png_path)


if __name__ == "__main__":
    main()
