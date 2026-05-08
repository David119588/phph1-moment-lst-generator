"""
Plot SCV histograms plus skewness/kurtosis-vs-SCV graphs from PH/PH/1 PKLs.

For each saved example, the script reads:
    - inter-arrival SCV
    - service-time SCV
    - first 10 inter-arrival raw moments
    - first 10 service raw moments

The PKLs store log moments, so this script exponentiates them before computing
skewness and kurtosis.
"""

import argparse
import csv
import pickle
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


FILENAME_RE = re.compile(
    r"_ia_scv_(?P<arrival>.*?)_svc_mean_.*?_svc_scv_(?P<service>.*?)(?:_ex_\d+)?\.pkl$"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot SCV, skewness, and kurtosis graphs for PH/PH/1 examples."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=r"C:\phph1",
        help="Folder containing the PH/PH/1 PKL files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Folder where PNG/CSV outputs are saved.",
    )
    parser.add_argument("--bins", type=int, default=40)
    return parser.parse_args()


def decode_float_from_name(text):
    return float(text.replace("m", "-").replace("p", "."))


def filename_scvs(path):
    match = FILENAME_RE.search(path.name)
    if match is None:
        return None, None
    return (
        decode_float_from_name(match.group("arrival")),
        decode_float_from_name(match.group("service")),
    )


def standardized_shape(raw_moments):
    m = np.asarray(raw_moments, dtype=float)
    mean = m[0]
    second = m[1]
    third = m[2]
    fourth = m[3]

    variance = second - mean**2
    if variance <= 0.0 or not np.isfinite(variance):
        return np.nan, np.nan

    mu3 = third - 3.0 * mean * second + 2.0 * mean**3
    mu4 = fourth - 4.0 * mean * third + 6.0 * mean**2 * second - 3.0 * mean**4

    skewness = mu3 / variance**1.5
    kurtosis = mu4 / variance**2
    return float(skewness), float(kurtosis)


def row_from_pickle(path):
    with open(path, "rb") as f:
        payload = pickle.load(f)

    file_arrival_scv, file_service_scv = filename_scvs(path)

    if isinstance(payload, dict):
        log_input = np.asarray(payload["log_input_moments"], dtype=float)
        arrival_scv = float(payload.get("arrival_scv", file_arrival_scv))
        service_scv = float(payload.get("service_scv", file_service_scv))
        arrival_family = payload.get("arrival_family", "unknown")
        service_family = payload.get("service_family", "unknown")
        arrival_size = payload.get("arrival_size", "")
        service_size = payload.get("service_size", "")
        example_id = payload.get("example_id", "")
    else:
        log_input = np.asarray(payload[0], dtype=float)
        arrival_scv = file_arrival_scv
        service_scv = file_service_scv
        arrival_family = "unknown"
        service_family = "unknown"
        arrival_size = ""
        service_size = ""
        example_id = ""

    if arrival_scv is None or service_scv is None:
        raise ValueError(f"Could not read SCV values for {path}")

    input_moments = np.exp(log_input)
    arrival_moments = input_moments[:10]
    service_moments = input_moments[10:20]

    arrival_skewness, arrival_kurtosis = standardized_shape(arrival_moments)
    service_skewness, service_kurtosis = standardized_shape(service_moments)

    return {
        "example_id": example_id,
        "path": str(path),
        "arrival_family": arrival_family,
        "service_family": service_family,
        "arrival_size": arrival_size,
        "service_size": service_size,
        "arrival_scv": arrival_scv,
        "service_scv": service_scv,
        "arrival_skewness": arrival_skewness,
        "service_skewness": service_skewness,
        "arrival_kurtosis": arrival_kurtosis,
        "service_kurtosis": service_kurtosis,
    }


def load_rows(input_dir):
    rows = []
    for path in sorted(input_dir.glob("*.pkl")):
        if FILENAME_RE.search(path.name) is None:
            continue
        rows.append(row_from_pickle(path))
    return rows


def save_rows(rows, output_dir):
    csv_path = output_dir / "scv_shape_values.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def plot_histograms(rows, bins, output_dir):
    arrival = [row["arrival_scv"] for row in rows]
    service = [row["service_scv"] for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    axes[0].hist(arrival, bins=bins, color="#28666e", edgecolor="white", alpha=0.9)
    axes[0].set_title(f"Inter-arrival SCV ({len(arrival)} examples)")
    axes[0].set_xlabel("SCV")
    axes[0].set_ylabel("count")

    axes[1].hist(service, bins=bins, color="#b3532f", edgecolor="white", alpha=0.9)
    axes[1].set_title(f"Service-time SCV ({len(service)} examples)")
    axes[1].set_xlabel("SCV")

    fig.tight_layout()
    path = output_dir / "scv_histograms.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_shape_vs_scv(rows, shape_key, label, output_dir):
    arrival_scv = np.asarray([row["arrival_scv"] for row in rows], dtype=float)
    service_scv = np.asarray([row["service_scv"] for row in rows], dtype=float)
    arrival_shape = np.asarray([row[f"arrival_{shape_key}"] for row in rows], dtype=float)
    service_shape = np.asarray([row[f"service_{shape_key}"] for row in rows], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(arrival_scv, arrival_shape, s=14, alpha=0.55, color="#28666e")
    axes[0].set_title(f"Inter-arrival {label} vs SCV")
    axes[0].set_xlabel("SCV")
    axes[0].set_ylabel(label)
    axes[0].grid(True, alpha=0.25)

    axes[1].scatter(service_scv, service_shape, s=14, alpha=0.55, color="#b3532f")
    axes[1].set_title(f"Service-time {label} vs SCV")
    axes[1].set_xlabel("SCV")
    axes[1].set_ylabel(label)
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    path = output_dir / f"{shape_key}_vs_scv.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(input_dir)
    if not rows:
        raise RuntimeError(f"No matching PH/PH/1 PKLs found in {input_dir}")

    csv_path = save_rows(rows, output_dir)
    hist_path = plot_histograms(rows, args.bins, output_dir)
    skew_path = plot_shape_vs_scv(rows, "skewness", "skewness", output_dir)
    kurt_path = plot_shape_vs_scv(rows, "kurtosis", "kurtosis", output_dir)

    print("Number of examples:", len(rows))
    print("Saved values:", csv_path)
    print("Saved SCV histograms:", hist_path)
    print("Saved skewness vs SCV:", skew_path)
    print("Saved kurtosis vs SCV:", kurt_path)


if __name__ == "__main__":
    main()
