"""
Sample PH distributions from three families and plot shape coverage.

The output figure matches the first-station distributional coverage diagnostic:
SCV vs skewness and SCV vs kurtosis, colored by PH family.
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from sample_ph_family_histograms import scale_ph_to_mean


FAMILY_LABELS = {
    "hyper_erlang": "Hyper Erlang",
    "coxian": "Coxian",
    "general": "General",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot SCV/skewness/kurtosis coverage for sampled PH families."
    )
    parser.add_argument("--ph-size", type=int, default=100)
    parser.add_argument("--per-family", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument(
        "--families",
        type=str,
        default="hyper_erlang,coxian,general",
        help="Comma-separated families from: hyper_erlang, coxian, general.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ph_family_shape_coverage"),
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--skewness-y-max",
        type=float,
        default=200.0,
        help="Y-axis cap for the skewness panel. Use <=0 for automatic.",
    )
    parser.add_argument(
        "--kurtosis-y-max",
        type=float,
        default=500.0,
        help="Y-axis cap for the kurtosis panel. Use <=0 for automatic.",
    )
    parser.add_argument(
        "--scv-max",
        type=float,
        default=150.0,
        help="X-axis cap for both panels. Use <=0 for automatic.",
    )
    return parser.parse_args()


def ph_shape_metrics(alpha, t_matrix):
    alpha = np.asarray(alpha, dtype=float)
    if alpha.ndim == 1:
        alpha = alpha.reshape(1, -1)

    inv = np.linalg.inv(-t_matrix)
    ones = np.ones((t_matrix.shape[0], 1), dtype=float)
    moments = []
    power = np.eye(t_matrix.shape[0], dtype=float)

    for order in range(1, 5):
        power = power @ inv
        moment = math.factorial(order) * (alpha @ power @ ones)[0, 0]
        moments.append(float(moment))

    mean, second, third, fourth = moments
    variance = second - mean**2
    if variance <= 0.0:
        return None

    scv = variance / mean**2
    mu3 = third - 3.0 * mean * second + 2.0 * mean**3
    mu4 = fourth - 4.0 * mean * third + 6.0 * mean**2 * second - 3.0 * mean**4
    skewness = mu3 / variance**1.5
    kurtosis = mu4 / variance**2

    values = [mean, scv, skewness, kurtosis]
    if not all(np.isfinite(value) for value in values):
        return None

    return {
        "mean": mean,
        "scv": scv,
        "skewness": float(skewness),
        "kurtosis": float(kurtosis),
    }


def random_composition(total, parts, rng):
    cuts = sorted(rng.choice(np.arange(1, total), size=parts - 1, replace=False))
    sizes = [cuts[0]]
    sizes += [cuts[i] - cuts[i - 1] for i in range(1, parts - 1)]
    sizes += [total - cuts[-1]]
    return sizes


def random_hyper_erlang_shape_ph(n, rng):
    branch_count = int(rng.integers(2, min(25, n) + 1))
    sizes = random_composition(n, branch_count, rng)
    weights = rng.dirichlet(np.full(branch_count, 0.25))
    rates = np.exp(rng.uniform(np.log(0.005), np.log(200.0), size=branch_count))

    alpha = np.zeros((1, n))
    t_matrix = np.zeros((n, n))
    offset = 0

    for branch_idx, branch_size in enumerate(sizes):
        alpha[0, offset] = weights[branch_idx]
        rate = rates[branch_idx]
        for local_idx in range(branch_size):
            pos = offset + local_idx
            t_matrix[pos, pos] = -rate
            if local_idx < branch_size - 1:
                t_matrix[pos, pos + 1] = rate
        offset += branch_size

    return scale_ph_to_mean(alpha, t_matrix, target_mean=1.0)


def random_coxian_shape_ph(n, rng):
    alpha = np.zeros((1, n))
    alpha[0, 0] = 1.0
    rates = np.exp(rng.uniform(np.log(0.005), np.log(200.0), size=n))
    continue_probs = rng.beta(0.45, 0.45, size=n - 1)
    t_matrix = np.zeros((n, n))

    for i in range(n):
        t_matrix[i, i] = -rates[i]
        if i < n - 1:
            t_matrix[i, i + 1] = continue_probs[i] * rates[i]

    return scale_ph_to_mean(alpha, t_matrix, target_mean=1.0)


def random_general_shape_ph(n, rng):
    alpha = rng.dirichlet(np.full(n, 0.08)).reshape(1, n)
    rates = np.exp(rng.uniform(np.log(0.0005), np.log(500.0), size=n))
    t_matrix = np.zeros((n, n))

    for i in range(n):
        absorb_prob = rng.beta(2.0, 0.35)
        transition_probs = rng.dirichlet(np.full(n - 1, 0.35)) * (1.0 - absorb_prob)
        t_matrix[i, i] = -rates[i]

        off_diag = [j for j in range(n) if j != i]
        for prob, j in zip(transition_probs, off_diag):
            t_matrix[i, j] = rates[i] * prob

    return scale_ph_to_mean(alpha, t_matrix, target_mean=1.0)


def random_ph_by_family(family, n, rng):
    if family == "hyper_erlang":
        return random_hyper_erlang_shape_ph(n, rng)
    if family == "coxian":
        return random_coxian_shape_ph(n, rng)
    if family == "general":
        return random_general_shape_ph(n, rng)
    raise ValueError(f"Unknown family: {family}")


def sample_rows(families, ph_size, per_family, seed):
    rng = np.random.default_rng(seed)
    rows = []

    for family in families:
        accepted = 0
        attempts = 0
        max_attempts = per_family * 20

        while accepted < per_family and attempts < max_attempts:
            attempts += 1
            alpha, t_matrix = random_ph_by_family(family, ph_size, rng)
            metrics = ph_shape_metrics(alpha, t_matrix)
            if metrics is None:
                continue

            rows.append(
                {
                    "family": family,
                    "family_label": FAMILY_LABELS.get(family, family),
                    "sample_index": accepted,
                    "attempt": attempts,
                    "ph_size": ph_size,
                    **metrics,
                }
            )
            accepted += 1

        if accepted < per_family:
            raise RuntimeError(
                f"Only accepted {accepted} finite {family} samples "
                f"after {attempts} attempts."
            )

    return rows


def write_rows(rows, output_dir):
    csv_path = output_dir / "ph_family_shape_coverage.csv"
    fieldnames = [
        "family",
        "family_label",
        "sample_index",
        "attempt",
        "ph_size",
        "mean",
        "scv",
        "skewness",
        "kurtosis",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


def set_axis_limits(axis, x_max, y_max):
    axis.set_xlim(left=0.0)
    axis.set_ylim(bottom=0.0)
    if x_max > 0.0:
        axis.set_xlim(right=x_max)
    if y_max > 0.0:
        axis.set_ylim(top=y_max)


def plot_coverage(rows, families, output_dir, dpi, scv_max, skewness_y_max, kurtosis_y_max):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    colors = {
        "hyper_erlang": "#2f74bc",
        "coxian": "#f28e2b",
        "general": "#59a14f",
    }

    for family in families:
        family_rows = [row for row in rows if row["family"] == family]
        scv = [row["scv"] for row in family_rows]
        skewness = [row["skewness"] for row in family_rows]
        kurtosis = [row["kurtosis"] for row in family_rows]
        label = FAMILY_LABELS.get(family, family)

        axes[0].scatter(
            scv,
            skewness,
            s=9,
            alpha=0.82,
            linewidths=0.0,
            color=colors.get(family),
            label=label,
            rasterized=True,
        )
        axes[1].scatter(
            scv,
            kurtosis,
            s=9,
            alpha=0.82,
            linewidths=0.0,
            color=colors.get(family),
            label=label,
            rasterized=True,
        )

    axes[0].set_xlabel("SCV", fontsize=12)
    axes[0].set_ylabel("Skewness", fontsize=12)
    axes[1].set_xlabel("SCV", fontsize=12)
    axes[1].set_ylabel("Kurtosis", fontsize=12)

    set_axis_limits(axes[0], scv_max, skewness_y_max)
    set_axis_limits(axes[1], scv_max, kurtosis_y_max)

    for axis in axes:
        axis.legend(fontsize=8, frameon=True)
        axis.grid(False)

    fig.tight_layout(w_pad=4.5)
    png_path = output_dir / "ph_family_scv_skewness_kurtosis.png"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return png_path


def family_summary(rows, families):
    lines = []
    for family in families:
        family_rows = [row for row in rows if row["family"] == family]
        scv = np.asarray([row["scv"] for row in family_rows], dtype=float)
        skewness = np.asarray([row["skewness"] for row in family_rows], dtype=float)
        kurtosis = np.asarray([row["kurtosis"] for row in family_rows], dtype=float)
        lines.append(
            (
                FAMILY_LABELS.get(family, family),
                len(family_rows),
                float(np.min(scv)),
                float(np.max(scv)),
                float(np.min(skewness)),
                float(np.max(skewness)),
                float(np.min(kurtosis)),
                float(np.max(kurtosis)),
            )
        )
    return lines


def main():
    args = parse_args()
    families = [item.strip() for item in args.families.split(",") if item.strip()]
    unknown = sorted(set(families) - set(FAMILY_LABELS))
    if unknown:
        raise ValueError(f"Unknown families: {unknown}. Allowed: {sorted(FAMILY_LABELS)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = sample_rows(families, args.ph_size, args.per_family, args.seed)
    csv_path = write_rows(rows, args.output_dir)
    png_path = plot_coverage(
        rows,
        families,
        args.output_dir,
        args.dpi,
        args.scv_max,
        args.skewness_y_max,
        args.kurtosis_y_max,
    )

    print(f"Saved CSV: {csv_path}")
    print(f"Saved plot: {png_path}")
    print("Family,count,SCV min,SCV max,skewness min,skewness max,kurtosis min,kurtosis max")
    for line in family_summary(rows, families):
        print(
            f"{line[0]},{line[1]},{line[2]:.6g},{line[3]:.6g},"
            f"{line[4]:.6g},{line[5]:.6g},{line[6]:.6g},{line[7]:.6g}"
        )


if __name__ == "__main__":
    main()
