"""
Sample PH/PH/1 first-station input distributions and plot shape coverage.

Each example contains one arrival PH with mean 1 and one service PH with mean
sampled uniformly from the requested service-mean interval. Both distributions
are accepted only when their SCV is in the requested interval, unless the SCV
upper bound is disabled.
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
    "hyper_general": "Hyper General",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot SCV/skewness/kurtosis coverage for sampled PH families."
    )
    parser.add_argument("--num-examples", type=int, default=1000)
    parser.add_argument("--ph-size-min", type=int, default=2)
    parser.add_argument("--ph-size-max", type=int, default=100)
    parser.add_argument("--arrival-mean", type=float, default=1.0)
    parser.add_argument("--service-mean-min", type=float, default=0.5)
    parser.add_argument("--service-mean-max", type=float, default=0.95)
    parser.add_argument("--scv-min", type=float, default=0.0)
    parser.add_argument(
        "--scv-max-accepted",
        type=float,
        default=0.0,
        help="Maximum accepted SCV. Use <=0 for no upper bound.",
    )
    parser.add_argument("--max-attempts", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument(
        "--families",
        type=str,
        default="hyper_erlang,coxian,hyper_general",
        help="Comma-separated families from: hyper_erlang, coxian, hyper_general.",
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
        default=20.0,
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


def random_general_block_ph(n, rng):
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


def random_hyper_general_shape_ph(n, rng):
    branch_count = int(rng.integers(2, min(12, n) + 1))
    sizes = random_composition(n, branch_count, rng)
    weights = rng.dirichlet(np.full(branch_count, 0.2))

    alpha = np.zeros((1, n))
    t_matrix = np.zeros((n, n))
    offset = 0

    for branch_idx, branch_size in enumerate(sizes):
        branch_alpha, branch_t = random_general_block_ph(branch_size, rng)
        alpha[0, offset : offset + branch_size] = weights[branch_idx] * branch_alpha
        t_matrix[offset : offset + branch_size, offset : offset + branch_size] = branch_t
        offset += branch_size

    return scale_ph_to_mean(alpha, t_matrix, target_mean=1.0)


def random_ph_by_family(family, n, rng):
    if family == "hyper_erlang":
        return random_hyper_erlang_shape_ph(n, rng)
    if family == "coxian":
        return random_coxian_shape_ph(n, rng)
    if family == "hyper_general":
        return random_hyper_general_shape_ph(n, rng)
    raise ValueError(f"Unknown family: {family}")


def sample_one_ph(family, ph_size, target_mean, scv_min, scv_max, max_attempts, rng):
    for attempt in range(1, max_attempts + 1):
        alpha, t_matrix = random_ph_by_family(family, ph_size, rng)
        alpha, t_matrix = scale_ph_to_mean(alpha, t_matrix, target_mean=target_mean)
        metrics = ph_shape_metrics(alpha, t_matrix)
        if metrics is None:
            continue
        if metrics["scv"] >= scv_min and (scv_max <= 0.0 or metrics["scv"] <= scv_max):
            return attempt, metrics

    upper_text = "infinity" if scv_max <= 0.0 else f"{scv_max:.6g}"
    raise RuntimeError(
        f"Could not sample {family} PH with size={ph_size}, mean={target_mean:.6g}, "
        f"and SCV in [{scv_min:.6g}, {upper_text}] after {max_attempts} attempts."
    )


def sample_rows(args, families):
    rng = np.random.default_rng(args.seed)
    rows = []

    for example_id in range(args.num_examples):
        arrival_family = families[example_id % len(families)]
        service_family = families[(example_id + 1) % len(families)]
        arrival_size = int(rng.integers(args.ph_size_min, args.ph_size_max + 1))
        service_size = int(rng.integers(args.ph_size_min, args.ph_size_max + 1))
        service_mean = float(rng.uniform(args.service_mean_min, args.service_mean_max))

        for role, family, ph_size, target_mean in [
            ("arrival", arrival_family, arrival_size, args.arrival_mean),
            ("service", service_family, service_size, service_mean),
        ]:
            attempts, metrics = sample_one_ph(
                family,
                ph_size,
                target_mean,
                args.scv_min,
                args.scv_max_accepted,
                args.max_attempts,
                rng,
            )
            rows.append(
                {
                    "example_id": example_id,
                    "role": role,
                    "family": family,
                    "family_label": FAMILY_LABELS.get(family, family),
                    "attempt": attempts,
                    "ph_size": ph_size,
                    "target_mean": target_mean,
                    **metrics,
                }
            )

    return rows


def write_rows(rows, output_dir):
    csv_path = output_dir / "ph_family_shape_coverage.csv"
    fieldnames = [
        "example_id",
        "role",
        "family",
        "family_label",
        "attempt",
        "ph_size",
        "target_mean",
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


def set_axis_limits(axis, x_max, y_max, x_min=0.0):
    axis.set_xlim(left=x_min)
    axis.set_ylim(bottom=0.0)
    if x_max > 0.0:
        axis.set_xlim(right=x_max)
    if y_max > 0.0:
        axis.set_ylim(top=y_max)


def plot_coverage(
    rows,
    families,
    output_dir,
    dpi,
    scv_max,
    skewness_y_max,
    kurtosis_y_max,
    *,
    xscale="linear",
    filename="ph_family_scv_skewness_kurtosis.png",
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    colors = {
        "hyper_erlang": "#2f74bc",
        "coxian": "#f28e2b",
        "hyper_general": "#59a14f",
    }

    positive_scv = [row["scv"] for row in rows if row["scv"] > 0.0]
    x_min = 0.8 * min(positive_scv) if xscale == "log" and positive_scv else 0.0

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

    for axis in axes:
        axis.set_xscale(xscale)
        set_axis_limits(axis, scv_max, skewness_y_max if axis is axes[0] else kurtosis_y_max, x_min)
        axis.legend(fontsize=8, frameon=True)
        axis.grid(False)

    fig.tight_layout(w_pad=4.5)
    png_path = output_dir / filename
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
    if args.ph_size_min < 1 or args.ph_size_max < args.ph_size_min:
        raise ValueError("Require 1 <= --ph-size-min <= --ph-size-max.")
    if args.arrival_mean <= 0.0:
        raise ValueError("--arrival-mean must be positive.")
    if args.service_mean_min <= 0.0 or args.service_mean_max < args.service_mean_min:
        raise ValueError("Require 0 < --service-mean-min <= --service-mean-max.")
    if args.service_mean_max >= args.arrival_mean:
        raise ValueError("--service-mean-max must be smaller than --arrival-mean.")
    if args.scv_min < 0.0:
        raise ValueError("--scv-min must be nonnegative.")
    if 0.0 < args.scv_max_accepted < args.scv_min:
        raise ValueError("Require --scv-max-accepted <= 0 or --scv-max-accepted >= --scv-min.")

    families = [item.strip() for item in args.families.split(",") if item.strip()]
    unknown = sorted(set(families) - set(FAMILY_LABELS))
    if unknown:
        raise ValueError(f"Unknown families: {unknown}. Allowed: {sorted(FAMILY_LABELS)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = sample_rows(args, families)
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
    log_png_path = plot_coverage(
        rows,
        families,
        args.output_dir,
        args.dpi,
        args.scv_max,
        args.skewness_y_max,
        args.kurtosis_y_max,
        xscale="log",
        filename="ph_family_scv_skewness_kurtosis_log_scv.png",
    )

    print(f"Saved CSV: {csv_path}")
    print(f"Saved plot: {png_path}")
    print(f"Saved log-SCV plot: {log_png_path}")
    print("Family,count,SCV min,SCV max,skewness min,skewness max,kurtosis min,kurtosis max")
    for line in family_summary(rows, families):
        print(
            f"{line[0]},{line[1]},{line[2]:.6g},{line[3]:.6g},"
            f"{line[4]:.6g},{line[5]:.6g},{line[6]:.6g},{line[7]:.6g}"
        )


if __name__ == "__main__":
    main()
