"""
Sample many 100-phase PH distributions from several families and plot histograms.

Families:
    - hyper_erlang
    - coxian
    - general
    - hyper_general

Each PH is scaled to mean 1 before simulation. Histograms show sampled
absorption times from the PH distribution.
"""

import argparse
import math
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample 100-phase PH distributions and plot histograms."
    )
    parser.add_argument("--ph-size", type=int, default=100)
    parser.add_argument("--per-family", type=int, default=10)
    parser.add_argument("--samples-per-distribution", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=r"C:\ph_distribution_histograms",
    )
    return parser.parse_args()


def as_row(x):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return x


def col_ones(n):
    return np.ones((n, 1), dtype=float)


def ph_mean(alpha, t_matrix):
    alpha = as_row(alpha)
    inv = np.linalg.inv(-t_matrix)
    return float((alpha @ inv @ col_ones(t_matrix.shape[0]))[0, 0])


def ph_moment(alpha, t_matrix, k):
    alpha = as_row(alpha)
    inv = np.linalg.inv(-t_matrix)
    power = np.linalg.matrix_power(inv, k)
    return float(math.factorial(k) * (alpha @ power @ col_ones(t_matrix.shape[0]))[0, 0])


def ph_scv(alpha, t_matrix):
    mean = ph_moment(alpha, t_matrix, 1)
    second = ph_moment(alpha, t_matrix, 2)
    return second / mean**2 - 1.0


def scale_ph_to_mean(alpha, t_matrix, target_mean=1.0):
    mean = ph_mean(alpha, t_matrix)
    return alpha, t_matrix * (mean / target_mean)


def random_general_ph(n, rng):
    alpha = rng.dirichlet(np.ones(n)).reshape(1, n)
    rates = np.exp(rng.uniform(np.log(0.05), np.log(20.0), size=n))
    t_matrix = np.zeros((n, n))

    for i in range(n):
        probs = rng.dirichlet(np.ones(n + 1))
        probs_trans = probs[:n].copy()
        probs_abs = probs[-1] + probs_trans[i]
        probs_trans[i] = 0.0

        total = probs_abs + probs_trans.sum()
        probs_abs /= total
        probs_trans /= total

        t_matrix[i, i] = -rates[i]
        for j in range(n):
            if i != j:
                t_matrix[i, j] = rates[i] * probs_trans[j]

    return scale_ph_to_mean(alpha, t_matrix, target_mean=1.0)


def random_coxian_ph(n, rng):
    alpha = np.zeros((1, n))
    alpha[0, 0] = 1.0
    rates = np.exp(rng.uniform(np.log(0.05), np.log(20.0), size=n))
    continue_probs = rng.beta(1.2, 1.2, size=n - 1)
    t_matrix = np.zeros((n, n))

    for i in range(n):
        t_matrix[i, i] = -rates[i]
        if i < n - 1:
            t_matrix[i, i + 1] = continue_probs[i] * rates[i]

    return scale_ph_to_mean(alpha, t_matrix, target_mean=1.0)


def random_hyper_erlang_ph(n, rng):
    max_branches = min(20, n)
    branch_count = int(rng.integers(2, max_branches + 1))
    cuts = sorted(rng.choice(np.arange(1, n), size=branch_count - 1, replace=False))
    sizes = [cuts[0]]
    sizes += [cuts[i] - cuts[i - 1] for i in range(1, branch_count - 1)]
    sizes += [n - cuts[-1]]

    weights = rng.dirichlet(np.ones(branch_count))
    rates = np.exp(rng.uniform(np.log(0.05), np.log(20.0), size=branch_count))

    alpha = np.zeros((1, n))
    t_matrix = np.zeros((n, n))
    idx = 0

    for branch_idx, branch_size in enumerate(sizes):
        alpha[0, idx] = weights[branch_idx]
        rate = rates[branch_idx]
        for j in range(branch_size):
            pos = idx + j
            t_matrix[pos, pos] = -rate
            if j < branch_size - 1:
                t_matrix[pos, pos + 1] = rate
        idx += branch_size

    return scale_ph_to_mean(alpha, t_matrix, target_mean=1.0)


def random_hyper_general_ph(n, rng):
    max_branches = min(10, n)
    branch_count = int(rng.integers(2, max_branches + 1))
    cuts = sorted(rng.choice(np.arange(1, n), size=branch_count - 1, replace=False))
    sizes = [cuts[0]]
    sizes += [cuts[i] - cuts[i - 1] for i in range(1, branch_count - 1)]
    sizes += [n - cuts[-1]]

    weights = rng.dirichlet(np.ones(branch_count))
    alpha = np.zeros((1, n))
    t_matrix = np.zeros((n, n))
    offset = 0

    for branch_idx, branch_size in enumerate(sizes):
        branch_alpha, branch_t = random_general_ph(branch_size, rng)
        alpha[0, offset : offset + branch_size] = weights[branch_idx] * branch_alpha
        t_matrix[offset : offset + branch_size, offset : offset + branch_size] = branch_t
        offset += branch_size

    return scale_ph_to_mean(alpha, t_matrix, target_mean=1.0)


def random_ph_by_family(family, n, rng):
    if family == "hyper_erlang":
        return random_hyper_erlang_ph(n, rng)
    if family == "coxian":
        return random_coxian_ph(n, rng)
    if family == "general":
        return random_general_ph(n, rng)
    if family == "hyper_general":
        return random_hyper_general_ph(n, rng)
    raise ValueError(f"Unknown family: {family}")


def simulate_ph(alpha, t_matrix, sample_count, rng):
    alpha = np.asarray(alpha, dtype=float).ravel()
    n = t_matrix.shape[0]
    samples = np.zeros(sample_count, dtype=float)
    start_states = rng.choice(n, size=sample_count, p=alpha / alpha.sum())

    for sample_idx, state in enumerate(start_states):
        elapsed = 0.0
        while True:
            rate = -t_matrix[state, state]
            elapsed += rng.exponential(1.0 / rate)

            transition_rates = t_matrix[state].copy()
            transition_rates[state] = 0.0
            transient_mass = transition_rates.sum()
            absorb_rate = max(rate - transient_mass, 0.0)
            total = transient_mass + absorb_rate

            if total <= 0.0 or rng.random() < absorb_rate / total:
                samples[sample_idx] = elapsed
                break

            probs = transition_rates / transient_mass
            state = int(rng.choice(n, p=probs))

    return samples


def save_family_histogram(family, rows, output_dir):
    cols = 2
    rows_count = int(np.ceil(len(rows) / cols))
    fig, axes = plt.subplots(rows_count, cols, figsize=(14, 4 * rows_count))
    axes = np.asarray(axes).reshape(-1)

    for ax, row in zip(axes, rows):
        samples = row["samples"]
        cutoff = np.quantile(samples, 0.995)
        ax.hist(samples, bins=80, range=(0.0, cutoff), density=True, alpha=0.82)
        ax.set_title(
            f"{family} #{row['index']} | mean={row['mean']:.3f}, "
            f"SCV={row['scv']:.3f}"
        )
        ax.set_xlabel("sampled PH absorption time")
        ax.set_ylabel("density")

    for ax in axes[len(rows) :]:
        ax.axis("off")

    fig.tight_layout()
    path = output_dir / f"{family}_histograms.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_overlay_histogram(all_rows, output_dir):
    fig, ax = plt.subplots(figsize=(13, 8))
    for family, rows in all_rows.items():
        samples = np.concatenate([row["samples"] for row in rows])
        cutoff = np.quantile(samples, 0.995)
        ax.hist(
            samples,
            bins=100,
            range=(0.0, cutoff),
            density=True,
            histtype="step",
            linewidth=1.6,
            label=family,
        )
    ax.set_title("PH family histogram overlay")
    ax.set_xlabel("sampled PH absorption time")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    path = output_dir / "all_families_overlay.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    families = ["hyper_erlang", "coxian", "general", "hyper_general"]
    all_rows = {}
    summary_rows = []

    print("PH size:", args.ph_size)
    print("Distributions per family:", args.per_family)
    print("Samples per distribution:", args.samples_per_distribution)
    print("Output directory:", output_dir)

    for family in families:
        family_rows = []
        for index in range(args.per_family):
            alpha, t_matrix = random_ph_by_family(family, args.ph_size, rng)
            samples = simulate_ph(
                alpha,
                t_matrix,
                args.samples_per_distribution,
                rng,
            )
            mean = ph_mean(alpha, t_matrix)
            scv = ph_scv(alpha, t_matrix)

            pkl_path = output_dir / f"{family}_{index:03d}_ph_size_{args.ph_size}.pkl"
            with open(pkl_path, "wb") as f:
                pickle.dump(
                    {
                        "family": family,
                        "index": index,
                        "alpha": alpha,
                        "T": t_matrix,
                        "samples": samples,
                        "mean": mean,
                        "scv": scv,
                    },
                    f,
                )

            row = {
                "family": family,
                "index": index,
                "samples": samples,
                "mean": mean,
                "scv": scv,
                "pkl_path": str(pkl_path),
            }
            family_rows.append(row)
            summary_rows.append(row)
            print(
                f"{family} #{index:03d}: mean={mean:.6f}, "
                f"SCV={scv:.6f}, saved={pkl_path}"
            )

        all_rows[family] = family_rows
        figure_path = save_family_histogram(family, family_rows, output_dir)
        print("Saved histogram:", figure_path)

    overlay_path = save_overlay_histogram(all_rows, output_dir)
    print("Saved overlay histogram:", overlay_path)

    summary_path = output_dir / "summary.csv"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("family,index,mean,scv,pkl_path\n")
        for row in summary_rows:
            f.write(
                f"{row['family']},{row['index']},{row['mean']:.12g},"
                f"{row['scv']:.12g},{row['pkl_path']}\n"
            )
    print("Saved summary:", summary_path)


if __name__ == "__main__":
    main()
