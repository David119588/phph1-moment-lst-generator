from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sample_map_ph1_sojourn import (
    MAPMAP1,
    as_mat,
    as_row,
    col_ones,
    lag1_autocorrelation_from_map,
    ph_moments,
    ph_to_renewal_map,
    random_hyperexponential_ph,
    scale_map_to_mean,
    scale_ph_to_mean,
    stationary_distribution,
)

try:
    from butools.ph import MEFromMoments
except Exception as exc:
    raise ImportError(
        "Could not import MEFromMoments from BuTools. Set BUTOOLS_PATH correctly."
    ) from exc


DATASET_PATH = Path("interdeparture_balanced_dataset.pkl")
OUT_DIR = Path("map_ph1_lst_moment_budget_1000")
K_VALUES = tuple(range(1, 16, 2))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Sample MAP/PH/1 queues from the empirical PKL region and test how "
            "many odd moments up to 15 reconstruct the sojourn-time LST."
        )
    )
    parser.add_argument("--num-examples", type=int, default=1000)
    parser.add_argument("--queue-id", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--max-map-size", type=int, default=100)
    parser.add_argument("--min-map-size", type=int, default=2)
    parser.add_argument("--service-size-min", type=int, default=2)
    parser.add_argument("--service-size-max", type=int, default=100)
    parser.add_argument("--rho-min", type=float, default=0.3)
    parser.add_argument("--rho-max", type=float, default=0.99)
    parser.add_argument("--service-scv-min", type=float, default=0.15)
    parser.add_argument("--service-scv-max", type=float, default=20.0)
    parser.add_argument("--candidate-count", type=int, default=60)
    parser.add_argument("--n-s", type=int, default=180)
    parser.add_argument("--s-max-factor", type=float, default=20.0)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--example-id-offset",
        type=int,
        default=0,
        help="Add this offset to example IDs. Useful when running SLURM chunks.",
    )
    return parser.parse_args()


def map_interarrival_moments(d0, d1, max_order=4):
    n = d0.shape[0]
    inv_minus_d0 = np.linalg.inv(-d0)
    p = inv_minus_d0 @ d1
    alpha = stationary_distribution(p - np.eye(n))
    ones = col_ones(n)

    moments = []
    power = np.eye(n)
    for order in range(1, max_order + 1):
        power = power @ inv_minus_d0
        moments.append(float(math.factorial(order) * (alpha @ power @ ones).item()))
    return np.asarray(moments, dtype=float)


def descriptors_from_raw_moments(moments):
    m1, m2, m3, m4 = [float(x) for x in moments[:4]]
    variance = m2 - m1 * m1
    if variance <= 0 or not np.isfinite(variance):
        return np.nan, np.nan, np.nan
    mu3 = m3 - 3.0 * m2 * m1 + 2.0 * m1**3
    mu4 = m4 - 4.0 * m3 * m1 + 6.0 * m2 * m1**2 - 3.0 * m1**4
    return variance / (m1 * m1), mu3 / (variance**1.5), mu4 / (variance * variance)


def empirical_targets(queue_id):
    df = pd.read_pickle(DATASET_PATH)
    df = df[df["queue_id"] == queue_id].copy()
    if df.empty:
        raise ValueError(f"No rows found for queue_id={queue_id}")

    m1 = np.exp(pd.to_numeric(df["log_interdeparture_moment_1"], errors="coerce"))
    m2 = np.exp(pd.to_numeric(df["log_interdeparture_moment_2"], errors="coerce"))
    m3 = np.exp(pd.to_numeric(df["log_interdeparture_moment_3"], errors="coerce"))
    m4 = np.exp(pd.to_numeric(df["log_interdeparture_moment_4"], errors="coerce"))
    variance = m2 - m1 * m1
    mu3 = m3 - 3.0 * m2 * m1 + 2.0 * m1**3
    mu4 = m4 - 4.0 * m3 * m1 + 6.0 * m2 * m1**2 - 3.0 * m1**4

    targets = pd.DataFrame(
        {
            "autocorrelation": pd.to_numeric(df["first_autocorrelation"], errors="coerce"),
            "scv": variance / (m1 * m1),
            "skewness": mu3 / (variance**1.5),
            "kurtosis": mu4 / (variance * variance),
        }
    )
    targets = targets.replace([np.inf, -np.inf], np.nan).dropna()
    targets = targets[(targets[["scv", "skewness", "kurtosis"]] > 0).all(axis=1)].copy()
    return targets


def hard_region_weights(targets):
    corr = targets["autocorrelation"].to_numpy(float)
    scv = targets["scv"].to_numpy(float)
    skew = targets["skewness"].to_numpy(float)
    kurt = targets["kurtosis"].to_numpy(float)

    def rank_weight(values):
        order = np.argsort(values)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.linspace(0.0, 1.0, len(values))
        return ranks

    weights = np.ones(len(targets), dtype=float)
    weights += 2.0 * rank_weight(np.log(scv))
    weights += 2.0 * rank_weight(np.log(skew))
    weights += 2.0 * rank_weight(np.log(kurt))
    weights += 2.0 * (corr < 0.0)
    return weights / weights.sum()


def sample_map_size(rng, min_size, max_size):
    if max_size <= min_size:
        return int(min_size)
    # Bias toward small and medium MAPs, while still allowing up to 100.
    u = rng.random() ** 2.0
    return int(round(min_size + u * (max_size - min_size)))


def sample_mmpp_map_n(rng, size, target_mean=1.0):
    base = float(np.exp(rng.uniform(np.log(0.02), np.log(1.5))))
    spread = float(np.exp(rng.uniform(np.log(2.0), np.log(2000.0))))
    lambdas = base * np.exp(np.linspace(0.0, np.log(spread), size))
    rng.shuffle(lambdas)

    switch = float(np.exp(rng.uniform(np.log(1e-5), np.log(0.4))))
    q = np.zeros((size, size), dtype=float)
    for i in range(size):
        q[i, (i - 1) % size] = switch * np.exp(rng.uniform(-1.0, 1.0))
        q[i, (i + 1) % size] = switch * np.exp(rng.uniform(-1.0, 1.0))

    d1 = np.diag(lambdas)
    d0 = q.copy()
    np.fill_diagonal(d0, -(q.sum(axis=1) + lambdas))
    return (*scale_map_to_mean(d0, d1, target_mean), f"mmpp_n{size}")


def sample_alternating_map_n(rng, size, target_mean=1.0):
    base = float(np.exp(rng.uniform(np.log(0.02), np.log(1.5))))
    spread = float(np.exp(rng.uniform(np.log(2.0), np.log(2000.0))))
    rates = base * np.exp(np.linspace(0.0, np.log(spread), size))
    rng.shuffle(rates)

    silent = float(np.exp(rng.uniform(np.log(1e-4), np.log(3.0))))
    d1 = np.zeros((size, size), dtype=float)
    silent_q = np.zeros((size, size), dtype=float)
    order = np.argsort(rates)
    for rank, i in enumerate(order):
        target = order[(rank + 1) % size]
        d1[i, target] = rates[i]
        silent_q[i, order[(rank - 1) % size]] = silent * np.exp(rng.uniform(-1.0, 1.0))

    d0 = silent_q.copy()
    np.fill_diagonal(d0, -(silent_q.sum(axis=1) + d1.sum(axis=1)))
    return (*scale_map_to_mean(d0, d1, target_mean), f"alternating_n{size}")


def sample_renewal_map_n(rng, size, target_mean=1.0):
    concentration = float(np.exp(rng.uniform(np.log(0.04), np.log(1.2))))
    weights = rng.dirichlet(np.full(size, concentration))
    spread = float(np.exp(rng.uniform(np.log(3.0), np.log(5000.0))))
    base = float(np.exp(rng.uniform(np.log(0.05), np.log(20.0))))
    rates = base * np.exp(rng.uniform(0.0, np.log(spread), size))
    alpha = weights.reshape(1, -1)
    t_matrix = -np.diag(rates)
    alpha, t_matrix = scale_ph_to_mean(alpha, t_matrix, target_mean)
    d0, d1 = ph_to_renewal_map(alpha, t_matrix)
    return d0, d1, f"renewal_n{size}"


def sample_candidate_map(rng, target_corr, min_size, max_size):
    size = sample_map_size(rng, min_size, max_size)
    draw = rng.random()
    if abs(target_corr) < 0.006 and draw < 0.55:
        return sample_renewal_map_n(rng, size, target_mean=1.0)
    if target_corr < -0.01:
        return sample_alternating_map_n(rng, size, target_mean=1.0)
    if target_corr > 0.01:
        return sample_mmpp_map_n(rng, size, target_mean=1.0)
    if draw < 0.45:
        return sample_alternating_map_n(rng, size, target_mean=1.0)
    if draw < 0.90:
        return sample_mmpp_map_n(rng, size, target_mean=1.0)
    return sample_renewal_map_n(rng, size, target_mean=1.0)


def map_feature_error(features, target):
    corr, scv, skew, kurt = features
    return (
        ((corr - target[0]) / 0.01) ** 2
        + ((np.log(max(scv, 1e-12)) - np.log(max(target[1], 1e-12))) / 1.0) ** 2
        + ((np.log(max(skew, 1e-12)) - np.log(max(target[2], 1e-12))) / 1.0) ** 2
        + ((np.log(max(kurt, 1e-12)) - np.log(max(target[3], 1e-12))) / 1.5) ** 2
    )


def sample_map_for_target(target, rng, args):
    best = None
    best_error = float("inf")
    for _ in range(args.candidate_count):
        d0, d1, family = sample_candidate_map(
            rng, target[0], args.min_map_size, args.max_map_size
        )
        corr = float(lag1_autocorrelation_from_map(d0, d1))
        if not np.isfinite(corr):
            continue
        if target[0] >= 0.01 and corr < -0.002:
            continue
        if target[0] <= -0.01 and corr > 0.002:
            continue
        moments = map_interarrival_moments(d0, d1, max_order=4)
        scv, skew, kurt = descriptors_from_raw_moments(moments)
        if not np.all(np.isfinite([scv, skew, kurt])) or min(scv, skew, kurt) <= 0:
            continue
        error = map_feature_error((corr, scv, skew, kurt), target)
        if error < best_error:
            best = d0, d1, family, corr, scv, skew, kurt, error
            best_error = error
    if best is None:
        raise RuntimeError(f"Could not sample MAP for target {target}")
    return best


def parse_mapmap1_result(res):
    if isinstance(res, tuple) and len(res) == 2:
        return as_row(res[0]), as_mat(res[1])
    if isinstance(res, list):
        if len(res) == 2:
            return as_row(res[0]), as_mat(res[1])
        if len(res) == 1 and isinstance(res[0], (tuple, list)) and len(res[0]) == 2:
            return as_row(res[0][0]), as_mat(res[0][1])
    raise ValueError(f"Could not parse MAPMAP1 result: {type(res)}")


def lst_from_me(alpha, t_matrix, s_grid):
    alpha = as_row(alpha)
    t_matrix = as_mat(t_matrix)
    n = t_matrix.shape[0]
    eye = np.eye(n)
    exit_vec = (-t_matrix) @ col_ones(n)
    values = []
    for s in s_grid:
        values.append(float((alpha @ np.linalg.solve(s * eye - t_matrix, exit_vec)).item()))
    return np.asarray(values, dtype=float)


def normalize_moments_by_mean(moments):
    moments = np.asarray(moments, dtype=float)
    mean = float(moments[0])
    if mean <= 0 or not np.isfinite(mean):
        raise ValueError(f"Bad mean: {mean}")
    return np.asarray([moments[k - 1] / mean**k for k in range(1, len(moments) + 1)]), mean


def rescale_me(alpha, t_matrix, mean):
    return as_row(alpha), as_mat(t_matrix) / mean


def safe_l2(fitted, true, grid):
    fitted = np.asarray(fitted, dtype=float)
    true = np.asarray(true, dtype=float)
    if not np.all(np.isfinite(fitted)) or not np.all(np.isfinite(true)):
        return np.nan
    return float(np.sqrt(np.trapezoid((fitted - true) ** 2, grid)))


def fit_lsts_for_example(beta, t_soj, s_grid):
    true_moments = ph_moments(beta, t_soj, max(K_VALUES))
    true_lst = lst_from_me(beta, t_soj, s_grid)
    rows = []
    for k in K_VALUES:
        try:
            fit_input, mean = normalize_moments_by_mean(true_moments[:k])
            alpha_norm, t_norm = MEFromMoments(fit_input.tolist())
            alpha_hat, t_hat = rescale_me(alpha_norm, t_norm, mean)
            fitted_lst = lst_from_me(alpha_hat, t_hat, s_grid)
            l2 = safe_l2(fitted_lst, true_lst, s_grid)
            status = "ok"
            order = int(t_hat.shape[0])
        except Exception as exc:
            l2 = np.nan
            status = f"{type(exc).__name__}: {exc}"
            order = np.nan
        rows.append({"K": k, "l2": l2, "fit_order": order, "fit_status": status})
    return rows


def plot_l2(summary, output_dir):
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
    path = output_dir / "map_ph1_lst_mean_l2_vs_odd_moments_up_to_15.png"
    fig.savefig(path, dpi=250)
    plt.close(fig)
    return path


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    targets = empirical_targets(args.queue_id)
    weights = hard_region_weights(targets)
    target_array = targets[["autocorrelation", "scv", "skewness", "kurtosis"]].to_numpy(float)

    example_rows = []
    fit_rows = []

    for index in range(args.num_examples):
        target_index = int(rng.choice(len(target_array), p=weights))
        target = target_array[target_index]
        service_size = np.nan

        try:
            d0, d1, family, corr, arr_scv, arr_skew, arr_kurt, map_match_error = (
                sample_map_for_target(target, rng, args)
            )
            rho = float(rng.uniform(args.rho_min, args.rho_max))
            service_size = int(rng.integers(args.service_size_min, args.service_size_max + 1))
            beta_s, t_s, service_scv = random_hyperexponential_ph(
                service_size,
                target_mean=rho,
                scv_min=args.service_scv_min,
                scv_max=args.service_scv_max,
                rng=rng,
            )
            s0, s1 = ph_to_renewal_map(beta_s, t_s)
            beta_w, t_w = parse_mapmap1_result(MAPMAP1(d0, d1, s0, s1, "stDistrME"))

            mean_w = float(ph_moments(beta_w, t_w, 1)[0])
            s_grid = np.linspace(0.0, args.s_max_factor / max(mean_w, 1e-12), args.n_s)
            rows = fit_lsts_for_example(beta_w, t_w, s_grid)
            status = "ok"
        except Exception as exc:
            rows = [{"K": k, "l2": np.nan, "fit_order": np.nan, "fit_status": f"sample failed: {type(exc).__name__}: {exc}"} for k in K_VALUES]
            family = "failed"
            corr = arr_scv = arr_skew = arr_kurt = service_scv = rho = mean_w = map_match_error = np.nan
            d0 = np.empty((0, 0))
            t_w = np.empty((0, 0))
            status = rows[0]["fit_status"]

        example_id = args.example_id_offset + index + 1
        example_rows.append(
            {
                "example_id": example_id,
                "status": status,
                "map_family": family,
                "map_size": int(d0.shape[0]),
                "sojourn_order": int(t_w.shape[0]),
                "target_autocorrelation": target[0],
                "target_scv": target[1],
                "target_skewness": target[2],
                "target_kurtosis": target[3],
                "arrival_autocorrelation": corr,
                "arrival_scv": arr_scv,
                "arrival_skewness": arr_skew,
                "arrival_kurtosis": arr_kurt,
                "map_match_error": map_match_error,
                "rho": rho,
                "service_size": service_size if "service_size" in locals() else np.nan,
                "service_scv": service_scv,
                "sojourn_mean": mean_w,
            }
        )
        for row in rows:
            row = dict(row)
            row["example_id"] = example_id
            fit_rows.append(row)

        if (index + 1) % args.progress_every == 0 or index + 1 == args.num_examples:
            print(f"Processed {index + 1}/{args.num_examples}")

    examples = pd.DataFrame(example_rows)
    fits = pd.DataFrame(fit_rows)
    examples_path = args.output_dir / "map_ph1_examples_summary.csv"
    fits_path = args.output_dir / "map_ph1_lst_fit_results.csv"
    examples.to_csv(examples_path, index=False)
    fits.to_csv(fits_path, index=False)

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
    )
    summary_path = args.output_dir / "map_ph1_lst_l2_summary_by_K.csv"
    summary.to_csv(summary_path, index=False)
    plot_path = plot_l2(summary, args.output_dir)

    print("Saved examples:", examples_path)
    print("Saved fits:", fits_path)
    print("Saved summary:", summary_path)
    print("Saved plot:", plot_path)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
