"""
Sample MAP/PH/1 queues and compute sojourn-time moments.

This script is intended for the "second queue" experiment where arrivals are a
MAP instead of renewal PH. It samples a 2-state MMPP/MAP arrival process, samples
a PH service-time distribution, computes the MAP/PH/1 sojourn-time ME
representation with BuTools MAPMAP1, and saves:

    - one PKL per example
    - a manifest CSV
    - summary plots

The arrival mean is scaled to 1, so the sampled service mean is also the
utilization rho. By default rho is sampled uniformly in [0.3, 0.99].
"""

import argparse
import csv
import math
import os
import pickle
import re
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def add_butools_paths():
    candidates = []
    env_path = os.environ.get("BUTOOLS_PATH")
    if env_path:
        candidates.append(Path(env_path))

    here = Path(__file__).resolve().parent
    candidates.extend(
        [
            here / "butools" / "Python",
            here / "butools2" / "Python",
            here.parent / "butools" / "Python",
            here.parent / "butools2" / "Python",
            Path(r"C:\Users\osamb\Downloads\butools2 (2)\butools2\Python"),
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            text = str(candidate)
            if text not in sys.path:
                sys.path.append(text)


add_butools_paths()

try:
    import butools
    from butools.queues import MAPMAP1

    butools.verbose = False
    butools.checkInput = False
except Exception as exc:
    raise ImportError(
        "Could not import BuTools. Set BUTOOLS_PATH to the BuTools Python folder. "
        f"Original error: {type(exc).__name__}: {exc}"
    )


FILENAME_RE = re.compile(r"_ex_(\d+)\.pkl$")


def parse_args():
    parser = argparse.ArgumentParser(description="Sample MAP/PH/1 sojourn examples.")
    parser.add_argument("--output-dir", type=Path, default=Path(r"C:\map_ph1_second_queue"))
    parser.add_argument("--num-examples", type=int, default=1000)
    parser.add_argument("--job-index", type=int, default=None)
    parser.add_argument("--example-start", type=int, default=None)
    parser.add_argument("--resume", type=int, default=1)
    parser.add_argument("--clean-output", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--service-size",
        type=int,
        default=20,
        help="Fixed PH service size when --random-service-size is 0.",
    )
    parser.add_argument(
        "--random-service-size",
        type=int,
        default=1,
        help="Use 1 to sample the PH service size for each example.",
    )
    parser.add_argument(
        "--service-size-min",
        type=int,
        default=2,
        help="Minimum sampled PH service size when --random-service-size is 1.",
    )
    parser.add_argument(
        "--service-size-max",
        type=int,
        default=100,
        help="Maximum sampled PH service size when --random-service-size is 1.",
    )
    parser.add_argument("--service-mean-min", type=float, default=0.3)
    parser.add_argument("--service-mean-max", type=float, default=0.99)
    parser.add_argument("--service-scv-min", type=float, default=0.15)
    parser.add_argument("--service-scv-max", type=float, default=20.0)
    parser.add_argument("--map-rate-ratio-min", type=float, default=1.5)
    parser.add_argument("--map-rate-ratio-max", type=float, default=20.0)
    parser.add_argument("--map-switch-rate-min", type=float, default=0.02)
    parser.add_argument("--map-switch-rate-max", type=float, default=3.0)
    parser.add_argument(
        "--map-corr-mode",
        choices=("near_zero_mixed", "near_zero", "mixed", "positive", "negative"),
        default="near_zero_mixed",
        help=(
            "MAP autocorrelation family. near_zero_mixed samples weak positive "
            "and weak negative autocorrelation around zero, near_zero samples a "
            "weak positive 2-state MAP, positive uses an MMPP-like MAP, negative "
            "uses an alternating-arrival MAP, and mixed samples broad positive "
            "and negative autocorrelation."
        ),
    )
    parser.add_argument(
        "--map-corr-min",
        type=float,
        default=-0.02,
        help=(
            "Lower target lag-1 autocorrelation for --map-corr-mode near_zero_mixed. "
            "Default is based on the central empirical interdeparture range."
        ),
    )
    parser.add_argument(
        "--map-corr-max",
        type=float,
        default=0.035,
        help=(
            "Upper target lag-1 autocorrelation for --map-corr-mode near_zero_mixed. "
            "Default is based on the central empirical interdeparture range."
        ),
    )
    parser.add_argument(
        "--map-negative-prob",
        type=float,
        default=0.5,
        help="Probability of sampling the negative-correlation MAP when --map-corr-mode mixed.",
    )
    parser.add_argument("--num-moments", type=int, default=20)
    parser.add_argument("--plot", type=int, default=1)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print one progress line every N generated examples.",
    )
    return parser.parse_args()


def as_row(x):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return x


def as_mat(x):
    return np.asarray(x, dtype=float)


def col_ones(n):
    return np.ones((n, 1), dtype=float)


def ph_moments(alpha, t_matrix, n_moms):
    alpha = as_row(alpha)
    t_matrix = as_mat(t_matrix)
    inv_minus_t = np.linalg.inv(-t_matrix)
    power = np.eye(t_matrix.shape[0])
    ones = col_ones(t_matrix.shape[0])

    moments = []
    for k in range(1, n_moms + 1):
        power = power @ inv_minus_t
        moments.append(math.factorial(k) * (alpha @ power @ ones).item())
    return np.asarray(moments, dtype=float)


def ph_mean(alpha, t_matrix):
    return float(ph_moments(alpha, t_matrix, 1)[0])


def ph_scv(alpha, t_matrix):
    m1, m2 = ph_moments(alpha, t_matrix, 2)
    return float((m2 - m1 * m1) / (m1 * m1))


def scale_ph_to_mean(alpha, t_matrix, target_mean):
    current_mean = ph_mean(alpha, t_matrix)
    if current_mean <= 0 or not np.isfinite(current_mean):
        raise ValueError(f"Bad PH mean: {current_mean}")
    return as_row(alpha), as_mat(t_matrix) * (current_mean / target_mean)


def random_hyperexponential_ph(size, target_mean, scv_min, scv_max, rng):
    if size < 2:
        raise ValueError("--service-size must be at least 2")

    # Hyperexponential PH: alpha chooses the phase, T is diagonal.
    for _ in range(10000):
        weights = rng.dirichlet(np.ones(size))
        rates = np.exp(rng.uniform(np.log(0.05), np.log(30.0), size))
        alpha = weights.reshape(1, -1)
        t_matrix = -np.diag(rates)
        alpha, t_matrix = scale_ph_to_mean(alpha, t_matrix, target_mean)
        scv = ph_scv(alpha, t_matrix)
        if scv_min <= scv <= scv_max:
            return alpha, t_matrix, scv

    # Fallback: accept the last sampled PH rather than failing the whole job.
    return alpha, t_matrix, scv


def choose_service_size(args, rng):
    if not args.random_service_size:
        return int(args.service_size)
    return int(rng.integers(args.service_size_min, args.service_size_max + 1))


def ph_to_renewal_map(alpha, t_matrix):
    alpha = as_row(alpha)
    t_matrix = as_mat(t_matrix)
    exit_vec = (-t_matrix) @ col_ones(t_matrix.shape[0])
    s0 = t_matrix.copy()
    s1 = exit_vec @ alpha
    return s0, s1


def stationary_distribution(q):
    n = q.shape[0]
    a = q.T.copy()
    b = np.zeros(n)
    a[-1, :] = 1.0
    b[-1] = 1.0
    return np.linalg.solve(a, b).reshape(1, -1)


def map_arrival_rate(d0, d1):
    q = d0 + d1
    pi = stationary_distribution(q)
    return (pi @ d1 @ col_ones(d0.shape[0])).item()


def scale_map_to_mean(d0, d1, target_mean):
    rate = map_arrival_rate(d0, d1)
    target_rate = 1.0 / target_mean
    scale = target_rate / rate
    return d0 * scale, d1 * scale


def sample_mMPP_map(rng, target_mean=1.0, rate_ratio_min=1.5, rate_ratio_max=20.0,
                    switch_rate_min=0.02, switch_rate_max=3.0):
    """
    Sample a 2-state Markov-modulated Poisson process as a MAP.

    D0 contains state transitions without arrivals and Poisson arrival rates on
    the diagonal; D1 is diagonal with arrival rates.
    """
    low_rate = float(np.exp(rng.uniform(np.log(0.05), np.log(2.0))))
    ratio = float(np.exp(rng.uniform(np.log(rate_ratio_min), np.log(rate_ratio_max))))
    high_rate = low_rate * ratio
    q01 = float(np.exp(rng.uniform(np.log(switch_rate_min), np.log(switch_rate_max))))
    q10 = float(np.exp(rng.uniform(np.log(switch_rate_min), np.log(switch_rate_max))))

    if rng.random() < 0.5:
        lambdas = np.array([low_rate, high_rate], dtype=float)
    else:
        lambdas = np.array([high_rate, low_rate], dtype=float)

    d1 = np.diag(lambdas)
    d0 = np.array(
        [
            [-(q01 + lambdas[0]), q01],
            [q10, -(q10 + lambdas[1])],
        ],
        dtype=float,
    )
    d0, d1 = scale_map_to_mean(d0, d1, target_mean)
    return d0, d1, "positive_mmpp"


def sample_near_zero_map(rng, target_mean=1.0):
    """
    Sample a weakly correlated 2-state MAP.

    This is the default because the measured interdeparture autocorrelations in
    the extracted queue data are close to zero. The two arrival rates are close
    to each other and the hidden-state switching is fast, so lag-1 dependence is
    small while the process is still represented as a MAP.
    """
    base_rate = float(np.exp(rng.uniform(np.log(0.8), np.log(1.25))))
    rate_ratio = float(np.exp(rng.uniform(np.log(1.0), np.log(1.003))))
    lambdas = np.array([base_rate, base_rate * rate_ratio], dtype=float)
    if rng.random() < 0.5:
        lambdas = lambdas[::-1]

    q01 = float(np.exp(rng.uniform(np.log(100.0), np.log(500.0))))
    q10 = float(np.exp(rng.uniform(np.log(100.0), np.log(500.0))))

    d1 = np.diag(lambdas)
    d0 = np.array(
        [
            [-(q01 + lambdas[0]), q01],
            [q10, -(q10 + lambdas[1])],
        ],
        dtype=float,
    )
    d0, d1 = scale_map_to_mean(d0, d1, target_mean)
    return d0, d1, "near_zero_map"


def sample_zero_corr_map(rng, target_mean=1.0):
    """
    Sample a MAP representation of an almost Poisson arrival process.

    The two hidden states have the same arrival rate. This fills the region
    around autocorrelation zero instead of creating two separated clusters.
    """
    arrival_rate = float(np.exp(rng.uniform(np.log(0.8), np.log(1.25))))
    q01 = float(np.exp(rng.uniform(np.log(20.0), np.log(200.0))))
    q10 = float(np.exp(rng.uniform(np.log(20.0), np.log(200.0))))

    d1 = np.diag([arrival_rate, arrival_rate])
    d0 = np.array(
        [
            [-(q01 + arrival_rate), q01],
            [q10, -(q10 + arrival_rate)],
        ],
        dtype=float,
    )
    d0, d1 = scale_map_to_mean(d0, d1, target_mean)
    return d0, d1, "zero_corr_map"


def sample_near_zero_negative_map(rng, target_mean=1.0):
    """
    Sample a weak alternating MAP with small negative lag-1 autocorrelation.
    """
    base_rate = float(np.exp(rng.uniform(np.log(0.8), np.log(1.25))))
    rate_ratio = float(np.exp(rng.uniform(np.log(1.0), np.log(1.003))))
    high_rate = base_rate * rate_ratio
    low_rate = base_rate

    arrival_switch_01 = high_rate
    arrival_switch_10 = low_rate
    no_arrival_switch_01 = float(np.exp(rng.uniform(np.log(100.0), np.log(500.0))))
    no_arrival_switch_10 = float(np.exp(rng.uniform(np.log(100.0), np.log(500.0))))

    if rng.random() < 0.5:
        d1 = np.array([[0.0, arrival_switch_01], [arrival_switch_10, 0.0]], dtype=float)
    else:
        d1 = np.array([[0.0, arrival_switch_10], [arrival_switch_01, 0.0]], dtype=float)

    d0 = np.array(
        [
            [-(no_arrival_switch_01 + d1[0].sum()), no_arrival_switch_01],
            [no_arrival_switch_10, -(no_arrival_switch_10 + d1[1].sum())],
        ],
        dtype=float,
    )
    d0, d1 = scale_map_to_mean(d0, d1, target_mean)
    return d0, d1, "near_zero_negative_map"


def sample_alternating_map(rng, target_mean=1.0, rate_ratio_min=1.5, rate_ratio_max=20.0,
                           switch_rate_min=0.02, switch_rate_max=3.0):
    """
    Sample a 2-state MAP where arrivals tend to switch the hidden state.

    This creates alternating short/long interarrival periods when the two state
    rates differ, so the lag-1 interarrival autocorrelation can be negative.
    """
    low_rate = float(np.exp(rng.uniform(np.log(0.05), np.log(2.0))))
    ratio = float(np.exp(rng.uniform(np.log(rate_ratio_min), np.log(rate_ratio_max))))
    high_rate = low_rate * ratio

    arrival_switch_01 = high_rate
    arrival_switch_10 = low_rate
    no_arrival_switch_01 = float(
        np.exp(rng.uniform(np.log(switch_rate_min), np.log(switch_rate_max)))
    )
    no_arrival_switch_10 = float(
        np.exp(rng.uniform(np.log(switch_rate_min), np.log(switch_rate_max)))
    )

    if rng.random() < 0.5:
        d1 = np.array([[0.0, arrival_switch_01], [arrival_switch_10, 0.0]], dtype=float)
    else:
        d1 = np.array([[0.0, arrival_switch_10], [arrival_switch_01, 0.0]], dtype=float)

    d0 = np.array(
        [
            [-(no_arrival_switch_01 + d1[0].sum()), no_arrival_switch_01],
            [no_arrival_switch_10, -(no_arrival_switch_10 + d1[1].sum())],
        ],
        dtype=float,
    )
    d0, d1 = scale_map_to_mean(d0, d1, target_mean)
    return d0, d1, "negative_alternating"


def sample_continuous_near_zero_map(rng, target_mean=1.0, corr_min=-0.015, corr_max=0.015):
    """
    Sample a weakly correlated MAP with lag-1 autocorrelation spread around zero.

    We first draw a target autocorrelation uniformly in the requested interval.
    Then we generate candidate positive or negative 2-state MAPs and keep the
    candidate whose measured lag-1 autocorrelation is closest to that target.
    This avoids the artificial spike that happens when many examples are forced
    to be exactly zero-correlation MAPs.
    """
    target_corr = float(rng.uniform(corr_min, corr_max))
    prefer_negative = target_corr < 0.0
    best = None
    best_error = float("inf")

    for _ in range(500):
        target_abs = abs(target_corr)
        if target_abs > 0.05:
            rate_ratio_min = float(np.exp(rng.uniform(np.log(1.5), np.log(10.0))))
            rate_ratio_max = float(np.exp(rng.uniform(np.log(rate_ratio_min), np.log(80.0))))
            switch_rate_min = float(np.exp(rng.uniform(np.log(0.001), np.log(0.5))))
            switch_rate_max = float(np.exp(rng.uniform(np.log(switch_rate_min), np.log(5.0))))
        elif target_abs > 0.02:
            rate_ratio_min = float(np.exp(rng.uniform(np.log(1.05), np.log(3.0))))
            rate_ratio_max = float(np.exp(rng.uniform(np.log(rate_ratio_min), np.log(20.0))))
            switch_rate_min = float(np.exp(rng.uniform(np.log(0.01), np.log(2.0))))
            switch_rate_max = float(np.exp(rng.uniform(np.log(switch_rate_min), np.log(50.0))))
        else:
            rate_ratio_min = float(np.exp(rng.uniform(np.log(1.001), np.log(1.03))))
            rate_ratio_max = float(np.exp(rng.uniform(np.log(rate_ratio_min), np.log(1.35))))
            switch_rate_min = float(np.exp(rng.uniform(np.log(5.0), np.log(80.0))))
            switch_rate_max = float(np.exp(rng.uniform(np.log(switch_rate_min), np.log(600.0))))

        if prefer_negative:
            d0, d1, base_kind = sample_alternating_map(
                rng,
                target_mean=target_mean,
                rate_ratio_min=rate_ratio_min,
                rate_ratio_max=rate_ratio_max,
                switch_rate_min=switch_rate_min,
                switch_rate_max=switch_rate_max,
            )
        else:
            d0, d1, base_kind = sample_mMPP_map(
                rng,
                target_mean=target_mean,
                rate_ratio_min=rate_ratio_min,
                rate_ratio_max=rate_ratio_max,
                switch_rate_min=switch_rate_min,
                switch_rate_max=switch_rate_max,
            )

        corr = float(lag1_autocorrelation_from_map(d0, d1))
        if not np.isfinite(corr):
            continue
        if prefer_negative and corr > 0.001:
            continue
        if not prefer_negative and corr < -0.001:
            continue
        if corr < corr_min or corr > corr_max:
            continue

        error = abs(corr - target_corr)
        if error < best_error:
            best = (d0, d1, f"continuous_near_zero_{base_kind}")
            best_error = error
        tolerance = max(0.00025, 0.02 * max(abs(target_corr), 0.001))
        if best_error <= tolerance:
            break

    if best is not None:
        return best

    if prefer_negative:
        return sample_near_zero_negative_map(rng, target_mean=target_mean)
    return sample_near_zero_map(rng, target_mean=target_mean)


def sample_map_arrival(args, rng):
    if args.map_corr_mode == "near_zero_mixed":
        return sample_continuous_near_zero_map(
            rng,
            target_mean=1.0,
            corr_min=args.map_corr_min,
            corr_max=args.map_corr_max,
        )

    if args.map_corr_mode == "near_zero":
        return sample_near_zero_map(rng, target_mean=1.0)

    if args.map_corr_mode == "positive":
        return sample_mMPP_map(
            rng,
            target_mean=1.0,
            rate_ratio_min=args.map_rate_ratio_min,
            rate_ratio_max=args.map_rate_ratio_max,
            switch_rate_min=args.map_switch_rate_min,
            switch_rate_max=args.map_switch_rate_max,
        )

    if args.map_corr_mode == "negative":
        return sample_alternating_map(
            rng,
            target_mean=1.0,
            rate_ratio_min=args.map_rate_ratio_min,
            rate_ratio_max=args.map_rate_ratio_max,
            switch_rate_min=args.map_switch_rate_min,
            switch_rate_max=args.map_switch_rate_max,
        )

    if rng.random() < args.map_negative_prob:
        return sample_alternating_map(
            rng,
            target_mean=1.0,
            rate_ratio_min=args.map_rate_ratio_min,
            rate_ratio_max=args.map_rate_ratio_max,
            switch_rate_min=args.map_switch_rate_min,
            switch_rate_max=args.map_switch_rate_max,
        )

    return sample_mMPP_map(
        rng,
        target_mean=1.0,
        rate_ratio_min=args.map_rate_ratio_min,
        rate_ratio_max=args.map_rate_ratio_max,
        switch_rate_min=args.map_switch_rate_min,
        switch_rate_max=args.map_switch_rate_max,
    )


def lag1_autocorrelation_from_map(d0, d1):
    """
    Estimate lag-1 autocorrelation from the embedded post-arrival phase chain.
    """
    n = d0.shape[0]
    inv_minus_d0 = np.linalg.inv(-d0)
    p = inv_minus_d0 @ d1
    pi = stationary_distribution(p - np.eye(n))
    mean_by_phase = inv_minus_d0 @ col_ones(n)
    mean = (pi @ mean_by_phase).item()
    centered = mean_by_phase - mean
    var = (pi @ (centered * centered)).item()
    if var <= 1e-12:
        return 0.0
    cov = (pi @ (centered * (p @ centered))).item()
    return cov / var


def parse_mapmap1_st_distr_me_result(res):
    if isinstance(res, tuple) and len(res) == 2:
        return as_row(res[0]), as_mat(res[1])
    if isinstance(res, list):
        if len(res) == 2:
            return as_row(res[0]), as_mat(res[1])
        if len(res) == 1:
            inner = res[0]
            if isinstance(inner, (tuple, list)) and len(inner) == 2:
                return as_row(inner[0]), as_mat(inner[1])
    raise ValueError(f"Could not parse MAPMAP1 stDistrME result: {type(res)} {res}")


def log_checked(values, label):
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError(f"{label} moments must be positive and finite: {values}")
    return np.log(values)


def resolved_job_index(args):
    if args.job_index is not None:
        return args.job_index
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    return int(task_id) if task_id is not None else 0


def existing_example_ids(output_dir):
    existing = set()
    for path in output_dir.glob("*.pkl"):
        match = FILENAME_RE.search(path.name)
        if match is not None:
            existing.add(int(match.group(1)))
    return existing


def example_rng(seed, example_id):
    return np.random.default_rng(np.random.SeedSequence([seed, example_id]))


def build_output_path(output_dir, example_id, rho, autocorr):
    return output_dir / (
        f"map_ph1_queue2_rho_{rho:.6f}_corr_{autocorr:.6f}_ex_{example_id:08d}.pkl"
    )


def generate_example(args, example_id):
    rng = example_rng(args.seed, example_id)
    d0, d1, map_type = sample_map_arrival(args, rng)

    rho = float(rng.uniform(args.service_mean_min, args.service_mean_max))
    service_size = choose_service_size(args, rng)
    service_alpha, service_t, service_scv = random_hyperexponential_ph(
        service_size,
        target_mean=rho,
        scv_min=args.service_scv_min,
        scv_max=args.service_scv_max,
        rng=rng,
    )
    s0, s1 = ph_to_renewal_map(service_alpha, service_t)
    beta, t_sojourn = parse_mapmap1_st_distr_me_result(
        MAPMAP1(d0, d1, s0, s1, "stDistrME")
    )

    service_moments = ph_moments(service_alpha, service_t, 10)
    sojourn_moments = ph_moments(beta, t_sojourn, args.num_moments)
    autocorr = float(lag1_autocorrelation_from_map(d0, d1))

    return {
        "example_id": example_id,
        "rho": rho,
        "arrival_mean": 1.0,
        "map_type": map_type,
        "service_mean": ph_mean(service_alpha, service_t),
        "service_size": int(service_size),
        "service_scv": service_scv,
        "map_lag1_autocorrelation": autocorr,
        "D0": d0.astype(float),
        "D1": d1.astype(float),
        "service_alpha": service_alpha.astype(float),
        "service_T": service_t.astype(float),
        "sojourn_beta": beta.astype(float),
        "sojourn_T": t_sojourn.astype(float),
        "log_service_moments": log_checked(service_moments, "Service").astype(float),
        "log_sojourn_moments": log_checked(sojourn_moments, "Sojourn").astype(float),
        "sojourn_mean": float(sojourn_moments[0]),
        "sojourn_order": int(t_sojourn.shape[0]),
    }


def clean_output_dir(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.iterdir():
        if path.is_file() and path.suffix.lower() in {".pkl", ".csv", ".png"}:
            path.unlink()


def write_manifest(output_dir, rows, suffix):
    if not rows:
        return None
    path = output_dir / f"map_ph1_manifest{suffix}.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot_summary(output_dir, rows):
    if not rows:
        return []

    rho = np.asarray([row["rho"] for row in rows], dtype=float)
    corr = np.asarray([row["map_lag1_autocorrelation"] for row in rows], dtype=float)
    soj_mean = np.asarray([row["sojourn_mean"] for row in rows], dtype=float)
    service_scv = np.asarray([row["service_scv"] for row in rows], dtype=float)

    paths = []

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(rho, bins=40, color="#28666e", edgecolor="white")
    axes[0].set_title("Sampled utilization across models")
    axes[0].set_xlabel("rho")
    axes[0].set_ylabel("count")
    axes[1].hist(corr, bins=40, color="#b3532f", edgecolor="white")
    axes[1].set_title("Sampled arrival MAP lag-1 autocorrelation")
    axes[1].set_xlabel("arrival MAP autocorrelation")
    axes[2].hist(np.log(soj_mean), bins=40, color="#585858", edgecolor="white")
    axes[2].set_title("Log sojourn mean across models")
    axes[2].set_xlabel("log E[W]")
    fig.tight_layout()
    path = output_dir / "map_ph1_histograms.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    fig, axis = plt.subplots(figsize=(8, 5))
    scatter = axis.scatter(rho, np.log(soj_mean), c=corr, s=14, alpha=0.75, cmap="viridis")
    axis.set_title("MAP/PH/1 sojourn area")
    axis.set_xlabel("utilization rho")
    axis.set_ylabel("log sojourn mean")
    colorbar = fig.colorbar(scatter, ax=axis)
    colorbar.set_label("arrival MAP lag-1 autocorrelation")
    fig.tight_layout()
    path = output_dir / "map_ph1_sojourn_area.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    fig, axis = plt.subplots(figsize=(8, 5))
    scatter = axis.scatter(service_scv, np.log(soj_mean), c=rho, s=14, alpha=0.75, cmap="plasma")
    axis.set_xscale("log")
    axis.set_title("Service variability vs MAP/PH/1 sojourn")
    axis.set_xlabel("service SCV")
    axis.set_ylabel("log sojourn mean")
    colorbar = fig.colorbar(scatter, ax=axis)
    colorbar.set_label("rho")
    fig.tight_layout()
    path = output_dir / "map_ph1_service_scv_vs_sojourn.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    return paths


def main():
    args = parse_args()
    if args.num_examples < 1:
        raise ValueError("--num-examples must be at least 1")
    if args.service_mean_min <= 0 or args.service_mean_max >= 1:
        raise ValueError("Require 0 < service mean min/max < 1")
    if args.service_mean_min > args.service_mean_max:
        raise ValueError("--service-mean-min must be <= --service-mean-max")
    if args.service_size < 2:
        raise ValueError("--service-size must be at least 2")
    if args.service_size_min < 2:
        raise ValueError("--service-size-min must be at least 2")
    if args.service_size_min > args.service_size_max:
        raise ValueError("--service-size-min must be <= --service-size-max")
    if args.map_corr_min >= args.map_corr_max:
        raise ValueError("--map-corr-min must be smaller than --map-corr-max")
    if args.map_corr_min < -1.0 or args.map_corr_max > 1.0:
        raise ValueError("MAP autocorrelation targets must stay inside [-1, 1]")
    if args.map_negative_prob < 0.0 or args.map_negative_prob > 1.0:
        raise ValueError("--map-negative-prob must be between 0 and 1")
    if args.progress_every < 1:
        raise ValueError("--progress-every must be at least 1")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.clean_output:
        clean_output_dir(output_dir)

    job_index = resolved_job_index(args)
    first_example = (
        job_index * args.num_examples if args.example_start is None else args.example_start
    )
    existing = existing_example_ids(output_dir) if args.resume else set()

    print("MAP/PH/1 second queue sampler")
    print("Job index:", job_index)
    print("Examples:", args.num_examples)
    print("Global examples:", first_example, "to", first_example + args.num_examples - 1)
    print("Output dir:", output_dir)
    print("Service mean/rho range:", args.service_mean_min, args.service_mean_max)
    print("MAP correlation mode:", args.map_corr_mode)
    print("MAP negative sampling probability:", args.map_negative_prob)
    if args.random_service_size:
        print("Random service PH size range:", args.service_size_min, args.service_size_max)
    else:
        print("Fixed service PH size:", args.service_size)
    print("Service SCV range:", args.service_scv_min, args.service_scv_max)

    rows = []
    start_all = time.perf_counter()
    for local_id in range(args.num_examples):
        example_id = first_example + local_id
        if example_id in existing:
            print(f"Skipping existing example {example_id}")
            continue

        start = time.perf_counter()
        payload = generate_example(args, example_id)
        output_path = build_output_path(
            output_dir,
            example_id,
            payload["rho"],
            payload["map_lag1_autocorrelation"],
        )
        with open(output_path, "wb") as f:
            pickle.dump(payload, f)

        rows.append(
            {
                "example_id": example_id,
                "path": str(output_path),
                "rho": payload["rho"],
                "map_type": payload["map_type"],
                "service_mean": payload["service_mean"],
                "service_size": payload["service_size"],
                "service_scv": payload["service_scv"],
                "map_lag1_autocorrelation": payload["map_lag1_autocorrelation"],
                "sojourn_mean": payload["sojourn_mean"],
                "sojourn_order": payload["sojourn_order"],
            }
        )
        if (local_id + 1) % args.progress_every == 0 or local_id + 1 == args.num_examples:
            elapsed = time.perf_counter() - start_all
            print(
                f"Saved {local_id + 1}/{args.num_examples} examples "
                f"(latest id {example_id}): rho={payload['rho']:.4f}, "
                f"corr={payload['map_lag1_autocorrelation']:.4f}, "
                f"E[W]={payload['sojourn_mean']:.4g}, "
                f"elapsed={elapsed:.1f}s"
            )

    suffix = f"_job_{job_index:04d}_ex_{first_example:08d}_{first_example + args.num_examples - 1:08d}"
    manifest_path = write_manifest(output_dir, rows, suffix)
    if manifest_path:
        print("Saved manifest:", manifest_path)

    if args.plot and rows:
        for path in plot_summary(output_dir, rows):
            print("Saved plot:", path)

    print(f"Total time: {time.perf_counter() - start_all:.3f}s")


if __name__ == "__main__":
    main()
