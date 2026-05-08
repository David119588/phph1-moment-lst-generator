# %%
# ============================================================
# CELL 1:
# Imports and BuTools setup
# ============================================================

import os
import sys
import math
import time
import pickle
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
# PyCharm compatibility: define display() like in Jupyter
try:
    display
except NameError:
    def display(x):
        print(x)

# ------------------------------------------------------------
# OPTIONAL:
# If BuTools is not installed globally, set BUTOOLS_PATH.
# Linux example:
# export BUTOOLS_PATH=/scratch200/davidfine/butools2/Python
# ------------------------------------------------------------

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
            here / "butools2" / "butools2" / "Python",
            here.parent / "butools" / "Python",
            here.parent / "butools2" / "Python",
            here.parent / "butools2" / "butools2" / "Python",
            Path(r"C:\Users\osamb\Downloads\butools2 (2)\butools2\Python"),
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            candidate_text = str(candidate)
            if candidate_text not in sys.path:
                sys.path.append(candidate_text)


add_butools_paths()

# ------------------------------------------------------------
# Import BuTools
# ------------------------------------------------------------

try:
    import butools
    butools.verbose = False
    butools.checkInput = False

    from butools.ph import MEFromMoments
    from butools.queues import MAPMAP1

    print("BuTools loaded successfully.")

except Exception as e:
    raise ImportError(
        "Could not import BuTools. "
        "Install BuTools or set BUTOOLS_PATH correctly.\n"
        "Linux example:\n"
        "  export BUTOOLS_PATH=/scratch200/davidfine/butools2/Python\n"
        f"Original error: {type(e).__name__}: {e}"
    )
# %%
# %%
# ============================================================
# CELL 2:
# User parameters
# ============================================================

# ============================================================
# CONFIGURATION VIA COMMAND LINE ARGUMENTS
# ============================================================

import argparse

def parse_k_list(s):
    """
    Convert string like '1,3,5,7,9,11,13,15' to list [1,3,5,7,9,11,13,15].
    """
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate PH/PH/1 dataset with arrival/service/sojourn moments and LST fits."
    )

    parser.add_argument(
        "--num-examples",
        type=int,
        default=10,
        help="Number of PH/PH/1 examples to generate."
    )

    parser.add_argument(
        "--arrival-size",
        type=int,
        default=20,
        help="Number of phases in the arrival PH distribution."
    )

    parser.add_argument(
        "--service-size",
        type=int,
        default=10,
        help="Number of phases in the service PH distribution."
    )

    parser.add_argument(
        "--rho-min",
        type=float,
        default=0.30,
        help="Minimum utilization rho."
    )

    parser.add_argument(
        "--rho-max",
        type=float,
        default=0.95,
        help="Maximum utilization rho."
    )

    parser.add_argument(
        "--num-moments",
        type=int,
        default=20,
        help="Number of raw moments to save for arrival, service, and sojourn time."
    )

    parser.add_argument(
        "--k-list",
        type=parse_k_list,
        default=[1, 3, 5, 7, 9, 11, 13, 15],
        help="Comma-separated list of moment counts used for fitting, e.g. '1,3,5,7,9'."
    )

    parser.add_argument(
        "--s-max",
        type=float,
        default=20.0,
        help="Maximum s value in the LST grid."
    )

    parser.add_argument(
        "--s-points",
        type=int,
        default=200,
        help="Number of points in the LST grid."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed."
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=r"C:\phph1_dataset_medium",
        help="Output directory."
    )

    parser.add_argument(
        "--resume",
        type=int,
        default=0,
        help="Use 1 to resume from existing files, 0 to start a clean run."
    )

    parser.add_argument(
        "--per-family-per-band",
        type=int,
        default=3,
        help="Number of PH distributions per family per SCV band in each pool."
    )

    return parser.parse_args()


args = parse_args()

SEED = args.seed

NUM_EXAMPLES = args.num_examples

ARRIVAL_PH_SIZE = args.arrival_size
SERVICE_PH_SIZE = args.service_size

RHO_RANGE = (args.rho_min, args.rho_max)

NUM_MOMENTS_SAVE = args.num_moments

K_LIST = args.k_list

S_GRID = np.linspace(0.0, args.s_max, args.s_points)

PH_FAMILIES = ["general", "coxian", "hyper_erlang"]

SCV_BANDS = [
    (0.15, 0.30),
    (0.30, 0.60),
    (0.60, 1.00),
    (1.00, 2.00),
    (2.00, 5.00),
]

PER_FAMILY_PER_BAND = args.per_family_per_band

OUTPUT_DIR = Path(args.output_dir)
EXAMPLES_DIR = OUTPUT_DIR / "examples_pkl"

POOLS_PKL = OUTPUT_DIR / "ph_pools.pkl"
PLAN_CSV = OUTPUT_DIR / "experiment_plan.csv"

SUMMARY_EXAMPLES_CSV = OUTPUT_DIR / "examples_summary.csv"
SUMMARY_FITS_CSV = OUTPUT_DIR / "fits_summary.csv"
FAILURES_CSV = OUTPUT_DIR / "failures.csv"

BOUND_TOL = 1e-8
MONOTONE_TOL = 1e-7
L0_TOL = 1e-6

RESUME_IF_FILES_EXIST = bool(args.resume)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(SEED)
random.seed(SEED)

print("=" * 70)
print("PH/PH/1 DATASET GENERATOR")
print("=" * 70)
print("NUM_EXAMPLES       =", NUM_EXAMPLES)
print("ARRIVAL_PH_SIZE    =", ARRIVAL_PH_SIZE)
print("SERVICE_PH_SIZE    =", SERVICE_PH_SIZE)
print("RHO_RANGE          =", RHO_RANGE)
print("NUM_MOMENTS_SAVE   =", NUM_MOMENTS_SAVE)
print("K_LIST             =", K_LIST)
print("S_GRID             =", f"0 to {args.s_max}, {args.s_points} points")
print("OUTPUT_DIR         =", OUTPUT_DIR)
print("RESUME             =", RESUME_IF_FILES_EXIST)
print("=" * 70)
# %%
# %%
# ============================================================
# CELL 3:
# Basic PH/ME utilities
# ============================================================

def as_row(x):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return x


def as_mat(A):
    return np.asarray(A, dtype=float)


def col_ones(n):
    return np.ones((n, 1), dtype=float)


def ph_moments(alpha, T, n_moms):
    """
    Moments of a PH/ME distribution:
        m_k = k! * alpha * (-T)^(-k) * 1
    """
    alpha = as_row(alpha)
    T = as_mat(T)

    n = T.shape[0]
    e = col_ones(n)

    inv_minus_T = np.linalg.inv(-T)
    power = np.eye(n)

    moms = []

    for k in range(1, n_moms + 1):
        power = power @ inv_minus_T
        mk = math.factorial(k) * float(alpha @ power @ e)
        moms.append(mk)

    return np.array(moms, dtype=float)


def lst_me_points(alpha, T, s_grid):
    """
    LST of PH/ME distribution:
        L(s) = alpha * (sI - T)^(-1) * (-T * 1)
    """
    alpha = as_row(alpha)
    T = as_mat(T)
    s_grid = np.asarray(s_grid, dtype=float)

    n = T.shape[0]
    I = np.eye(n)
    exit_vec = (-T) @ col_ones(n)

    vals = []

    for s in s_grid:
        M = s * I - T
        val = float(alpha @ np.linalg.solve(M, exit_vec))
        vals.append(val)

    return np.array(vals, dtype=float)


def l2_distance(f_true, f_hat, x_grid):
    """
    L2 distance on a grid:
        sqrt( integral (f_true - f_hat)^2 dx )
    """
    f_true = np.asarray(f_true, dtype=float)
    f_hat = np.asarray(f_hat, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)

    diff2 = (f_true - f_hat) ** 2

    return float(np.sqrt(np.trapz(diff2, x_grid)))


def sup_error(f_true, f_hat):
    f_true = np.asarray(f_true, dtype=float)
    f_hat = np.asarray(f_hat, dtype=float)

    return float(np.max(np.abs(f_true - f_hat)))


def normalized_moments(raw_moms):
    """
    Normalize moments by the mean:
        m_k_norm = m_k / m_1^k
    """
    raw_moms = np.asarray(raw_moms, dtype=float)
    m1 = raw_moms[0]

    if m1 <= 0 or not np.isfinite(m1):
        raise ValueError(f"Bad first moment: {m1}")

    norm = []

    for k in range(1, len(raw_moms) + 1):
        norm.append(raw_moms[k - 1] / (m1 ** k))

    return np.array(norm, dtype=float)


def max_percent_moment_error(alpha_fit, T_fit, target_moms):
    """
    Maximum percent error across the target moments.
    """
    target_moms = np.asarray(target_moms, dtype=float)

    fitted_moms = ph_moments(
        alpha_fit,
        T_fit,
        len(target_moms)
    )

    denom = np.maximum(np.abs(target_moms), 1e-14)
    pct_err = 100.0 * np.abs(fitted_moms - target_moms) / denom

    return float(np.max(pct_err))


def check_lst_validity(lst_vals):
    """
    Check fitted LST validity:
      1. finite
      2. in [0,1]
      3. starts at 1
      4. monotone non-increasing
    """
    lst_vals = np.asarray(lst_vals, dtype=float)

    is_finite = bool(np.all(np.isfinite(lst_vals)))

    if not is_finite:
        return {
            "is_finite": False,
            "is_in_0_1": False,
            "starts_at_1": False,
            "is_monotone": False,
            "fit_valid": False,
            "lst_min": np.nan,
            "lst_max": np.nan,
            "lst_at_0": np.nan,
            "max_upward_jump": np.nan,
            "num_bound_violations": np.nan,
            "num_monotone_violations": np.nan,
        }

    diffs = np.diff(lst_vals)

    lst_min = float(np.min(lst_vals))
    lst_max = float(np.max(lst_vals))
    lst_at_0 = float(lst_vals[0])
    max_upward_jump = float(np.max(diffs)) if len(diffs) > 0 else 0.0

    num_bound_violations = int(
        np.sum((lst_vals < -BOUND_TOL) | (lst_vals > 1.0 + BOUND_TOL))
    )

    num_monotone_violations = int(
        np.sum(diffs > MONOTONE_TOL)
    )

    is_in_0_1 = bool(num_bound_violations == 0)
    starts_at_1 = bool(abs(lst_at_0 - 1.0) <= L0_TOL)
    is_monotone = bool(num_monotone_violations == 0)

    fit_valid = bool(
        is_finite
        and is_in_0_1
        and starts_at_1
        and is_monotone
    )

    return {
        "is_finite": is_finite,
        "is_in_0_1": is_in_0_1,
        "starts_at_1": starts_at_1,
        "is_monotone": is_monotone,
        "fit_valid": fit_valid,
        "lst_min": lst_min,
        "lst_max": lst_max,
        "lst_at_0": lst_at_0,
        "max_upward_jump": max_upward_jump,
        "num_bound_violations": num_bound_violations,
        "num_monotone_violations": num_monotone_violations,
    }


def ph_mean(alpha, T):
    return ph_moments(alpha, T, 1)[0]


def ph_scv(alpha, T):
    m1, m2 = ph_moments(alpha, T, 2)
    var = m2 - m1 ** 2

    if m1 <= 0:
        return np.nan

    return float(var / (m1 ** 2))


def scale_ph_to_mean(alpha, T, target_mean):
    """
    If X ~ PH(alpha,T), multiplying T by c divides the mean by c.
    Therefore to obtain target_mean:
        c = current_mean / target_mean
    """
    alpha = as_row(alpha)
    T = as_mat(T)

    current_mean = ph_mean(alpha, T)

    if current_mean <= 0 or not np.isfinite(current_mean):
        raise ValueError(f"Bad current mean: {current_mean}")

    if target_mean <= 0 or not np.isfinite(target_mean):
        raise ValueError(f"Bad target mean: {target_mean}")

    rate_factor = current_mean / target_mean

    return alpha, T * rate_factor


def embed_ph(alpha_small, T_small, n_total):
    """
    Embed a smaller PH representation into a larger n_total-dimensional PH.
    The additional phases are unreachable, but keep the matrix dimension fixed.
    """
    alpha_small = as_row(alpha_small)
    T_small = as_mat(T_small)

    n_small = T_small.shape[0]

    if n_small > n_total:
        raise ValueError("Small PH dimension is larger than n_total.")

    if n_small == n_total:
        return alpha_small, T_small

    alpha_big = np.zeros((1, n_total))
    alpha_big[0, :n_small] = alpha_small

    T_big = np.zeros((n_total, n_total))
    T_big[:n_small, :n_small] = T_small

    # unreachable dummy phases, each absorbing with rate 1
    for i in range(n_small, n_total):
        T_big[i, i] = -1.0

    return alpha_big, T_big


def get_ph_from_item(item):
    """
    Extract alpha and T from a pool item.
    """
    alpha_keys = ["alpha", "a", "beta", "init", "initial"]
    matrix_keys = ["T", "A", "S", "subgenerator", "generator"]

    alpha = None
    T = None

    for key in alpha_keys:
        if key in item:
            alpha = item[key]
            break

    for key in matrix_keys:
        if key in item:
            T = item[key]
            break

    if alpha is None or T is None:
        raise KeyError(
            "Could not find alpha/T in item. "
            f"Available keys: {list(item.keys())}"
        )

    return as_row(alpha), as_mat(T)
# %%
# %%
# ============================================================
# CELL 4:
# PH sampling functions
# ============================================================

def erlang_ph(order, mean=1.0):
    """
    Erlang(order, rate=order/mean)
    """
    order = int(order)
    rate = order / mean

    alpha = np.zeros((1, order))
    alpha[0, 0] = 1.0

    T = np.zeros((order, order))

    for i in range(order):
        T[i, i] = -rate
        if i < order - 1:
            T[i, i + 1] = rate

    return alpha, T


def hypo2_ph(target_scv):
    """
    Two-phase hypoexponential with mean 1 and target SCV in [0.5,1).
    X = Exp(r1) + Exp(r2)
    """
    c = float(target_scv)
    c = min(max(c, 0.500001), 0.999999)

    d = math.sqrt(2.0 * c - 1.0)

    x = 0.5 * (1.0 + d)
    y = 1.0 - x

    r1 = 1.0 / x
    r2 = 1.0 / y

    alpha = np.array([[1.0, 0.0]])
    T = np.array([
        [-r1, r1],
        [0.0, -r2],
    ])

    return alpha, T


def hyperexp2_ph(target_scv):
    """
    Two-phase hyperexponential with mean 1 and target SCV > 1.
    Balanced means construction.
    """
    c = float(target_scv)
    c = max(c, 1.000001)

    p = 0.5 * (1.0 + math.sqrt((c - 1.0) / (c + 1.0)))

    mu1 = 2.0 * p
    mu2 = 2.0 * (1.0 - p)

    alpha = np.array([[p, 1.0 - p]])
    T = np.array([
        [-mu1, 0.0],
        [0.0, -mu2],
    ])

    return alpha, T


def target_scv_ph(n, target_scv):
    """
    Construct a simple PH with approximate target SCV and mean 1,
    then embed it into dimension n.
    """
    c = float(target_scv)

    if c < 0.5:
        order = int(round(1.0 / c))
        order = max(2, min(order, n))
        alpha, T = erlang_ph(order, mean=1.0)

    elif c < 1.0:
        alpha, T = hypo2_ph(c)

    else:
        alpha, T = hyperexp2_ph(c)

    return embed_ph(alpha, T, n)


def random_general_ph(n, rng):
    """
    Random general PH of size n.
    """
    alpha = rng.dirichlet(np.ones(n)).reshape(1, n)

    rates = np.exp(rng.uniform(np.log(0.1), np.log(10.0), size=n))

    T = np.zeros((n, n))

    for i in range(n):
        # probabilities over n transient states plus absorption
        probs = rng.dirichlet(np.ones(n + 1))

        # force diagonal probability to zero by redistributing
        probs_trans = probs[:n].copy()
        probs_abs = probs[-1]

        probs_abs += probs_trans[i]
        probs_trans[i] = 0.0

        total = probs_abs + probs_trans.sum()
        probs_abs /= total
        probs_trans /= total

        T[i, i] = -rates[i]

        for j in range(n):
            if j != i:
                T[i, j] = rates[i] * probs_trans[j]

        # absorption rate is rates[i] * probs_abs implicitly

    alpha, T = scale_ph_to_mean(alpha, T, target_mean=1.0)

    return alpha, T


def random_coxian_ph(n, rng):
    """
    Random Coxian PH of size n.
    """
    alpha = np.zeros((1, n))
    alpha[0, 0] = 1.0

    rates = np.exp(rng.uniform(np.log(0.1), np.log(10.0), size=n))
    probs = rng.beta(1.2, 1.2, size=n - 1)

    T = np.zeros((n, n))

    for i in range(n):
        T[i, i] = -rates[i]

        if i < n - 1:
            T[i, i + 1] = probs[i] * rates[i]

    alpha, T = scale_ph_to_mean(alpha, T, target_mean=1.0)

    return alpha, T


def random_hyper_erlang_ph(n, rng):
    """
    Random Hyper-Erlang PH of total size n.
    """
    max_branches = max(1, min(10, n))
    k = int(rng.integers(1, max_branches + 1))

    # random composition of n into k positive block sizes
    if k == 1:
        sizes = [n]
    else:
        cuts = sorted(rng.choice(np.arange(1, n), size=k - 1, replace=False))
        parts = [cuts[0]] + [cuts[i] - cuts[i - 1] for i in range(1, k - 1)] + [n - cuts[-1]]
        sizes = parts

    weights = rng.dirichlet(np.ones(k))
    rates = np.exp(rng.uniform(np.log(0.1), np.log(10.0), size=k))

    alpha = np.zeros((1, n))
    T = np.zeros((n, n))

    idx = 0

    for branch_idx, d in enumerate(sizes):
        alpha[0, idx] = weights[branch_idx]
        rate = rates[branch_idx]

        for j in range(d):
            pos = idx + j
            T[pos, pos] = -rate

            if j < d - 1:
                T[pos, pos + 1] = rate

        idx += d

    alpha, T = scale_ph_to_mean(alpha, T, target_mean=1.0)

    return alpha, T


def random_ph_by_family(family, n, rng):
    if family == "general":
        return random_general_ph(n, rng)

    if family == "coxian":
        return random_coxian_ph(n, rng)

    if family == "hyper_erlang":
        return random_hyper_erlang_ph(n, rng)

    raise ValueError(f"Unknown family: {family}")


def build_diverse_ph_pool(
    ph_size,
    per_band=3,
    families=("general", "coxian", "hyper_erlang"),
    scv_bands=None,
    seed=123,
    max_attempts_per_family_band=3000,
):
    """
    Build a pool of PH distributions across families and SCV bands.

    If random rejection fails to fill a band, the function uses a simple
    targeted PH fallback with a target SCV inside the band.
    """
    if scv_bands is None:
        scv_bands = SCV_BANDS

    rng = np.random.default_rng(seed)

    pool = []
    item_id = 0

    for family in families:
        for band_idx, (lo, hi) in enumerate(scv_bands):

            accepted = []
            attempts = 0

            while len(accepted) < per_band and attempts < max_attempts_per_family_band:
                attempts += 1

                try:
                    alpha, T = random_ph_by_family(family, ph_size, rng)
                    scv = ph_scv(alpha, T)

                    if np.isfinite(scv) and lo <= scv <= hi:
                        accepted.append((alpha, T, scv))

                except Exception:
                    continue

            # Fallback if not enough were found
            while len(accepted) < per_band:
                target = float(rng.uniform(lo, hi))

                try:
                    alpha, T = target_scv_ph(ph_size, target)
                    scv = ph_scv(alpha, T)
                    accepted.append((alpha, T, scv))

                    warnings.warn(
                        f"Fallback PH used for family={family}, band=({lo},{hi}), "
                        f"target_scv={target:.3f}, achieved_scv={scv:.3f}"
                    )

                except Exception as e:
                    raise RuntimeError(
                        f"Could not build fallback PH for family={family}, band=({lo},{hi}). "
                        f"Error: {e}"
                    )

            for alpha, T, scv in accepted:
                m1 = ph_mean(alpha, T)

                item = {
                    "id": item_id,
                    "family": family,
                    "band_idx": band_idx,
                    "band_lo": lo,
                    "band_hi": hi,
                    "alpha": alpha,
                    "T": T,
                    "mean": m1,
                    "scv": scv,
                    "ph_size": ph_size,
                }

                pool.append(item)
                item_id += 1

            print(
                f"family={family:13s}, band=({lo:.2f},{hi:.2f}), "
                f"accepted={len(accepted)}, attempts={attempts}"
            )

    return pool
#%%
# %%
# ============================================================
# CELL 5:
# PH/PH/1 queue construction using MAPMAP1
# ============================================================

def ph_to_renewal_map(alpha, T):
    """
    Convert a PH inter-event-time distribution to a renewal MAP.

    D0 = T
    D1 = exit_vector * alpha
    """
    alpha = as_row(alpha)
    T = as_mat(T)

    exit_vec = (-T) @ col_ones(T.shape[0])
    D0 = T.copy()
    D1 = exit_vec @ alpha

    return D0, D1


def parse_mapmap1_st_distr_me_result(res):
    """
    Robustly parse MAPMAP1(..., 'stDistrME') output.
    Depending on BuTools version, it can be tuple/list.
    """
    if isinstance(res, tuple) and len(res) == 2:
        return as_row(res[0]), as_mat(res[1])

    if isinstance(res, list):
        if len(res) == 2:
            return as_row(res[0]), as_mat(res[1])

        if len(res) == 1:
            inner = res[0]
            if isinstance(inner, tuple) and len(inner) == 2:
                return as_row(inner[0]), as_mat(inner[1])

            if isinstance(inner, list) and len(inner) == 2:
                return as_row(inner[0]), as_mat(inner[1])

    raise ValueError(
        "Could not parse MAPMAP1 stDistrME result. "
        f"Type={type(res)}, value={res}"
    )


def phph1_sojourn_me_from_arrival_service_items(
    arrival_item,
    service_item,
    rho_target,
):
    """
    Build PH/PH/1 queue from arrival and service PH items.

    We scale:
        E[A] = 1
        E[S] = rho_target

    Then traffic intensity is:
        rho = E[S] / E[A] = rho_target
    """
    alpha_A_raw, T_A_raw = get_ph_from_item(arrival_item)
    alpha_S_raw, T_S_raw = get_ph_from_item(service_item)

    # Scale arrival mean to 1
    alpha_A, T_A = scale_ph_to_mean(
        alpha_A_raw,
        T_A_raw,
        target_mean=1.0
    )

    # Scale service mean to rho_target
    alpha_S, T_S = scale_ph_to_mean(
        alpha_S_raw,
        T_S_raw,
        target_mean=rho_target
    )

    arrival_mean = ph_mean(alpha_A, T_A)
    service_mean = ph_mean(alpha_S, T_S)
    rho_actual = service_mean / arrival_mean

    if rho_actual >= 1.0:
        raise ValueError(f"Unstable queue, rho={rho_actual}")

    D0, D1 = ph_to_renewal_map(alpha_A, T_A)
    S0, S1 = ph_to_renewal_map(alpha_S, T_S)

    # Sojourn-time distribution as ME
    res = MAPMAP1(D0, D1, S0, S1, "stDistrME")
    beta_W, T_W = parse_mapmap1_st_distr_me_result(res)

    return {
        "beta": beta_W,
        "T": T_W,

        "arrival_alpha": alpha_A,
        "arrival_T": T_A,
        "service_alpha": alpha_S,
        "service_T": T_S,

        "arrival_mean": arrival_mean,
        "service_mean": service_mean,
        "rho": rho_actual,

        "arrival_family": arrival_item.get("family", "unknown"),
        "service_family": service_item.get("family", "unknown"),

        "arrival_scv": ph_scv(alpha_A, T_A),
        "service_scv": ph_scv(alpha_S, T_S),

        "arrival_pool_id": arrival_item.get("id", -1),
        "service_pool_id": service_item.get("id", -1),
    }
#%%
# %%
# ============================================================
# CELL 6:
# Build or load PH pools and experiment plan
# ============================================================

# ------------------------------------------------------------
# Load or build pools
# ------------------------------------------------------------

if POOLS_PKL.exists() and RESUME_IF_FILES_EXIST:
    print("Loading existing PH pools from:", POOLS_PKL)

    with open(POOLS_PKL, "rb") as f:
        pools_obj = pickle.load(f)

    arrival_pool = pools_obj["arrival_pool"]
    service_pool = pools_obj["service_pool"]

else:
    print("Building arrival pool...")
    arrival_pool = build_diverse_ph_pool(
        ph_size=ARRIVAL_PH_SIZE,
        per_band=PER_FAMILY_PER_BAND,
        families=PH_FAMILIES,
        scv_bands=SCV_BANDS,
        seed=SEED + 1,
    )

    print("Building service pool...")
    service_pool = build_diverse_ph_pool(
        ph_size=SERVICE_PH_SIZE,
        per_band=PER_FAMILY_PER_BAND,
        families=PH_FAMILIES,
        scv_bands=SCV_BANDS,
        seed=SEED + 2,
    )

    with open(POOLS_PKL, "wb") as f:
        pickle.dump(
            {
                "arrival_pool": arrival_pool,
                "service_pool": service_pool,
            },
            f,
        )

    print("Saved PH pools to:", POOLS_PKL)

print("Arrival pool size:", len(arrival_pool))
print("Service pool size:", len(service_pool))

# ------------------------------------------------------------
# Build dictionaries
# ------------------------------------------------------------

arrival_pool_by_id = {item["id"]: item for item in arrival_pool}
service_pool_by_id = {item["id"]: item for item in service_pool}

# ------------------------------------------------------------
# Load or build experiment plan
# ------------------------------------------------------------

if PLAN_CSV.exists() and RESUME_IF_FILES_EXIST:
    print("Loading existing experiment plan from:", PLAN_CSV)
    plan_df = pd.read_csv(PLAN_CSV)

else:
    rng_plan = np.random.default_rng(SEED + 100)

    plan_rows = []

    arrival_ids = [item["id"] for item in arrival_pool]
    service_ids = [item["id"] for item in service_pool]

    for example_id in range(NUM_EXAMPLES):
        arrival_pool_id = int(rng_plan.choice(arrival_ids))
        service_pool_id = int(rng_plan.choice(service_ids))
        rho_target = float(rng_plan.uniform(RHO_RANGE[0], RHO_RANGE[1]))

        plan_rows.append({
            "example_id": example_id,
            "arrival_pool_id": arrival_pool_id,
            "service_pool_id": service_pool_id,
            "rho_target": rho_target,
        })

    plan_df = pd.DataFrame(plan_rows)
    plan_df.to_csv(PLAN_CSV, index=False)

    print("Saved experiment plan to:", PLAN_CSV)

display(plan_df.head())
display(plan_df.tail())
#%%
# %%
# ============================================================
# CELL 7:
# Main dataset generation
# ============================================================

# ------------------------------------------------------------
# Load existing summaries if resuming
# ------------------------------------------------------------


