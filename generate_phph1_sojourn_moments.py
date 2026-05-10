# %%
"""
Generate one PH/PH/1 example and save log moments for learning.

By default, each saved pickle is a dictionary containing PH parameters,
family labels, SCV/mean metadata, and log moments. Use --payload-format tuple
to save only:
    (
        log_input_moments,    # shape (20,): first 10 inter-arrival, first 10 service
        log_sojourn_moments,  # shape (20,): first 20 sojourn-time moments
    )

All moments are raw moments and are stored as natural logarithms.
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

import numpy as np

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass


def add_butools_paths():
    """
    Add BuTools search paths without hard-coding one user's machine.

    On Linux clusters, set for example:
        export BUTOOLS_PATH=/scratch200/davidfine/butools2/Python
    """
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

try:
    import butools
    from butools.queues import MAPMAP1

    butools.verbose = False
    butools.checkInput = False
except Exception as e:
    raise ImportError(
        "Could not import BuTools. Install BuTools or set BUTOOLS_PATH correctly.\n"
        "Linux example:\n"
        "  export BUTOOLS_PATH=/scratch200/davidfine/butools2/Python\n"
        f"Original error: {type(e).__name__}: {e}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate one PH/PH/1 sojourn ME and save log input/sojourn moments."
    )
    parser.add_argument(
        "--arrival-size",
        type=int,
        default=20,
        help="Fixed arrival PH size when --random-complement-sizes is 0.",
    )
    parser.add_argument(
        "--service-size",
        type=int,
        default=10,
        help="Fixed service PH size when --random-complement-sizes is 0.",
    )
    parser.add_argument(
        "--total-ph-size",
        type=int,
        default=100,
        help="Total arrival+service PH size when --random-complement-sizes is 1.",
    )
    parser.add_argument(
        "--min-ph-size",
        type=int,
        default=2,
        help="Minimum arrival/service PH size when --random-complement-sizes is 1.",
    )
    parser.add_argument(
        "--random-complement-sizes",
        type=int,
        default=1,
        help="Use 1 to sample arrival size n and service size total_ph_size-n.",
    )
    parser.add_argument(
        "--arrival-scv",
        type=float,
        default=0.8,
        help="Target SCV for the inter-arrival PH distribution.",
    )
    parser.add_argument(
        "--service-mean",
        type=float,
        default=0.7,
        help="Mean service time. Arrival mean is fixed to 1, so this is also rho.",
    )
    parser.add_argument(
        "--service-scv",
        type=float,
        default=1.5,
        help="Target SCV for the service PH distribution.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=r"C:\phph1",
        help="Directory where the pickle file is saved.",
    )
    parser.add_argument(
        "--clean-output",
        type=int,
        default=0,
        help="Use 1 to delete old PKL/CSV/PNG files from --output-dir before generating.",
    )
    parser.add_argument(
        "--resume",
        type=int,
        default=1,
        help="Use 1 to skip example IDs whose PKL already exists in --output-dir.",
    )
    parser.add_argument(
        "--filename-prefix",
        type=str,
        default="phph1",
        help="Filename prefix before the moment descriptors.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=1000,
        help="Number of examples to generate. Prints elapsed time for each example.",
    )
    parser.add_argument(
        "--display-pkl",
        type=int,
        default=0,
        help="Use 1 to print each saved pickle payload after writing it.",
    )
    parser.add_argument(
        "--payload-format",
        choices=("dict", "tuple"),
        default="dict",
        help="Use dict to save PH metadata and moments, or tuple for only the two log-moment arrays.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for sampling PH families.",
    )
    parser.add_argument(
        "--families",
        type=str,
        default="general,coxian,hyper_erlang",
        help="Comma-separated PH families: general, coxian, hyper_erlang.",
    )
    parser.add_argument(
        "--arrival-scv-min",
        type=float,
        default=0.15,
        help="Minimum accepted inter-arrival SCV.",
    )
    parser.add_argument(
        "--arrival-scv-max",
        type=float,
        default=20.0,
        help="Maximum accepted inter-arrival SCV.",
    )
    parser.add_argument(
        "--service-scv-min",
        type=float,
        default=0.15,
        help="Minimum accepted service SCV.",
    )
    parser.add_argument(
        "--service-scv-max",
        type=float,
        default=20.0,
        help="Maximum accepted service SCV.",
    )
    parser.add_argument(
        "--max-sampling-attempts",
        type=int,
        default=20000,
        help="Maximum attempts to sample a PH inside the requested SCV range.",
    )
    parser.add_argument(
        "--scv-bands",
        type=int,
        default=20,
        help="Number of SCV bands used to cover the requested SCV interval.",
    )
    return parser.parse_args()


def as_row(x):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return x


def as_mat(a):
    return np.asarray(a, dtype=float)


def col_ones(n):
    return np.ones((n, 1), dtype=float)


def ph_moments(alpha, t_matrix, n_moms):
    """
    Raw moments of a PH/ME distribution:
        m_k = k! * alpha * (-T)^(-k) * 1
    """
    alpha = as_row(alpha)
    t_matrix = as_mat(t_matrix)
    inv_minus_t = np.linalg.inv(-t_matrix)
    power = np.eye(t_matrix.shape[0])
    ones = col_ones(t_matrix.shape[0])

    moments = []
    for k in range(1, n_moms + 1):
        power = power @ inv_minus_t
        moment = math.factorial(k) * (alpha @ power @ ones).item()
        moments.append(moment)

    return np.asarray(moments, dtype=float)


def ph_mean(alpha, t_matrix):
    return ph_moments(alpha, t_matrix, 1)[0]


def ph_scv(alpha, t_matrix):
    m1, m2 = ph_moments(alpha, t_matrix, 2)
    return float((m2 - m1**2) / m1**2)


def scale_ph_to_mean(alpha, t_matrix, target_mean):
    current_mean = ph_mean(alpha, t_matrix)
    if current_mean <= 0 or not np.isfinite(current_mean):
        raise ValueError(f"Bad current PH mean: {current_mean}")
    if target_mean <= 0 or not np.isfinite(target_mean):
        raise ValueError(f"Bad target mean: {target_mean}")

    return as_row(alpha), as_mat(t_matrix) * (current_mean / target_mean)


def embed_ph(alpha_small, t_small, n_total):
    alpha_small = as_row(alpha_small)
    t_small = as_mat(t_small)
    n_small = t_small.shape[0]

    if n_small > n_total:
        raise ValueError("Small PH dimension is larger than requested PH size.")

    if n_small == n_total:
        return alpha_small, t_small

    alpha = np.zeros((1, n_total))
    alpha[0, :n_small] = alpha_small

    t_matrix = np.zeros((n_total, n_total))
    t_matrix[:n_small, :n_small] = t_small

    for i in range(n_small, n_total):
        t_matrix[i, i] = -1.0

    return alpha, t_matrix


def erlang_ph(order, mean=1.0):
    rate = order / mean
    alpha = np.zeros((1, order))
    alpha[0, 0] = 1.0

    t_matrix = np.zeros((order, order))
    for i in range(order):
        t_matrix[i, i] = -rate
        if i < order - 1:
            t_matrix[i, i + 1] = rate

    return alpha, t_matrix


def hypo2_ph(target_scv):
    c = min(max(float(target_scv), 0.500001), 0.999999)
    d = math.sqrt(2.0 * c - 1.0)
    x = 0.5 * (1.0 + d)
    y = 1.0 - x

    return (
        np.array([[1.0, 0.0]]),
        np.array(
            [
                [-1.0 / x, 1.0 / x],
                [0.0, -1.0 / y],
            ]
        ),
    )


def hyperexp2_ph(target_scv):
    c = max(float(target_scv), 1.000001)
    p = 0.5 * (1.0 + math.sqrt((c - 1.0) / (c + 1.0)))
    mu1 = 2.0 * p
    mu2 = 2.0 * (1.0 - p)

    return (
        np.array([[p, 1.0 - p]]),
        np.array(
            [
                [-mu1, 0.0],
                [0.0, -mu2],
            ]
        ),
    )


def random_general_ph(n_phases, rng):
    alpha = rng.dirichlet(np.ones(n_phases)).reshape(1, n_phases)
    rates = np.exp(rng.uniform(np.log(0.01), np.log(100.0), size=n_phases))

    t_matrix = np.zeros((n_phases, n_phases))
    for i in range(n_phases):
        probs = rng.dirichlet(np.ones(n_phases + 1))
        probs_trans = probs[:n_phases].copy()
        probs_abs = probs[-1]

        probs_abs += probs_trans[i]
        probs_trans[i] = 0.0

        total = probs_abs + probs_trans.sum()
        probs_abs /= total
        probs_trans /= total

        t_matrix[i, i] = -rates[i]
        for j in range(n_phases):
            if j != i:
                t_matrix[i, j] = rates[i] * probs_trans[j]

    return scale_ph_to_mean(alpha, t_matrix, target_mean=1.0)


def random_coxian_ph(n_phases, rng):
    alpha = np.zeros((1, n_phases))
    alpha[0, 0] = 1.0

    rates = np.exp(rng.uniform(np.log(0.01), np.log(100.0), size=n_phases))
    probs = rng.beta(1.2, 1.2, size=n_phases - 1)

    t_matrix = np.zeros((n_phases, n_phases))
    for i in range(n_phases):
        t_matrix[i, i] = -rates[i]
        if i < n_phases - 1:
            t_matrix[i, i + 1] = probs[i] * rates[i]

    return scale_ph_to_mean(alpha, t_matrix, target_mean=1.0)


def random_hyper_erlang_ph(n_phases, rng):
    max_branches = max(1, min(10, n_phases))
    n_branches = int(rng.integers(1, max_branches + 1))

    if n_branches == 1:
        sizes = [n_phases]
    else:
        cuts = sorted(
            rng.choice(np.arange(1, n_phases), size=n_branches - 1, replace=False)
        )
        sizes = (
            [cuts[0]]
            + [cuts[i] - cuts[i - 1] for i in range(1, n_branches - 1)]
            + [n_phases - cuts[-1]]
        )

    weights = rng.dirichlet(np.ones(n_branches))
    rates = np.exp(rng.uniform(np.log(0.01), np.log(100.0), size=n_branches))

    alpha = np.zeros((1, n_phases))
    t_matrix = np.zeros((n_phases, n_phases))

    idx = 0
    for branch_idx, size in enumerate(sizes):
        alpha[0, idx] = weights[branch_idx]
        rate = rates[branch_idx]

        for j in range(size):
            pos = idx + j
            t_matrix[pos, pos] = -rate
            if j < size - 1:
                t_matrix[pos, pos + 1] = rate

        idx += size

    return scale_ph_to_mean(alpha, t_matrix, target_mean=1.0)


def random_ph_by_family(family, n_phases, rng):
    if family == "hyper_general":
        family = "general"
    if family == "general":
        return random_general_ph(n_phases, rng)
    if family == "coxian":
        return random_coxian_ph(n_phases, rng)
    if family == "hyper_erlang":
        return random_hyper_erlang_ph(n_phases, rng)
    raise ValueError(f"Unknown PH family: {family}")


def parse_families(text):
    families = [item.strip() for item in text.split(",") if item.strip()]
    if not families:
        raise ValueError("--families must contain at least one PH family.")

    allowed = {"general", "coxian", "hyper_erlang", "hyper_general"}
    unknown = sorted(set(families) - allowed)
    if unknown:
        raise ValueError(f"Unknown PH families: {unknown}. Allowed: {sorted(allowed)}")

    return families


def sample_ph_in_scv_range(
    family,
    n_phases,
    target_mean,
    scv_min,
    scv_max,
    rng,
    max_attempts,
):
    for _ in range(max_attempts):
        alpha, t_matrix = random_ph_by_family(family, n_phases, rng)
        alpha, t_matrix = scale_ph_to_mean(alpha, t_matrix, target_mean)
        scv = ph_scv(alpha, t_matrix)
        if scv_min <= scv <= scv_max:
            return alpha, t_matrix, scv, False

    target_scv = 0.5 * (scv_min + scv_max)
    alpha, t_matrix = target_scv_ph(n_phases, target_scv, target_mean)
    scv = ph_scv(alpha, t_matrix)
    print(
        "Fallback targeted PH used:",
        f"family={family}",
        f"requested_band=[{scv_min:.6g}, {scv_max:.6g}]",
        f"achieved_scv={scv:.6g}",
    )
    return alpha, t_matrix, scv, True


def scv_band_for_example(example_index, scv_min, scv_max, scv_bands):
    if scv_bands <= 1:
        return scv_min, scv_max

    band_width = (scv_max - scv_min) / scv_bands
    band_index = example_index % scv_bands
    band_min = scv_min + band_index * band_width
    band_max = scv_max if band_index == scv_bands - 1 else band_min + band_width
    return band_min, band_max


def target_scv_ph(n_phases, target_scv, target_mean):
    """
    Build a simple PH with the requested mean and approximate/requested SCV.

    For SCV < 0.5 this uses an Erlang fallback, so the achieved SCV may be
    the closest 1/order value supported by the requested PH size.
    """
    if target_scv < 0.5:
        order = int(round(1.0 / target_scv))
        order = max(2, min(order, n_phases))
        alpha, t_matrix = erlang_ph(order, mean=1.0)
    elif target_scv < 1.0:
        alpha, t_matrix = hypo2_ph(target_scv)
    else:
        alpha, t_matrix = hyperexp2_ph(target_scv)

    alpha, t_matrix = embed_ph(alpha, t_matrix, n_phases)
    return scale_ph_to_mean(alpha, t_matrix, target_mean)


def ph_to_renewal_map(alpha, t_matrix):
    alpha = as_row(alpha)
    t_matrix = as_mat(t_matrix)
    exit_vec = (-t_matrix) @ col_ones(t_matrix.shape[0])

    d0 = t_matrix.copy()
    d1 = exit_vec @ alpha
    return d0, d1


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

    raise ValueError(
        "Could not parse MAPMAP1 stDistrME result. "
        f"Type={type(res)}, value={res}"
    )


def phph1_sojourn_me(arrival_alpha, arrival_t, service_alpha, service_t):
    arrival_mean = ph_mean(arrival_alpha, arrival_t)
    service_mean = ph_mean(service_alpha, service_t)
    rho = service_mean / arrival_mean

    if rho >= 1.0:
        raise ValueError(
            f"Unstable PH/PH/1 queue: rho={rho:.6g}. "
            "Use --service-mean smaller than the arrival mean 1.0."
        )

    d0, d1 = ph_to_renewal_map(arrival_alpha, arrival_t)
    s0, s1 = ph_to_renewal_map(service_alpha, service_t)

    result = MAPMAP1(d0, d1, s0, s1, "stDistrME")
    return parse_mapmap1_st_distr_me_result(result)


def safe_float_for_name(x):
    text = f"{x:.6g}"
    text = text.replace("-", "m").replace(".", "p")
    return re.sub(r"[^A-Za-z0-9_p]+", "_", text)


def build_output_path(
    output_dir,
    prefix,
    arrival_size,
    service_size,
    arrival_scv,
    service_mean,
    service_scv,
    example_id=None,
):
    example_part = "" if example_id is None else f"_ex_{example_id:06d}"
    filename = (
        f"{prefix}"
        f"_arr_n_{arrival_size}"
        f"_svc_n_{service_size}"
        f"_ia_scv_{safe_float_for_name(arrival_scv)}"
        f"_svc_mean_{safe_float_for_name(service_mean)}"
        f"_svc_scv_{safe_float_for_name(service_scv)}"
        f"{example_part}"
        ".pkl"
    )
    return Path(output_dir) / filename


def log_checked(moments, label):
    moments = np.asarray(moments, dtype=float)
    if not np.all(np.isfinite(moments)) or np.any(moments <= 0):
        raise ValueError(f"{label} moments must be positive and finite: {moments}")
    return np.log(moments)


def print_vector(label, values):
    print(label)
    for i, value in enumerate(np.asarray(values, dtype=float), start=1):
        print(f"{i:2d}: {value:.12E}")


def display_pickle_payload(output_path):
    with open(output_path, "rb") as f:
        payload = pickle.load(f)

    if isinstance(payload, dict):
        print("PKL:", output_path)
        print("arrival_family:", payload["arrival_family"])
        print("service_family:", payload["service_family"])
        print("arrival_size:", payload.get("arrival_size", "unknown"))
        print("service_size:", payload.get("service_size", "unknown"))
        print("arrival_scv:", payload["arrival_scv"])
        print("service_scv:", payload["service_scv"])
        print("rho:", payload["rho"])
        log_input_moments = payload["log_input_moments"]
        log_sojourn_moments = payload["log_sojourn_moments"]
    else:
        log_input_moments, log_sojourn_moments = payload
        print("PKL:", output_path)

    print_vector("log_input_moments:", log_input_moments)
    print_vector("exp(log_input_moments):", np.exp(log_input_moments))
    print_vector("log_sojourn_moments:", log_sojourn_moments)
    print_vector("exp(log_sojourn_moments):", np.exp(log_sojourn_moments))


def choose_ph_sizes(args, rng):
    if not args.random_complement_sizes:
        return args.arrival_size, args.service_size

    min_size = int(args.min_ph_size)
    total_size = int(args.total_ph_size)
    if min_size < 1:
        raise ValueError("--min-ph-size must be at least 1.")
    if total_size < 2 * min_size:
        raise ValueError("--total-ph-size must be at least 2 * --min-ph-size.")

    arrival_size = int(rng.integers(min_size, total_size - min_size + 1))
    service_size = total_size - arrival_size
    return arrival_size, service_size


def generate_example(args, output_dir, rng, families, example_id=None):
    family_index = 0 if example_id is None else example_id
    arrival_family = families[family_index % len(families)]
    service_family = families[(family_index + 1) % len(families)]
    arrival_size, service_size = choose_ph_sizes(args, rng)
    arrival_scv_min, arrival_scv_max = scv_band_for_example(
        family_index,
        args.arrival_scv_min,
        args.arrival_scv_max,
        args.scv_bands,
    )
    service_scv_min, service_scv_max = scv_band_for_example(
        family_index + args.scv_bands // 2,
        args.service_scv_min,
        args.service_scv_max,
        args.scv_bands,
    )

    arrival_alpha, arrival_t, arrival_scv, arrival_used_fallback = sample_ph_in_scv_range(
        arrival_family,
        arrival_size,
        target_mean=1.0,
        scv_min=arrival_scv_min,
        scv_max=arrival_scv_max,
        rng=rng,
        max_attempts=args.max_sampling_attempts,
    )
    service_alpha, service_t, service_scv, service_used_fallback = sample_ph_in_scv_range(
        service_family,
        service_size,
        target_mean=args.service_mean,
        scv_min=service_scv_min,
        scv_max=service_scv_max,
        rng=rng,
        max_attempts=args.max_sampling_attempts,
    )

    sojourn_beta, sojourn_t = phph1_sojourn_me(
        arrival_alpha,
        arrival_t,
        service_alpha,
        service_t,
    )

    arrival_moments = ph_moments(arrival_alpha, arrival_t, 10)
    service_moments = ph_moments(service_alpha, service_t, 10)
    sojourn_moments = ph_moments(sojourn_beta, sojourn_t, 20)

    log_input_moments = np.concatenate(
        [
            log_checked(arrival_moments, "Inter-arrival"),
            log_checked(service_moments, "Service"),
        ]
    )
    log_sojourn_moments = log_checked(sojourn_moments, "Sojourn")

    service_mean = ph_mean(service_alpha, service_t)
    output_path = build_output_path(
        output_dir,
        args.filename_prefix,
        arrival_size,
        service_size,
        arrival_scv,
        service_mean,
        service_scv,
        example_id=example_id,
    )

    if args.payload_format == "tuple":
        payload = (
            log_input_moments.astype(float),
            log_sojourn_moments.astype(float),
        )
    else:
        payload = {
            "example_id": example_id,
            "arrival_family": arrival_family,
            "service_family": service_family,
            "arrival_size": arrival_size,
            "service_size": service_size,
            "total_ph_size": arrival_size + service_size,
            "arrival_used_fallback": arrival_used_fallback,
            "service_used_fallback": service_used_fallback,
            "arrival_alpha": arrival_alpha.astype(float),
            "arrival_T": arrival_t.astype(float),
            "service_alpha": service_alpha.astype(float),
            "service_T": service_t.astype(float),
            "sojourn_beta": sojourn_beta.astype(float),
            "sojourn_T": sojourn_t.astype(float),
            "arrival_mean": ph_mean(arrival_alpha, arrival_t),
            "arrival_scv": arrival_scv,
            "arrival_scv_band_min": arrival_scv_min,
            "arrival_scv_band_max": arrival_scv_max,
            "service_mean": service_mean,
            "service_scv": service_scv,
            "service_scv_band_min": service_scv_min,
            "service_scv_band_max": service_scv_max,
            "rho": service_mean / ph_mean(arrival_alpha, arrival_t),
            "log_input_moments": log_input_moments.astype(float),
            "log_sojourn_moments": log_sojourn_moments.astype(float),
        }
    with open(output_path, "wb") as f:
        pickle.dump(payload, f)

    print("Saved:", output_path)
    print("Arrival family:", arrival_family)
    print("Service family:", service_family)
    print("Arrival PH size:", arrival_size)
    print("Service PH size:", service_size)
    print("Moment shapes:", log_input_moments.shape, log_sojourn_moments.shape)
    print("Inter-arrival mean:", ph_mean(arrival_alpha, arrival_t))
    print("Inter-arrival SCV:", arrival_scv)
    print("Service mean:", service_mean)
    print("Service SCV:", service_scv)
    print("Sojourn ME order:", sojourn_t.shape[0])
    if args.display_pkl:
        display_pickle_payload(output_path)
    return {
        "example_id": example_id,
        "path": str(output_path),
        "arrival_family": arrival_family,
        "service_family": service_family,
        "arrival_size": arrival_size,
        "service_size": service_size,
        "total_ph_size": arrival_size + service_size,
        "arrival_used_fallback": arrival_used_fallback,
        "service_used_fallback": service_used_fallback,
        "arrival_mean": ph_mean(arrival_alpha, arrival_t),
        "arrival_scv": arrival_scv,
        "arrival_scv_band_min": arrival_scv_min,
        "arrival_scv_band_max": arrival_scv_max,
        "service_mean": service_mean,
        "service_scv": service_scv,
        "service_scv_band_min": service_scv_min,
        "service_scv_band_max": service_scv_max,
        "rho": service_mean / ph_mean(arrival_alpha, arrival_t),
        "sojourn_order": sojourn_t.shape[0],
        "payload_format": args.payload_format,
    }


def write_manifest(output_dir, rows):
    if not rows:
        return None

    manifest_path = Path(output_dir) / "phph1_examples_manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return manifest_path


def clean_output_dir(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    allowed_suffixes = {".pkl", ".csv", ".png"}
    for path in output_dir.iterdir():
        if path.is_file() and path.suffix.lower() in allowed_suffixes:
            path.unlink()


def existing_example_ids(output_dir):
    existing = set()
    pattern = re.compile(r"_ex_(\d+)\.pkl$")
    for path in Path(output_dir).glob("*.pkl"):
        match = pattern.search(path.name)
        if match is not None:
            existing.add(int(match.group(1)))
    return existing


def main():
    args = parse_args()
    families = parse_families(args.families)
    rng = np.random.default_rng(args.seed)

    if args.num_examples < 1:
        raise ValueError("--num-examples must be at least 1.")

    if args.service_mean >= 1.0:
        raise ValueError(
            "The script fixes E[inter-arrival] = 1, so --service-mean must be < 1."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.clean_output:
        clean_output_dir(output_dir)

    existing_ids = existing_example_ids(output_dir) if args.resume else set()
    if existing_ids:
        print(
            f"Resume enabled: found {len(existing_ids)} existing PKLs. "
            "Existing example IDs will be skipped."
        )

    manifest_rows = []
    total_start = time.perf_counter()
    for example_id in range(args.num_examples):
        visible_id = example_id + 1
        example_label = f"example {visible_id}/{args.num_examples}"
        start = time.perf_counter()
        output_example_id = None if args.num_examples == 1 else example_id
        if output_example_id is not None and output_example_id in existing_ids:
            print(f"Skipping {example_label}: PKL already exists.")
            continue

        print(f"Starting {example_label}...")
        try:
            row = generate_example(
                args,
                output_dir,
                rng,
                families,
                example_id=output_example_id,
            )
            manifest_rows.append(row)
        finally:
            elapsed = time.perf_counter() - start
            print(f"Time for {example_label}: {elapsed:.3f} seconds")

    manifest_path = write_manifest(output_dir, manifest_rows)
    if manifest_path is not None:
        print("Saved manifest:", manifest_path)

    total_elapsed = time.perf_counter() - total_start
    print(f"Total time for {args.num_examples} example(s): {total_elapsed:.3f} seconds")


if __name__ == "__main__":
    main()
