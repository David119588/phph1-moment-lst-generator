"""
Simulate two-station tandem queues with PH renewal primitives.

For each tandem system, the external interarrival PH is normalized to mean 1.
The two service PH means are sampled in the requested utilization interval.
After simulating the requested number of departures, the initial warm-up
departures are discarded and one inter-departure PKL is saved per station.
"""

import argparse
import csv
import math
import pickle
import time
from pathlib import Path

import numpy as np

from sample_ph_family_histograms import (
    random_coxian_ph,
    random_hyper_erlang_ph,
    scale_ph_to_mean,
)
from plot_ph_family_shape_coverage import random_hyper_general_shape_ph


FAMILY_LABELS = {
    "hyper_erlang": "Hyper Erlang",
    "coxian": "Coxian",
    "hyper_general": "Hyper General",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simulate tandem PH queues and save station inter-departure PKLs."
    )
    parser.add_argument("--num-systems", type=int, default=1000)
    parser.add_argument(
        "--example-start",
        type=int,
        default=0,
        help="First global example ID written by this process.",
    )
    parser.add_argument("--num-departures", type=int, default=100000)
    parser.add_argument("--warmup-departures", type=int, default=5000)
    parser.add_argument("--ph-size-min", type=int, default=2)
    parser.add_argument("--ph-size-max", type=int, default=100)
    parser.add_argument("--arrival-mean", type=float, default=1.0)
    parser.add_argument("--utilization-min", type=float, default=0.5)
    parser.add_argument("--utilization-max", type=float, default=0.95)
    parser.add_argument(
        "--families",
        type=str,
        default="hyper_erlang,coxian,hyper_general",
        help="Comma-separated PH families used cyclically.",
    )
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--output-dir", type=Path, default=Path("tandem_departure_pkls"))
    parser.add_argument(
        "--manifest-file",
        type=str,
        default="manifest.csv",
        help="Manifest CSV filename written inside --output-dir.",
    )
    parser.add_argument("--resume", type=int, default=1)
    return parser.parse_args()


def as_row(alpha):
    alpha = np.asarray(alpha, dtype=float)
    if alpha.ndim == 1:
        alpha = alpha.reshape(1, -1)
    return alpha


def col_ones(n):
    return np.ones((n, 1), dtype=float)


def ph_moment(alpha, t_matrix, order):
    alpha = as_row(alpha)
    inv = np.linalg.inv(-t_matrix)
    power = np.linalg.matrix_power(inv, order)
    return float(math.factorial(order) * (alpha @ power @ col_ones(t_matrix.shape[0]))[0, 0])


def ph_mean(alpha, t_matrix):
    return ph_moment(alpha, t_matrix, 1)


def ph_scv(alpha, t_matrix):
    mean = ph_moment(alpha, t_matrix, 1)
    second = ph_moment(alpha, t_matrix, 2)
    return second / mean**2 - 1.0


def parse_families(text):
    families = [item.strip() for item in text.split(",") if item.strip()]
    if not families:
        raise ValueError("--families must contain at least one family.")

    unknown = sorted(set(families) - set(FAMILY_LABELS))
    if unknown:
        raise ValueError(f"Unknown families: {unknown}. Allowed: {sorted(FAMILY_LABELS)}")

    return families


def sample_ph_by_family(family, size, target_mean, rng):
    if family == "hyper_erlang":
        alpha, t_matrix = random_hyper_erlang_ph(size, rng)
    elif family == "coxian":
        alpha, t_matrix = random_coxian_ph(size, rng)
    elif family == "hyper_general":
        alpha, t_matrix = random_hyper_general_shape_ph(size, rng)
    else:
        raise ValueError(f"Unknown PH family: {family}")

    return scale_ph_to_mean(alpha, t_matrix, target_mean=target_mean)


def sample_ph_time(alpha, t_matrix, rng):
    alpha = np.asarray(alpha, dtype=float).ravel()
    state = int(rng.choice(len(alpha), p=alpha / alpha.sum()))
    elapsed = 0.0

    while True:
        rate = -t_matrix[state, state]
        elapsed += rng.exponential(1.0 / rate)

        transition_rates = np.asarray(t_matrix[state], dtype=float).copy()
        transition_rates[state] = 0.0
        transient_mass = transition_rates.sum()
        absorb_rate = max(rate - transient_mass, 0.0)
        total_rate = transient_mass + absorb_rate

        if total_rate <= 0.0 or rng.random() < absorb_rate / total_rate:
            return elapsed

        probs = transition_rates / transient_mass
        state = int(rng.choice(len(alpha), p=probs))


def sample_ph_times(alpha, t_matrix, count, rng):
    values = np.empty(count, dtype=float)
    for idx in range(count):
        values[idx] = sample_ph_time(alpha, t_matrix, rng)
    return values


def simulate_tandem(interarrival_times, service1_times, service2_times):
    arrival_times = np.cumsum(interarrival_times)
    count = len(arrival_times)
    departure1 = np.empty(count, dtype=float)
    departure2 = np.empty(count, dtype=float)

    last_departure1 = 0.0
    last_departure2 = 0.0
    for idx in range(count):
        start1 = max(arrival_times[idx], last_departure1)
        last_departure1 = start1 + service1_times[idx]
        departure1[idx] = last_departure1

        start2 = max(last_departure1, last_departure2)
        last_departure2 = start2 + service2_times[idx]
        departure2[idx] = last_departure2

    return departure1, departure2


def retained_interdeparture_times(departures, warmup_departures):
    if warmup_departures <= 0:
        return np.diff(np.concatenate(([0.0], departures)))
    return np.diff(departures)[warmup_departures - 1 :]


def pkl_payload(
    *,
    example_id,
    station,
    interdeparture_times,
    departures,
    metadata,
):
    return {
        "example_id": example_id,
        "station": station,
        "interdeparture_times": interdeparture_times.astype(float),
        "retained_departure_times": departures.astype(float),
        **metadata,
    }


def validate_args(args):
    if args.num_systems <= 0:
        raise ValueError("--num-systems must be positive.")
    if args.num_departures <= 1:
        raise ValueError("--num-departures must be greater than 1.")
    if not (0 <= args.warmup_departures < args.num_departures):
        raise ValueError("Require 0 <= --warmup-departures < --num-departures.")
    if args.ph_size_min < 1 or args.ph_size_max < args.ph_size_min:
        raise ValueError("Require 1 <= --ph-size-min <= --ph-size-max.")
    if args.arrival_mean <= 0.0:
        raise ValueError("--arrival-mean must be positive.")
    if not (0.0 < args.utilization_min <= args.utilization_max < 1.0):
        raise ValueError("Require 0 < --utilization-min <= --utilization-max < 1.")


def write_manifest_row(path, row, write_header):
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    validate_args(args)
    families = parse_families(args.families)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / args.manifest_file
    write_header = not manifest_path.exists() or not args.resume
    if not args.resume and manifest_path.exists():
        manifest_path.unlink()
        write_header = True

    rng = np.random.default_rng(args.seed)
    retained_count = args.num_departures - args.warmup_departures
    print("Output directory:", args.output_dir)
    print("Systems:", args.num_systems)
    print("Departures per system:", args.num_departures)
    print("Retained inter-departure observations per station:", retained_count)

    for local_index in range(args.num_systems):
        example_id = args.example_start + local_index
        station1_path = args.output_dir / f"tandem_{example_id:06d}_station1_interdepartures.pkl"
        station2_path = args.output_dir / f"tandem_{example_id:06d}_station2_interdepartures.pkl"
        if args.resume and station1_path.exists() and station2_path.exists():
            print(f"Skipping existing system {example_id}")
            continue

        start_time = time.time()
        arrival_family = families[example_id % len(families)]
        service1_family = families[(example_id + 1) % len(families)]
        service2_family = families[(example_id + 2) % len(families)]

        arrival_size = int(rng.integers(args.ph_size_min, args.ph_size_max + 1))
        service1_size = int(rng.integers(args.ph_size_min, args.ph_size_max + 1))
        service2_size = int(rng.integers(args.ph_size_min, args.ph_size_max + 1))
        rho1 = float(rng.uniform(args.utilization_min, args.utilization_max))
        rho2 = float(rng.uniform(args.utilization_min, args.utilization_max))
        service1_mean = rho1 * args.arrival_mean
        service2_mean = rho2 * args.arrival_mean

        arrival_alpha, arrival_t = sample_ph_by_family(
            arrival_family, arrival_size, args.arrival_mean, rng
        )
        service1_alpha, service1_t = sample_ph_by_family(
            service1_family, service1_size, service1_mean, rng
        )
        service2_alpha, service2_t = sample_ph_by_family(
            service2_family, service2_size, service2_mean, rng
        )

        interarrival_times = sample_ph_times(
            arrival_alpha, arrival_t, args.num_departures, rng
        )
        service1_times = sample_ph_times(service1_alpha, service1_t, args.num_departures, rng)
        service2_times = sample_ph_times(service2_alpha, service2_t, args.num_departures, rng)
        departure1, departure2 = simulate_tandem(
            interarrival_times, service1_times, service2_times
        )

        retained1 = retained_interdeparture_times(departure1, args.warmup_departures)
        retained2 = retained_interdeparture_times(departure2, args.warmup_departures)
        retained_departure1 = departure1[args.warmup_departures :]
        retained_departure2 = departure2[args.warmup_departures :]

        metadata = {
            "num_departures": args.num_departures,
            "warmup_departures": args.warmup_departures,
            "retained_count": retained_count,
            "arrival_mean_target": args.arrival_mean,
            "arrival_family": arrival_family,
            "arrival_size": arrival_size,
            "arrival_mean": ph_mean(arrival_alpha, arrival_t),
            "arrival_scv": ph_scv(arrival_alpha, arrival_t),
            "service1_family": service1_family,
            "service1_size": service1_size,
            "service1_mean": ph_mean(service1_alpha, service1_t),
            "service1_scv": ph_scv(service1_alpha, service1_t),
            "service1_utilization": rho1,
            "service2_family": service2_family,
            "service2_size": service2_size,
            "service2_mean": ph_mean(service2_alpha, service2_t),
            "service2_scv": ph_scv(service2_alpha, service2_t),
            "service2_utilization": rho2,
            "seed": args.seed,
        }

        with open(station1_path, "wb") as f:
            pickle.dump(
                pkl_payload(
                    example_id=example_id,
                    station=1,
                    interdeparture_times=retained1,
                    departures=retained_departure1,
                    metadata=metadata,
                ),
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        with open(station2_path, "wb") as f:
            pickle.dump(
                pkl_payload(
                    example_id=example_id,
                    station=2,
                    interdeparture_times=retained2,
                    departures=retained_departure2,
                    metadata=metadata,
                ),
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        row = {
            "example_id": example_id,
            "station1_pkl": str(station1_path),
            "station2_pkl": str(station2_path),
            **metadata,
            "retained1_mean": float(np.mean(retained1)),
            "retained2_mean": float(np.mean(retained2)),
            "elapsed_seconds": time.time() - start_time,
        }
        write_manifest_row(manifest_path, row, write_header)
        write_header = False
        print(
            f"System {example_id:06d}: "
            f"rho1={rho1:.4f}, rho2={rho2:.4f}, "
            f"retained={len(retained1)}/{len(retained2)}, "
            f"elapsed={row['elapsed_seconds']:.2f}s"
        )


if __name__ == "__main__":
    main()
