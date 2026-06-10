"""
Fit MAPs directly from departure/interdeparture traces using BuTools.

This is different from the earlier coverage plots: those plots sampled MAPs
from a descriptor region. This script tries to read the raw trace from each PKL
and calls butools.fitting.MAPFromTrace on that observed sequence.

Important: MAPFromTrace requires raw observations, for example interdeparture
times. PKLs that contain only moments/descriptors are reported in the failures
CSV and skipped.
"""

from __future__ import annotations

import argparse
import inspect
import math
import os
import pickle
import random
import sys
import tarfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TRACE_KEYS = (
    "interdeparture_times",
    "inter_departure_times",
    "interdepartures",
    "inter_departures",
    "interdeparture",
    "inter_departure",
    "trace",
)

DEPARTURE_TIME_KEYS = (
    "departure_times",
    "departures",
    "departure_trace",
    "departure_moments",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit MAPs from raw departure/interdeparture traces in PKLs."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Folder of PKLs, one PKL file, or one .tar/.tar.gz archive.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("map_from_trace_fit"))
    parser.add_argument("--num-files", type=int, default=20)
    parser.add_argument(
        "--station",
        type=int,
        choices=(0, 1, 2),
        default=0,
        help="Use 1 or 2 to fit only one tandem station; 0 uses all matching PKLs.",
    )
    parser.add_argument("--map-order", type=int, default=5)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--max-trace-length", type=int, default=200000)
    parser.add_argument("--min-trace-length", type=int, default=50)
    parser.add_argument("--normalize-mean", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--stop-cond", type=float, default=1e-7)
    return parser.parse_args()


def add_butools_paths() -> None:
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
            Path("/scratch200/davidfine/butools2/Python"),
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            text = str(candidate)
            if text not in sys.path:
                sys.path.append(text)


def load_butools():
    add_butools_paths()
    try:
        import butools
        from butools.fitting import MAPFromTrace
        from butools.map import LagCorrelationsFromMAP, MarginalMomentsFromMAP

        butools.verbose = False
        butools.checkInput = False
        return MAPFromTrace, MarginalMomentsFromMAP, LagCorrelationsFromMAP
    except Exception as exc:
        raise ImportError(
            "Could not import BuTools fitting tools. Set BUTOOLS_PATH to the "
            f"BuTools Python folder. Original error: {type(exc).__name__}: {exc}"
        )


def iter_sources(input_path: Path) -> list[tuple[str, Any]]:
    if input_path.is_dir():
        return [(str(path), path) for path in sorted(input_path.rglob("*.pkl"))]
    if input_path.suffix == ".pkl":
        return [(str(input_path), input_path)]
    if input_path.name.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(input_path, "r:*") as tar:
            return [
                (f"{input_path}!{member.name}", (input_path, member.name))
                for member in tar.getmembers()
                if member.isfile() and member.name.endswith(".pkl")
            ]
    raise ValueError(f"Unsupported input path: {input_path}")


def station_matches(source_name: str, station: int) -> bool:
    if station == 0:
        return True
    marker = f"station{station}"
    return marker in Path(source_name).name.lower()


def load_payload(source: Any):
    if isinstance(source, Path):
        with open(source, "rb") as handle:
            return pickle.load(handle)

    archive_path, member_name = source
    with tarfile.open(archive_path, "r:*") as tar:
        extracted = tar.extractfile(member_name)
        if extracted is None:
            raise ValueError(f"Could not read archive member: {member_name}")
        return pickle.load(extracted)


def numeric_array(value: Any) -> np.ndarray | None:
    try:
        arr = np.asarray(value, dtype=float).ravel()
    except Exception:
        return None
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    return arr


def first_matching_dict_trace(payload: dict[str, Any]) -> tuple[np.ndarray | None, str]:
    lowered = {str(key).lower(): key for key in payload.keys()}

    for key in TRACE_KEYS:
        original = lowered.get(key)
        if original is not None:
            arr = numeric_array(payload[original])
            if arr is not None:
                return arr, f"dict key {original} as interdeparture trace"

    for key in DEPARTURE_TIME_KEYS:
        original = lowered.get(key)
        if original is not None:
            arr = numeric_array(payload[original])
            if arr is not None:
                return np.diff(arr), f"dict key {original} as departure times"

    return None, ""


def walk_numeric_arrays(payload: Any) -> list[tuple[str, np.ndarray]]:
    arrays: list[tuple[str, np.ndarray]] = []

    def visit(value: Any, label: str) -> None:
        arr = numeric_array(value)
        if arr is not None:
            arrays.append((label, arr))
            return
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, f"{label}.{key}")
            return
        if isinstance(value, (list, tuple)):
            for idx, child in enumerate(value):
                visit(child, f"{label}[{idx}]")

    visit(payload, "payload")
    return arrays


def looks_like_departure_times(arr: np.ndarray) -> bool:
    if arr.size < 3:
        return False
    diffs = np.diff(arr)
    return bool(np.all(diffs > 0) and np.median(diffs) > 0)


def extract_trace(payload: Any) -> tuple[np.ndarray | None, str]:
    if isinstance(payload, dict):
        trace, reason = first_matching_dict_trace(payload)
        if trace is not None:
            return trace, reason

    arrays = walk_numeric_arrays(payload)
    long_arrays = [(label, arr) for label, arr in arrays if arr.size >= 50]
    if not long_arrays:
        return None, "no numeric array long enough to be a trace"

    for label, arr in long_arrays:
        if looks_like_departure_times(arr):
            return np.diff(arr), f"{label} interpreted as departure times"

    positive = [
        (label, arr)
        for label, arr in long_arrays
        if np.all(arr > 0) and np.std(arr) > 0
    ]
    if positive:
        label, arr = max(positive, key=lambda item: item[1].size)
        return arr, f"{label} interpreted as interdeparture trace"

    return None, "numeric arrays exist, but none look like positive trace data"


def clean_trace(trace: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray | None, str, float]:
    trace = np.asarray(trace, dtype=float).ravel()
    trace = trace[np.isfinite(trace)]
    trace = trace[trace > 0]
    if trace.size < args.min_trace_length:
        return None, f"trace too short after cleaning: {trace.size}", math.nan

    if trace.size > args.max_trace_length:
        rng = np.random.default_rng(args.seed)
        indices = np.sort(rng.choice(trace.size, size=args.max_trace_length, replace=False))
        trace = trace[indices]

    scale = float(np.mean(trace))
    if not np.isfinite(scale) or scale <= 0:
        return None, "invalid trace mean", math.nan
    if args.normalize_mean:
        trace = trace / scale
    return trace, "", scale


def trace_stats(trace: np.ndarray) -> dict[str, float]:
    mean = float(np.mean(trace))
    var = float(np.var(trace))
    centered = trace - mean
    if var <= 0:
        scv = skew = kurt = math.nan
    else:
        scv = var / (mean * mean)
        skew = float(np.mean(centered**3) / (var ** 1.5))
        kurt = float(np.mean(centered**4) / (var * var))

    if trace.size > 1 and np.std(trace[:-1]) > 0 and np.std(trace[1:]) > 0:
        lag1 = float(np.corrcoef(trace[:-1], trace[1:])[0, 1])
    else:
        lag1 = math.nan

    return {
        "trace_length": int(trace.size),
        "mean": mean,
        "scv": float(scv),
        "skewness": float(skew),
        "kurtosis": float(kurt),
        "lag1_autocorrelation": lag1,
    }


def central_stats_from_raw_moments(moments: list[float]) -> dict[str, float]:
    m1, m2, m3, m4 = [float(x) for x in moments[:4]]
    var = m2 - m1 * m1
    if var <= 0 or m1 <= 0:
        return {"map_mean": m1, "map_scv": math.nan, "map_skewness": math.nan, "map_kurtosis": math.nan}
    mu3 = m3 - 3 * m2 * m1 + 2 * m1**3
    mu4 = m4 - 4 * m3 * m1 + 6 * m2 * m1**2 - 3 * m1**4
    return {
        "map_mean": m1,
        "map_scv": var / (m1 * m1),
        "map_skewness": mu3 / (var ** 1.5),
        "map_kurtosis": mu4 / (var * var),
    }


def fit_map(MAPFromTrace, trace: np.ndarray, args: argparse.Namespace):
    attempts = [
        (args.map_order, {"maxIter": args.max_iter, "stopCond": args.stop_cond}),
        ([args.map_order], {"maxIter": args.max_iter, "stopCond": args.stop_cond}),
        (args.map_order, {}),
        ([args.map_order], {}),
    ]
    last_error = None
    for order_arg, kwargs in attempts:
        try:
            result = MAPFromTrace(trace, order_arg, **kwargs)
            if isinstance(result, tuple) and len(result) >= 2:
                return np.asarray(result[0], dtype=float), np.asarray(result[1], dtype=float)
            if isinstance(result, list) and len(result) >= 2:
                return np.asarray(result[0], dtype=float), np.asarray(result[1], dtype=float)
            raise ValueError(f"Unexpected MAPFromTrace result type: {type(result).__name__}")
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"MAPFromTrace failed: {type(last_error).__name__}: {last_error}")


def map_stats(D0: np.ndarray, D1: np.ndarray, MarginalMomentsFromMAP, LagCorrelationsFromMAP):
    moments = MarginalMomentsFromMAP(D0, D1, 4)
    stats = central_stats_from_raw_moments(moments)
    try:
        lag = LagCorrelationsFromMAP(D0, D1, 1)
        stats["map_lag1_autocorrelation"] = float(np.asarray(lag).ravel()[0])
    except Exception:
        stats["map_lag1_autocorrelation"] = math.nan
    return stats


def plot_results(summary: pd.DataFrame, output_dir: Path) -> None:
    if summary.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].scatter(summary["lag1_autocorrelation"], summary["map_lag1_autocorrelation"], s=20)
    axes[0].set_xlabel("empirical trace lag-1 autocorrelation")
    axes[0].set_ylabel("fitted MAP lag-1 autocorrelation")
    axes[0].grid(alpha=0.3)

    axes[1].scatter(summary["scv"], summary["map_scv"], s=20)
    axes[1].set_xlabel("empirical trace SCV")
    axes[1].set_ylabel("fitted MAP SCV")
    axes[1].grid(alpha=0.3)

    axes[2].scatter(summary["skewness"], summary["map_skewness"], s=20, label="skewness")
    axes[2].scatter(summary["kurtosis"], summary["map_kurtosis"], s=20, label="kurtosis")
    axes[2].set_xlabel("empirical trace value")
    axes[2].set_ylabel("fitted MAP value")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    fig.suptitle("MAPFromTrace fitting diagnostics")
    fig.tight_layout()
    fig.savefig(output_dir / "map_from_trace_descriptor_fit.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    MAPFromTrace, MarginalMomentsFromMAP, LagCorrelationsFromMAP = load_butools()
    with open(args.output_dir / "map_from_trace_signature.txt", "w", encoding="utf-8") as handle:
        handle.write(str(inspect.signature(MAPFromTrace)))

    sources = [
        item for item in iter_sources(args.input) if station_matches(item[0], args.station)
    ]
    if not sources:
        raise RuntimeError(f"No PKLs matched input={args.input} and station={args.station}")
    rng = random.Random(args.seed)
    rng.shuffle(sources)
    selected = sources[: args.num_files]

    rows = []
    failures = []
    maps_dir = args.output_dir / "fitted_maps"
    maps_dir.mkdir(exist_ok=True)

    for index, (source_name, source) in enumerate(selected):
        try:
            payload = load_payload(source)
            raw_trace, trace_reason = extract_trace(payload)
            if raw_trace is None:
                raise ValueError(trace_reason)
            trace, clean_reason, original_mean = clean_trace(raw_trace, args)
            if trace is None:
                raise ValueError(clean_reason)

            D0, D1 = fit_map(MAPFromTrace, trace, args)
            empirical = trace_stats(trace)
            fitted = map_stats(D0, D1, MarginalMomentsFromMAP, LagCorrelationsFromMAP)
            out_path = maps_dir / f"map_from_trace_{index:05d}_order_{args.map_order}.pkl"
            with open(out_path, "wb") as handle:
                pickle.dump(
                    {
                        "D0": D0,
                        "D1": D1,
                        "source": source_name,
                        "map_order": args.map_order,
                        "normalized_trace_mean": args.normalize_mean,
                        "original_trace_mean": original_mean,
                        "trace_extraction": trace_reason,
                    },
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

            row = {
                "source": source_name,
                "fit_path": str(out_path),
                "trace_extraction": trace_reason,
                "original_trace_mean": original_mean,
                "map_order": args.map_order,
                **empirical,
                **fitted,
            }
            rows.append(row)
            print(
                f"fit {index + 1}/{len(selected)} ok: "
                f"lag1={row['lag1_autocorrelation']:.4g}, "
                f"map_lag1={row['map_lag1_autocorrelation']:.4g}, "
                f"scv={row['scv']:.4g}, map_scv={row['map_scv']:.4g}"
            )
        except Exception as exc:
            failures.append(
                {
                    "source": source_name,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            print(f"skip {index + 1}/{len(selected)}: {source_name}: {type(exc).__name__}: {exc}")

    summary = pd.DataFrame(rows)
    failure_df = pd.DataFrame(failures)
    summary.to_csv(args.output_dir / "map_from_trace_fit_summary.csv", index=False)
    failure_df.to_csv(args.output_dir / "map_from_trace_fit_failures.csv", index=False)
    plot_results(summary, args.output_dir)

    print(f"Saved summary: {args.output_dir / 'map_from_trace_fit_summary.csv'}")
    print(f"Saved failures: {args.output_dir / 'map_from_trace_fit_failures.csv'}")
    print(f"Saved fitted MAPs: {maps_dir}")
    print(f"Successful fits: {len(summary)}")
    print(f"Failures/skips: {len(failure_df)}")


if __name__ == "__main__":
    main()
