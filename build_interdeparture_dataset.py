"""
Build a balanced pandas dataset from interdeparture PKLs.

The departure archives contain PKLs named like:
    depart_0_trial_num_0correlation_-0.000100..._0.938sim_time_70000000_model_num_147233.pkl

For each non-empty PKL this script extracts:
    - queue label: 0 or 1
    - first 10 log interdeparture moments
    - first autocorrelation, validated to be in [-1, 1]

The final table is balanced: half of the examples come from queue 0 and half
from queue 1. By default the script writes both CSV and pickle outputs.
"""

import argparse
import pickle
import random
import re
import subprocess
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd


CORRELATION_RE = re.compile(
    r"correlation_(?P<corr>[-+0-9.eE]+)_(?P<rho>[-+0-9.eE]+)sim_time"
)
MODEL_RE = re.compile(r"model_num_(?P<model_num>\d+)\.pkl$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a balanced interdeparture moment/autocorrelation dataset."
    )
    parser.add_argument(
        "--depart-0-dir",
        type=Path,
        default=Path(r"C:\DEPART_0"),
        help="Folder containing extracted queue-0 departure PKLs.",
    )
    parser.add_argument(
        "--depart-1-dir",
        type=Path,
        default=Path(r"C:\DEPART_1"),
        help="Folder containing extracted queue-1 departure PKLs.",
    )
    parser.add_argument(
        "--depart-0-archives",
        type=Path,
        default=Path(r"C:\sojourn_archives\depart_0"),
        help="Folder containing queue-0 .tar.gz archives.",
    )
    parser.add_argument(
        "--depart-1-archives",
        type=Path,
        default=Path(r"C:\sojourn_archives\depart_1"),
        help="Folder containing queue-1 .tar.gz archives.",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract .tar.gz archives into the departure folders before loading.",
    )
    parser.add_argument(
        "--delete-empty",
        action="store_true",
        help="Delete zero-byte PKLs from the departure folders before loading.",
    )
    parser.add_argument(
        "--examples-per-queue",
        type=int,
        default=25000,
        help="How many examples to sample from each queue.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed used for balanced sampling.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Folder where the output CSV/PKL files are saved.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="interdeparture_balanced_dataset",
        help="Output filename prefix.",
    )
    return parser.parse_args()


def extract_archives(archive_dir, target_dir):
    target_dir.mkdir(parents=True, exist_ok=True)
    archives = sorted(archive_dir.glob("*.tar.gz"))
    if not archives:
        raise FileNotFoundError(f"No .tar.gz archives found in {archive_dir}")

    for archive_path in archives:
        print(f"Extracting {archive_path.name} -> {target_dir}")
        if try_system_tar_extract(archive_path, target_dir):
            continue
        with tarfile.open(archive_path, mode="r:gz") as tar:
            safe_extract(tar, target_dir)


def try_system_tar_extract(archive_path, target_dir):
    """
    Windows tar can extract these archives even when gzip reports a missing
    end-of-stream marker. Python tarfile raises EOFError for that case.
    """
    try:
        result = subprocess.run(
            ["tar", "-xzf", str(archive_path), "-C", str(target_dir)],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return False

    if result.returncode == 0:
        return True

    warning_text = (result.stderr or "") + (result.stdout or "")
    if "Truncated tar archive" in warning_text or "end-of-stream marker" in warning_text:
        print(f"Warning while extracting {archive_path.name}:")
        print(warning_text.strip())
        print("Continuing because tar extracted the readable PKLs.")
        return True

    raise RuntimeError(
        f"tar failed for {archive_path} with code {result.returncode}:\n"
        f"{warning_text}"
    )


def safe_extract(tar, target_dir):
    target_root = target_dir.resolve()
    for member in tar.getmembers():
        member_path = (target_root / member.name).resolve()
        if target_root != member_path and target_root not in member_path.parents:
            raise ValueError(f"Archive member escapes target folder: {member.name}")
    tar.extractall(target_root)


def delete_empty_pkls(folder):
    deleted = 0
    if not folder.exists():
        return deleted

    for path in folder.glob("*.pkl"):
        if path.stat().st_size == 0:
            path.unlink()
            deleted += 1
    return deleted


def filename_metadata(path):
    corr_match = CORRELATION_RE.search(path.name)
    if corr_match is None:
        raise ValueError(f"Could not parse first autocorrelation from {path.name}")

    model_match = MODEL_RE.search(path.name)
    return {
        "first_autocorrelation": float(corr_match.group("corr")),
        "rho_or_tag": float(corr_match.group("rho")),
        "model_num": int(model_match.group("model_num")) if model_match else -1,
    }


def flatten_numeric_arrays(payload):
    arrays = []

    def visit(value):
        if isinstance(value, np.ndarray):
            if np.issubdtype(value.dtype, np.number):
                arrays.append(np.asarray(value, dtype=float).ravel())
            return
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(payload)
    return arrays


def choose_log_moments(arrays):
    """
    Prefer the first numeric array with at least 10 values.

    In the inspected depart_0 files, this is the 20-value array whose first
    10 values are the log interdeparture moments. The code keeps this generic
    so it also works if the PKL payload is a tuple/list/dict after pickle.load.
    """
    candidates = [arr for arr in arrays if arr.size >= 10]
    if not candidates:
        raise ValueError("No numeric array with at least 10 values was found.")
    return candidates[0][:10].astype(float)


def choose_first_autocorrelation(arrays, filename_corr):
    """
    Use the filename value as the authoritative first autocorrelation.

    The extracted files also contain autocorrelation sequences internally, but
    the filename value is explicit and matched the inspected payload.
    """
    corr = float(filename_corr)
    if not np.isfinite(corr):
        raise ValueError(f"Invalid autocorrelation value: {corr}")
    if corr < -1.0 or corr > 1.0:
        raise ValueError(f"Autocorrelation outside [-1, 1]: {corr}")
    return corr


def row_from_pickle(path, queue_id):
    if path.stat().st_size == 0:
        raise ValueError(f"Empty PKL: {path}")

    metadata = filename_metadata(path)
    with open(path, "rb") as f:
        payload = pickle.load(f)

    arrays = flatten_numeric_arrays(payload)
    log_moments = choose_log_moments(arrays)
    if not np.all(np.isfinite(log_moments)):
        raise ValueError(f"Non-finite log moments in {path}")

    row = {
        "queue_id": int(queue_id),
        "source_file": path.name,
        "source_path": str(path),
        "model_num": metadata["model_num"],
        "rho_or_tag": metadata["rho_or_tag"],
        "first_autocorrelation": choose_first_autocorrelation(
            arrays, metadata["first_autocorrelation"]
        ),
    }
    for index, value in enumerate(log_moments, start=1):
        row[f"log_interdeparture_moment_{index}"] = float(value)
    return row


def load_queue_rows(folder, queue_id):
    rows = []
    failures = []
    paths = sorted(path for path in folder.glob("*.pkl") if path.stat().st_size > 0)
    if not paths:
        raise FileNotFoundError(f"No non-empty PKLs found in {folder}")

    for path in paths:
        try:
            rows.append(row_from_pickle(path, queue_id))
        except Exception as exc:
            failures.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})

    if failures:
        print(f"Skipped {len(failures)} files from queue {queue_id}.")
    return rows, failures


def balanced_sample(rows_0, rows_1, examples_per_queue, seed):
    n = min(examples_per_queue, len(rows_0), len(rows_1))
    if n < examples_per_queue:
        print(
            f"Requested {examples_per_queue} per queue, but using {n} because "
            "one queue has fewer usable examples."
        )

    rng = random.Random(seed)
    sampled = rng.sample(rows_0, n) + rng.sample(rows_1, n)
    rng.shuffle(sampled)
    return sampled


def save_outputs(rows, failures, output_dir, output_prefix):
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    ordered_columns = (
        ["queue_id", "first_autocorrelation"]
        + [f"log_interdeparture_moment_{i}" for i in range(1, 11)]
        + ["model_num", "rho_or_tag", "source_file", "source_path"]
    )
    df = df[ordered_columns]

    csv_path = output_dir / f"{output_prefix}.csv"
    pkl_path = output_dir / f"{output_prefix}.pkl"
    df.to_csv(csv_path, index=False)
    df.to_pickle(pkl_path)

    failures_path = None
    if failures:
        failures_path = output_dir / f"{output_prefix}_failures.csv"
        pd.DataFrame(failures).to_csv(failures_path, index=False)

    return df, csv_path, pkl_path, failures_path


def main():
    args = parse_args()

    if args.extract:
        extract_archives(args.depart_0_archives, args.depart_0_dir)
        extract_archives(args.depart_1_archives, args.depart_1_dir)

    if args.delete_empty:
        deleted_0 = delete_empty_pkls(args.depart_0_dir)
        deleted_1 = delete_empty_pkls(args.depart_1_dir)
        print(f"Deleted empty PKLs: queue_0={deleted_0}, queue_1={deleted_1}")

    rows_0, failures_0 = load_queue_rows(args.depart_0_dir, queue_id=0)
    rows_1, failures_1 = load_queue_rows(args.depart_1_dir, queue_id=1)
    rows = balanced_sample(rows_0, rows_1, args.examples_per_queue, args.seed)
    df, csv_path, pkl_path, failures_path = save_outputs(
        rows,
        failures_0 + failures_1,
        args.output_dir,
        args.output_prefix,
    )

    print(f"Saved CSV: {csv_path}")
    print(f"Saved pickle: {pkl_path}")
    if failures_path is not None:
        print(f"Saved skipped-file report: {failures_path}")
    print("Rows:", len(df))
    print("Rows by queue:")
    print(df["queue_id"].value_counts().sort_index())
    print("First autocorrelation range:")
    print(df["first_autocorrelation"].agg(["min", "max"]))


if __name__ == "__main__":
    main()
