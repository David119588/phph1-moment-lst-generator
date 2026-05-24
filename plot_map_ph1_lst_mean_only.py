from pathlib import Path
import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


OUT_DIR = Path("map_ph1_lst_moment_budget_1000")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot only mean L2 vs odd moments.")
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    summary_path = args.output_dir / "map_ph1_lst_l2_summary_by_K.csv"
    out_file = args.output_dir / "map_ph1_lst_mean_l2_vs_odd_moments_up_to_15.png"

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")

    df = pd.read_csv(summary_path)
    df["K"] = pd.to_numeric(df["K"], errors="coerce")
    df["mean_l2"] = pd.to_numeric(df["mean_l2"], errors="coerce")
    df = df.dropna(subset=["K", "mean_l2"]).sort_values("K")

    fig, axis = plt.subplots(figsize=(9, 6))
    axis.plot(df["K"], df["mean_l2"], marker="o", linewidth=2.5, color="#2f76b7", label="Mean L2")
    axis.set_title("Mean MAP/PH/1 LST reconstruction error vs number of moments", fontweight="bold")
    axis.set_xlabel("Number of moments K", fontweight="bold")
    axis.set_ylabel("Mean L2 distance", fontweight="bold")
    axis.set_xticks(df["K"].astype(int).tolist())
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(out_file, dpi=250)
    plt.close(fig)

    print("Saved plot:", out_file)


if __name__ == "__main__":
    main()
