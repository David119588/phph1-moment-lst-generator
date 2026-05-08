from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

SUMMARY_PATH = Path(r"C:\phph1_results\plots_l2\summary_valid_by_K.csv")
OUT_DIR = Path(r"C:\phph1_results\plots_l2")

if not SUMMARY_PATH.exists():
    raise FileNotFoundError(f"File not found: {SUMMARY_PATH}")

df = pd.read_csv(SUMMARY_PATH)

print("Loaded:")
print(SUMMARY_PATH)
print(df)

df["K"] = pd.to_numeric(df["K"], errors="coerce")
df["mean_l2"] = pd.to_numeric(df["mean_l2"], errors="coerce")
df["median_l2"] = pd.to_numeric(df["median_l2"], errors="coerce")

df = df.dropna(subset=["K", "mean_l2", "median_l2"]).sort_values("K")

if df.empty:
    raise RuntimeError("summary_valid_by_K.csv is empty after cleaning.")

plt.figure(figsize=(10, 6))

plt.plot(df["K"], df["mean_l2"], marker="o", label="Mean L2")
plt.plot(df["K"], df["median_l2"], marker="s", label="Median L2")

plt.yscale("log")
plt.xlabel("Number of moments K")
plt.ylabel("L2 distance")
plt.title("L2 vs K - valid fits only")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()

out_file = OUT_DIR / "l2_vs_K_valid_fits_only.png"
plt.savefig(out_file, dpi=250)
plt.show()

print("Saved:")
print(out_file)