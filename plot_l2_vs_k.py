from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

FITS_PATH = Path(r"C:\phph1_results\fits_all.csv")
OUT_DIR = Path(r"C:\phph1_results\plots_l2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

fits = pd.read_csv(FITS_PATH)

fits["K"] = pd.to_numeric(fits["K"], errors="coerce")
fits["l2_norm"] = pd.to_numeric(fits["l2_norm"], errors="coerce")

def to_bool(series):
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])

for col in ["fit_valid", "is_in_0_1", "is_monotone"]:
    if col in fits.columns:
        fits[col] = to_bool(fits[col])

fits_ok = fits[
    (fits["fit_status"].astype(str).str.lower() == "ok")
    & fits["K"].notna()
    & fits["l2_norm"].notna()
].copy()

valid = fits_ok.copy()

if "fit_valid" in valid.columns:
    valid = valid[valid["fit_valid"] == True]

if "is_in_0_1" in valid.columns:
    valid = valid[valid["is_in_0_1"] == True]

if "is_monotone" in valid.columns:
    valid = valid[valid["is_monotone"] == True]

summary_ok = (
    fits_ok
    .groupby("K")
    .agg(
        rows=("l2_norm", "count"),
        mean_l2=("l2_norm", "mean"),
        median_l2=("l2_norm", "median"),
        max_l2=("l2_norm", "max"),
    )
    .reset_index()
    .sort_values("K")
)

summary_valid = (
    valid
    .groupby("K")
    .agg(
        rows=("l2_norm", "count"),
        mean_l2=("l2_norm", "mean"),
        median_l2=("l2_norm", "median"),
        max_l2=("l2_norm", "max"),
    )
    .reset_index()
    .sort_values("K")
)

print("All successful fits:")
print(summary_ok)

print("\nValid fits only:")
print(summary_valid)

summary_ok.to_csv(OUT_DIR / "summary_ok_by_K.csv", index=False)
summary_valid.to_csv(OUT_DIR / "summary_valid_by_K.csv", index=False)

# גרף 1: כל ההתאמות שהצליחו
plt.figure(figsize=(10, 6))
plt.plot(summary_ok["K"], summary_ok["mean_l2"], marker="o", label="Mean L2")
plt.plot(summary_ok["K"], summary_ok["median_l2"], marker="s", label="Median L2")
plt.yscale("log")
plt.xlabel("Number of moments K")
plt.ylabel("L2 distance")
plt.title("L2 vs K - successful fits")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "l2_vs_K_successful_fits.png", dpi=250)
plt.show()

# גרף 2: רק התאמות תקינות מתמטית
plt.figure(figsize=(10, 6))
plt.plot(summary_valid["K"], summary_valid["mean_l2"], marker="o", label="Mean L2")
plt.plot(summary_valid["K"], summary_valid["median_l2"], marker="s", label="Median L2")
plt.yscale("log")
plt.xlabel("Number of moments K")
plt.ylabel("L2 distance")
plt.title("L2 vs K - valid fits only")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "l2_vs_K_valid_fits_only.png", dpi=250)
plt.show()

# גרף 3: לפי גודל PH
if "size_pair" in valid.columns:
    by_size = (
        valid
        .groupby(["size_pair", "K"])
        .agg(
            rows=("l2_norm", "count"),
            mean_l2=("l2_norm", "mean"),
            median_l2=("l2_norm", "median"),
        )
        .reset_index()
        .sort_values(["size_pair", "K"])
    )

    by_size.to_csv(OUT_DIR / "summary_valid_by_size_and_K.csv", index=False)

    plt.figure(figsize=(11, 7))

    for size_pair, grp in by_size.groupby("size_pair"):
        grp = grp.sort_values("K")
        plt.plot(grp["K"], grp["median_l2"], marker="o", label=size_pair)

    plt.yscale("log")
    plt.xlabel("Number of moments K")
    plt.ylabel("Median L2 distance")
    plt.title("Median L2 vs K by PH size")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "median_l2_vs_K_by_PH_size.png", dpi=250)
    plt.show()

print("\nDone. Plots saved in:")
print(OUT_DIR)