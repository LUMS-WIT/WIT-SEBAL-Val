import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# -----------------------------
# Inputs & Outputs
# -----------------------------
merged_csv = r"merged_calibration_multisite.csv"   # merged SEBAL + Tower dataset
out_dir    = r"./bowen_calibration"
os.makedirs(out_dir, exist_ok=True)

params_json = os.path.join(out_dir, "global_Bowen_params.json")
report_csv  = os.path.join(out_dir, "Bowen_calibration_report.csv")
out_csv     = os.path.join(out_dir, "merged_calibration_Bowen_corrected.csv")

# -----------------------------
# 1) Load data
# -----------------------------
df = pd.read_csv(merged_csv)

# Ensure numeric (include H columns for β)
for col in ["Rn_inst_avg","G_inst_avg","H_inst_avg","LE_inst_avg",
            "NETRAD","G_F_MDS","H_F_MDS","LE_F_MDS","EF_inst_avg"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# -----------------------------
# 1b) Filtering rules (row-wise remove)
# -----------------------------
mask_valid  = df[["H_F_MDS","LE_F_MDS"]].notna().all(axis=1)        # need both tower H, LE for β_t
mask_valid &= df[["H_inst_avg","LE_inst_avg"]].notna().all(axis=1)  # need model H, LE for β_m
mask_valid &= (df["LE_inst_avg"] <= 800) & (df["LE_inst_avg"] >= -400)  # practical LE bounds
mask_valid &= df["Rn_inst_avg"].notna() & df["G_inst_avg"].notna()      # for plotting/diagnostics

before = len(df)
df = df[mask_valid].copy()
after = len(df)
print(f"✅ Filtering complete: removed {before - after} rows, kept {after}")

# -----------------------------
# 2) Define β for tower and model
# -----------------------------
# Tower: raw β_t from measured H, LE (closure later is implicit through β)
beta_t = df["H_F_MDS"] / df["LE_F_MDS"]

# Model: raw β_m from delivered H, LE (no model closure)
beta_m = df["H_inst_avg"] / df["LE_inst_avg"]

# Optional: Available energy (diagnostic only)
A_t = df["NETRAD"] - df["G_F_MDS"]
A_m = df["Rn_inst_avg"] - df["G_inst_avg"]

# -----------------------------
# 3) QC for regression in β-space
#     (avoid division blow-ups & non-midday extremes)
# -----------------------------
eps = 5.0  # W/m², avoid tiny |LE| in β_m
mask_reg  = np.isfinite(beta_m) & np.isfinite(beta_t)
mask_reg &= np.isfinite(df["LE_inst_avg"]) & (np.abs(df["LE_inst_avg"]) >= eps)
mask_reg &= np.isfinite(df["LE_F_MDS"])    & (np.abs(df["LE_F_MDS"])    >= eps)
# reasonable β range (tune for site/season):
mask_reg &= (beta_m > -0.5) & (beta_m < 5.0) & (beta_t > -0.5) & (beta_t < 5.0)
# (optional) require positive available energy at tower
mask_reg &= np.isfinite(A_t) & (A_t > 0)

print("Valid calibration pairs (β):", int(mask_reg.sum()))
if mask_reg.sum() < 3:
    print("⚠️ Not enough pairs for a stable β fit. Results may be unreliable.")

# -----------------------------
# 4) Fit β calibration: β_t ≈ α_b * β_m + β_b
# -----------------------------
if mask_reg.sum() >= 3:
    reg_b = LinearRegression()
    reg_b.fit(beta_m[mask_reg].to_numpy().reshape(-1,1), beta_t[mask_reg].to_numpy())
    alpha_b, beta_b = float(reg_b.coef_[0]), float(reg_b.intercept_)
else:
    alpha_b, beta_b = 1.0, 0.0

print(f"β_tower ≈ {alpha_b:.3f} * β_model + {beta_b:.3f}")

# -----------------------------
# 5) Apply β mapping WITHOUT imposing model closure
#     Conserve the model turbulent-flux sum S_m = H_m + LE_m
# -----------------------------
S_m = df["H_inst_avg"] + df["LE_inst_avg"]   # turbulent sum from model

beta_corr = alpha_b * beta_m + beta_b
EF_corr   = 1.0 / (1.0 + beta_corr)          # EF from corrected β

# Correct turbulent partition while conserving S_m
LE_corr_B = EF_corr * S_m
H_corr_B  = (1.0 - EF_corr) * S_m

# -----------------------------
# (EF-based path kept here but commented out)
# -----------------------------
# # Tower EF by closure:
# # EF_t = df["LE_F_MDS"] / (df["NETRAD"] - df["G_F_MDS"])
# # Model EF (no closure): prefer EF_inst_avg if present
# # if "EF_inst_avg" in df.columns and df["EF_inst_avg"].notna().any():
# #     EF_m = df["EF_inst_avg"]
# # else:
# #     EF_m = df["LE_inst_avg"] / (A_m)  # fallback (uses closure on model)
# # mask_EF = np.isfinite(EF_m) & np.isfinite(EF_t)
# # reg = LinearRegression().fit(EF_m[mask_EF].to_numpy().reshape(-1,1), EF_t[mask_EF].to_numpy())
# # alpha, beta = float(reg.coef_[0]), float(reg.intercept_)
# # EF_corr = alpha * EF_m + beta
# # ratio   = np.where(np.isfinite(EF_m) & (EF_m != 0), EF_corr/EF_m, np.nan)
# # LE_corr_EF = df["LE_inst_avg"] * ratio
# # H_corr_EF  = (df["Rn_inst_avg"] - df["G_inst_avg"]) - LE_corr_EF  # diagnostic only

# -----------------------------
# 6) Save calibration params
# -----------------------------
params = {
    "mode": "BOWEN_RATIO_PARTITION_NO_MODEL_CLOSURE",
    "Bowen_map": {"alpha_b": alpha_b, "beta_b": beta_b},
    "notes": "β mapping on overlapping dates; corrected β partitions the model turbulent sum S_m = Hm+LEm."
}
with open(params_json, "w") as f:
    json.dump(params, f, indent=2)

# -----------------------------
# 7) Evaluation on LE (before vs after)
# -----------------------------
def stats(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 3:
        return np.nan, np.nan, np.nan
    err = y_pred[mask] - y_true[mask]
    rmse = float(np.sqrt(np.mean(err**2)))
    bias = float(np.mean(err))
    r2 = float(np.corrcoef(y_true[mask], y_pred[mask])[0,1]**2)
    return rmse, r2, bias

LE_t = df["LE_F_MDS"]
rmse_b, r2_b, bias_b     = stats(LE_t, df["LE_inst_avg"])   # before
rmse_bow, r2_bow, bias_bow = stats(LE_t, LE_corr_B)         # after Bowen

report = pd.DataFrame({
    "metric": ["RMSE", "R2", "Bias"],
    "Before_LE": [rmse_b, r2_b, bias_b],
    "After_Bowen_LE":  [rmse_bow, r2_bow, bias_bow]
})
report.to_csv(report_csv, index=False)
print(report)

# -----------------------------
# 8) Detailed report (sites & dates used in β fit)
# -----------------------------
mask_used = mask_reg
cols_keep = [c for c in ["site","scene","date","H_inst_avg","LE_inst_avg","H_F_MDS","LE_F_MDS"] if c in df.columns]
df_pairs = df.loc[mask_used, cols_keep].copy()
df_pairs["beta_model"] = beta_m[mask_used].values
df_pairs["beta_tower"] = beta_t[mask_used].values
if "date" in df_pairs.columns:
    df_pairs["date"] = pd.to_datetime(df_pairs["date"], errors="coerce").dt.date
if "site" not in df_pairs.columns:
    df_pairs["site"] = "ALL"

rows = []
for site, g in df_pairs.groupby("site"):
    dates_list = sorted([str(d) for d in pd.Series(g["date"]).dropna().unique()]) if "date" in g.columns else []
    rows.append({"site": site, "n_points": int(len(g)), "dates_used": ", ".join(dates_list)})
by_site = pd.DataFrame(rows).sort_values("site")
summary = pd.DataFrame([{"total_points_used_for_beta_fit": int(df_pairs.shape[0])}])

detailed_xlsx = os.path.join(out_dir, "Bowen_calibration_report_detailed.xlsx")
with pd.ExcelWriter(detailed_xlsx, engine="xlsxwriter") as writer:
    report.to_excel(writer, sheet_name="metrics", index=False)
    by_site.to_excel(writer, sheet_name="by_site", index=False)
    df_pairs.to_excel(writer, sheet_name="pairs", index=False)
    summary.to_excel(writer, sheet_name="summary", index=False)

print("✅ Detailed report written to:", detailed_xlsx)

# -----------------------------
# 9) Save corrected dataset
# -----------------------------
df_out = df.copy()
df_out["beta_model"] = beta_m
df_out["beta_corr"]  = beta_corr
df_out["EF_from_beta_corr"] = EF_corr
df_out["LE_corr_Bowen"] = LE_corr_B
df_out["H_corr_Bowen"]  = H_corr_B
df_out.to_csv(out_csv, index=False, float_format="%.3f")
print(f"✅ Corrected dataset saved: {out_csv}")

# -----------------------------
# 10) Plots
# -----------------------------
def scatter_plot(x, y, title, fname, rmse, r2, bias):
    plt.figure(figsize=(6,6))
    plt.scatter(x, y, c="blue", alpha=0.65, edgecolor="k")
    min_val = np.nanmin([np.nanmin(x), np.nanmin(y)])
    max_val = np.nanmax([np.nanmax(x), np.nanmax(y)])
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="1:1")
    plt.xlabel("Model (W/m²)")
    plt.ylabel("Tower (W/m²)")
    plt.title(title)
    plt.text(0.05, 0.95, f"RMSE={rmse:.1f}\nR²={r2:.2f}\nBias={bias:.1f}",
             transform=plt.gca().transAxes, va="top", ha="left",
             bbox=dict(facecolor="white", alpha=0.7))
    plt.legend(); plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, fname), dpi=300); plt.close()

def calib_scatter_beta(x, y, slope, intercept, title, fname):
    plt.figure(figsize=(6,6))
    plt.scatter(x, y, c="blue", alpha=0.65, edgecolor="k")
    xv = x[mask_reg]; yv = y[mask_reg]
    min_val = np.nanmin([np.nanmin(xv), np.nanmin(yv)])
    max_val = np.nanmax([np.nanmax(xv), np.nanmax(yv)])
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="1:1")
    xx = np.linspace(min_val, max_val, 200)
    yy = slope*xx + intercept
    plt.plot(xx, yy, "g-", label=f"Fit: y={slope:.3f}x+{intercept:.3f}")
    # Fit-score on used pairs
    y_hat = slope*xv + intercept
    r2 = np.corrcoef(yv, y_hat)[0,1]**2 if len(yv)>2 else np.nan
    rmse = np.sqrt(np.mean((y_hat - yv)**2)) if len(yv)>2 else np.nan
    plt.text(0.05, 0.95, f"RMSE={rmse:.2f}\nR²={r2:.2f}", transform=plt.gca().transAxes,
             va="top", ha="left", bbox=dict(facecolor="white", alpha=0.7))
    plt.xlabel("β (Model)"); plt.ylabel("β (Tower)"); plt.title(title)
    plt.legend(); plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, fname), dpi=300); plt.close()

scatter_plot(df["LE_inst_avg"], LE_t, "LE Before Calibration", "LE_before.png", rmse_b, r2_b, bias_b)
scatter_plot(LE_corr_B,        LE_t, "LE After Bowen Calibration", "LE_after_Bowen.png", rmse_bow, r2_bow, bias_bow)
calib_scatter_beta(beta_m, beta_t, alpha_b, beta_b, "β Calibration (Tower vs Model)", "beta_calibration.png")

print(f"✅ Outputs:\n- Params: {params_json}\n- Report: {report_csv}\n- CSV: {out_csv}\n- Plots in: {out_dir}")
