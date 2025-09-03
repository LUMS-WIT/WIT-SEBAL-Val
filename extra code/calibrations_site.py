import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# -----------------------------
# Inputs & Outputs
# -----------------------------
merged_csv = r"merged_calibration_multisite.csv"   # requires a 'site' column
out_dir    = r"./ef_calibration"
os.makedirs(out_dir, exist_ok=True)

site_params_json = os.path.join(out_dir, "per_site_EF_params.json")
metrics_csv      = os.path.join(out_dir, "per_site_metrics.csv")
detailed_xlsx    = os.path.join(out_dir, "per_site_details.xlsx")
corrected_csv    = os.path.join(out_dir, "merged_calibration_EF_corrected_per_site.csv")

# -----------------------------
# Options
# -----------------------------
A_MIN   = 1e-6          # avoid divide-by-zero in At
# EF_CLIP = None          # e.g., (0.0, 1.3) to clip EF; or None to skip
EF_CLIP = (0.0, 1.8)

# -----------------------------
# 1) Load + outlier filtering
# -----------------------------
df = pd.read_csv(merged_csv)

for col in ["Rn_inst_avg","G_inst_avg","LE_inst_avg",
            "NETRAD","G_F_MDS","LE_F_MDS","H_F_MDS","EF_inst_avg"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

mask_valid = df[["NETRAD","G_F_MDS","LE_F_MDS"]].notna().all(axis=1)
mask_valid &= df["Rn_inst_avg"] >= 0
mask_valid &= (df["LE_inst_avg"] <= 800) & (df["LE_inst_avg"] >= -400)

before, df = len(df), df[mask_valid].copy()
print(f"✅ Filtering complete: removed {before - len(df)} rows, kept {len(df)}")

# -----------------------------
# 2) Tower EF via Bowen closure
#     beta_t = H/LE, EF_t = 1/(1+beta_t)
# -----------------------------
At = (df["NETRAD"] - df["G_F_MDS"]).where(lambda s: (s > A_MIN) & np.isfinite(s), np.nan)
beta_t = (df["H_F_MDS"] / df["LE_F_MDS"]).where(
    np.isfinite(df["H_F_MDS"]) & np.isfinite(df["LE_F_MDS"]), np.nan
)
EF_t = 1.0 / (1.0 + beta_t)
fallback_mask = ~np.isfinite(EF_t)
EF_t.loc[fallback_mask] = (df["LE_F_MDS"] / At).loc[fallback_mask]

if EF_CLIP is not None:
    lo, hi = EF_CLIP
    EF_t = EF_t.clip(lo, hi)

# -----------------------------
# 3) Model EF (no model closure)
# -----------------------------
if "EF_inst_avg" in df.columns and df["EF_inst_avg"].notna().any():
    EF_m = df["EF_inst_avg"].copy()
else:
    EF_m = df["LE_inst_avg"] / (df["Rn_inst_avg"] - df["G_inst_avg"])
    print("⚠️ EF_inst_avg missing; fell back to EF_m = LE_m / (Rn_m - G_m).")

if EF_CLIP is not None:
    EF_m = EF_m.clip(lo, hi)

# -----------------------------
# 4) Helpers
# -----------------------------
def stats(y_true, y_pred):
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() < 3:
        return np.nan, np.nan, np.nan
    err = y_pred[m] - y_true[m]
    rmse = float(np.sqrt(np.mean(err**2)))
    bias = float(np.mean(err))
    r2   = float(np.corrcoef(y_true[m], y_pred[m])[0,1]**2)
    return rmse, r2, bias

def fit_eval_group(g):
    efm = g["EF_m"]; eft = g["EF_t"]
    m   = np.isfinite(efm) & np.isfinite(eft)

    if m.sum() >= 3:
        reg = LinearRegression()
        reg.fit(efm[m].to_numpy().reshape(-1,1), eft[m].to_numpy())
        alpha, beta = float(reg.coef_[0]), float(reg.intercept_)
    else:
        alpha, beta = 1.0, 0.0

    ef_corr = alpha * efm + beta
    if EF_CLIP is not None:
        ef_corr = ef_corr.clip(lo, hi)

    ratio = np.where(np.isfinite(efm) & (efm != 0), ef_corr / efm, np.nan)
    le_corr = g["LE_inst_avg"] * ratio

    y_t = g["LE_F_MDS"]
    rmse_b, r2_b, bias_b = stats(y_t, g["LE_inst_avg"])
    rmse_a, r2_a, bias_a = stats(y_t, le_corr)

    out = {
        "alpha": alpha, "beta": beta, "N_pairs": int(m.sum()),
        "RMSE_before": rmse_b, "R2_before": r2_b, "Bias_before": bias_b,
        "RMSE_after": rmse_a,  "R2_after": r2_a,  "Bias_after": bias_a,
    }
    return out, ef_corr, le_corr

# Working frame
need_cols = ["site","scene","date","LE_inst_avg","LE_F_MDS",
             "Rn_inst_avg","G_inst_avg","NETRAD","G_F_MDS","H_F_MDS"]
for c in need_cols:
    if c not in df.columns:
        df[c] = np.nan

W = df.copy()
W["EF_m"] = EF_m
W["EF_t"] = EF_t

if "site" not in W.columns:
    raise ValueError("merged_csv must contain a 'site' column.")

# -----------------------------
# 5) Per-site results
# -----------------------------
site_rows   = []
per_site_ab = {}
W["EF_corr_site"] = np.nan
W["LE_corr_site"] = np.nan

for site, g in W.groupby("site"):
    res, ef_corr, le_corr = fit_eval_group(g)
    site_rows.append({"site": site, **res})
    per_site_ab[site] = {"alpha": res["alpha"], "beta": res["beta"], "N_pairs": res["N_pairs"]}
    W.loc[g.index, "EF_corr_site"] = ef_corr
    W.loc[g.index, "LE_corr_site"] = le_corr

# -----------------------------
# 6) ### US-R* COMBINED (single stats for the three US-R* sites)
#     Fit ONE α,β on all rows whose site starts with "US-R",
#     compute ONE set of metrics, and store separate corrected columns.
# -----------------------------
is_usr = W["site"].astype(str).str.startswith("US-R", na=False)
W["EF_corr_USR"] = np.nan
W["LE_corr_USR"] = np.nan

if is_usr.any():
    res_usr, ef_corr_usr, le_corr_usr = fit_eval_group(W[is_usr])
    site_rows.append({"site": "US-R* (combined)", **res_usr})
    per_site_ab["US-R* (combined)"] = {
        "alpha": res_usr["alpha"], "beta": res_usr["beta"], "N_pairs": res_usr["N_pairs"]
    }
    # write combined-correction back only to US-R* rows
    W.loc[is_usr, "EF_corr_USR"] = ef_corr_usr
    W.loc[is_usr, "LE_corr_USR"] = le_corr_usr
    print("✅ US-R* combined metrics computed")
else:
    print("ℹ️ No US-R* rows found; skipping combined metrics.")

# -----------------------------
# 7) Save params + metrics
# -----------------------------
with open(site_params_json, "w") as f:
    json.dump(per_site_ab, f, indent=2)

metrics_df = pd.DataFrame(site_rows).sort_values("site")
metrics_df.to_csv(metrics_csv, index=False)
print("✅ Saved per-site and US-R* combined params/metrics")

# -----------------------------
# 8) Detailed Excel + corrected CSV
# -----------------------------
pairs_cols = ["site","scene","date","EF_m","EF_t",
              "EF_corr_site","LE_inst_avg","LE_F_MDS","LE_corr_site",
              "EF_corr_USR","LE_corr_USR"]
pairs = W[pairs_cols].copy()
pairs["date"] = pd.to_datetime(pairs["date"], errors="coerce").dt.date

with pd.ExcelWriter(detailed_xlsx, engine="xlsxwriter") as xw:
    metrics_df.to_excel(xw, sheet_name="metrics", index=False)
    pairs.to_excel(xw, sheet_name="pairs", index=False)

W.to_csv(corrected_csv, index=False, float_format="%.3f")
print(f"✅ Outputs:\n- Params JSON: {site_params_json}\n- Metrics CSV: {metrics_csv}\n- Detailed XLSX: {detailed_xlsx}\n- Corrected CSV: {corrected_csv}")
