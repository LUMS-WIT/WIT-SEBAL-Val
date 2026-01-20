#####################################################################
####################################################################



import pandas as pd
import numpy as np
from scipy.stats import chi2

# ------------------------------------------------------------
# 1. READ DATA
# ------------------------------------------------------------
df = pd.read_csv("BariDoab_vs_FluxSites_Climate_v7.csv")

years = np.arange(1981, 2011)
vars_dyn = ["TempRange", "PrecipWQ", "VPD", "Wind"]

# def extract_ts(row):
#     return np.column_stack([
#         [row[f"{v}_{y}"] for y in years] for v in vars_dyn
#     ])

# --------------------------------------------------
# 2. Helper: extract annual time series robustly
# --------------------------------------------------
def extract_ts(row):
    ts = []
    for v in vars_dyn:
        # cols = [c for c in df.columns if c.endswith(f"{v}_{y}") for y in years]
        cols = [c for c in df.columns if any(c.endswith(f"{v}_{y}") for y in years)]
        vals = [row[c] for c in cols]
        ts.append(vals)
    return np.column_stack(ts)

# -------------------------
# Reference climate (Bari Doab)
# -------------------------
ref = df[df["Site"] == "Bari_Doab"].iloc[0]
X_ref = extract_ts(ref)

# Remove mean (interannual anomalies)
X_ref_anom = X_ref - X_ref.mean(axis=0)

# Covariance of interannual variability
Sigma = np.cov(X_ref_anom, rowvar=False)
Sigma_inv = np.linalg.inv(Sigma)

mu_ref = X_ref.mean(axis=0)

# --------------------------------------------------
# 4. AI spatial variance
# --------------------------------------------------
ai_vals = df["AI"].to_numpy(dtype=float)
mu_ai = ref["AI"]
sigma_ai = ai_vals.std(ddof=1)

# -------------------------
# Mahalanobis distance
# -------------------------
def mahalanobis(x, mu, S_inv):
    d = x - mu
    return np.sqrt(d @ S_inv @ d)

# df["Mahalanobis_D"] = [
#     mahalanobis(extract_ts(row).mean(axis=0), mu_ref, Sigma_inv)
#     for _, row in df.iterrows()
# ]

D_total = []

for _, row in df.iterrows():
    X_site = extract_ts(row).mean(axis=0)

    D_dyn = mahalanobis(X_site, mu_ref, Sigma_inv)
    D_ai  = ((row["AI"] - mu_ai) / sigma_ai) ** 2

    D_total.append(np.sqrt(D_dyn + D_ai))

df["Mahalanobis_D"] = D_total

# -------------------------
# σ thresholds (χ²)
# -------------------------
k = 5
D2 = np.sqrt(chi2.ppf(0.95, k))
D4 = np.sqrt(chi2.ppf(0.9999, k))

def classify(D):
    if D <= D2:
        return "Acceptable Analog"
    elif D <= D4:
        return "Marginal Analog (2–4σ)"
    else:
        return "Novel Climate (>4σ)"

df["Analog_Class"] = df["Mahalanobis_D"].apply(classify)

print("\nSigma thresholds:")
print(f"2σ threshold: {D2:.2f}")
print(f"4σ threshold: {D4:.2f}")

print(df[["Site", "Mahalanobis_D", "Analog_Class"]]
      .sort_values("Mahalanobis_D"))

# Optional export

# df.to_csv("BariDoab_vs_FluxSites_Climate_v7_results.csv", index=False)
