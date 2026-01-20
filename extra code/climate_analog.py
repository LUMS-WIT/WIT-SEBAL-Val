#####################################################################
####################################################################



import pandas as pd
import numpy as np
from scipy.stats import chi2

# ------------------------------------------------------------
# 1. READ DATA
# ------------------------------------------------------------
df = pd.read_csv("BariDoab_vs_FluxSites_Climate_v6.csv")

# Ensure consistent ordering
# df = df.reset_index(drop=True)
df = df.apply(pd.to_numeric, errors="ignore")

# ------------------------------------------------------------
# 2. DEFINE VARIABLES
# ------------------------------------------------------------
# Static climatological descriptors (mean-state)
vars_static = [
    "AI",
    # "VPD_kPa",
    # "Temp_WarmestQ_C",
    # "Precip_WarmestQ_mm",
    # "Temp_Annual_Range_C",
    # "Precip_Seasonality_pct",
    # "MAP_mm",
    # "MAT_C"

]

# vars_climate = ["AI", "VPD_kPa", "Wind_ms", "Precip_Seasonality_pct", "Precip_WarmestQ_mm", "Temp_Annual_Range_C", "Temp_WarmestQ_C"]
# vars_climate = ["AI", "VPD_kPa", "Wind_ms","Precip_WarmestQ_mm", "Temp_Annual_Range_C"]

# Dynamic process-relevant variables (have interannual variability)
vars_dynamic = [
    "Precip_WarmestQ_mm",
    "Temp_Annual_Range_C",    
    "VPD_kPa",
    "Wind_ms"
]

vars_all = vars_static + vars_dynamic

# --------------------------------------------------
# 3. Extract reference climate (Bari Doab)
# --------------------------------------------------
ref_row = df.loc[df["Site"] == "Bari_Doab"]

if ref_row.empty:
    raise ValueError("Bari_Doab not found in Site column")

x_ref = ref_row[vars_all].to_numpy(dtype=float).squeeze()

# ------------------------------------------------------------
# 4. CONSTRUCT σ (INTERANNUAL VARIABILITY)
# ------------------------------------------------------------
"""
Mahony et al. (2017):
σ must represent interannual climate variability (ICV),
NOT spatial variability and NOT target-climate variability.
"""

# 4.1 σ for dynamic variables (from ERA5 interannual variability)
#     Exclude Bari Doab itself
sigma_dynamic = (
    df.loc[df["Site"] == "Bari_Doab",
           ["Precip_WarmestQ_mm_sigma", "Temp_Annual_Range_C_sigma", "VPD_kPa_sigma", "Wind_ms_sigma"]]
    # .mean()
    .to_numpy(dtype=float)
)

# sigma_dynamic = (
#     df.loc[df["Site"] != "Bari_Doab",
#            ["Wind_ms_sigma"]]
#     .mean()
#     .to_numpy(dtype=float)
# )

# 4.2 σ for static variables (fallback: spatial SD across sites)
#     Used only because WorldClim has no time dimension
sigma_static = (
    df.loc[df["Site"] == "Bari_Doab", vars_static]
    # .std()
    .to_numpy(dtype=float)
)

# 4.3 Combine σ
sigma = np.concatenate([sigma_static[0], sigma_dynamic[0]])

# ------------------------------------------------------------
# 5. COMPUTE σ-DISSIMILARITY (MAHALANOBIS FORM)
# ------------------------------------------------------------
"""
Mahony framework equivalent:

D = sqrt( Σ ((x_i − x_ref_i) / σ_i)^2 )

This is a Mahalanobis distance with diagonal covariance
after scaling by interannual variability.
"""

weights = np.array([0.5, 0.20, 0.20, 0.025, 0.075])  # VPD and Wind
# weights = np.array([1,1,1,1,1])  # VPD and Wind

X = df[vars_all].to_numpy(dtype=float)
D = np.sqrt(np.sum( weights * ((X - x_ref) / sigma) ** 2, axis=1))
df["Sigma_Dissimilarity"] = D

# ------------------------------------------------------------
# 6. MAP DISTANCE → σ THRESHOLDS (CHI DISTRIBUTION)
# ------------------------------------------------------------
k = len(vars_all)

D_2sigma = np.sqrt(chi2.ppf(0.95, df=k))
D_4sigma = np.sqrt(chi2.ppf(0.9999, df=k))

df["Threshold_2sigma"] = D_2sigma
df["Threshold_4sigma"] = D_4sigma

def classify_sigma(D):
    if D <= D_2sigma:
        return "Acceptable Analog"
    elif D <= D_4sigma:
        return "Marginal Analog (2–4σ)"
    else:
        return "Novel Climate (>4σ)"

df["Climate_Analog_Class"] = df["Sigma_Dissimilarity"].apply(classify_sigma)

# ------------------------------------------------------------
# 7. OUTPUT
# ------------------------------------------------------------
print("\nSigma thresholds:")
print(f"2σ threshold: {D_2sigma:.2f}")
print(f"4σ threshold: {D_4sigma:.2f}")

print("\nClimate analog results:")
print(
    df[["Site", "Sigma_Dissimilarity", "Climate_Analog_Class"]]
    .sort_values("Sigma_Dissimilarity")
)

# Optional export

df.to_csv("BariDoab_vs_FluxSites_Climate_v6_results.csv", index=False)
