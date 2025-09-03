import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

# -----------------------------
# Inputs
# -----------------------------
merged_csv = r"merged_calibration.csv"   # path to merged CSV
out_dir = r"./calibration_plots"         # folder to save plots

# -----------------------------
# Load data
# -----------------------------
df_calib = pd.read_csv(merged_csv)

# Ensure output dir exists
os.makedirs(out_dir, exist_ok=True)

# -----------------------------
# Scatter plot helper with stats
# -----------------------------
def scatter_plot(x, y, xlabel, ylabel, title, fname):
    xvals = df_calib[x].values
    yvals = df_calib[y].values
    
    # Drop NaNs
    mask = ~np.isnan(xvals) & ~np.isnan(yvals)
    xvals, yvals = xvals[mask], yvals[mask]
    
    # Compute statistics
    r2 = r2_score(yvals, xvals)
    rmse = np.sqrt(mean_squared_error(yvals, xvals))
    bias = np.mean(xvals - yvals)
    
    # Plot
    plt.figure(figsize=(6,6))
    plt.scatter(xvals, yvals, c="blue", alpha=0.7, edgecolor="k")
    
    # 1:1 line
    min_val = min(xvals.min(), yvals.min())
    max_val = max(xvals.max(), yvals.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="1:1 Line")
    
    # Labels & Title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
    # Add text box with metrics
    textstr = f"$R^2$ = {r2:.3f}\nRMSE = {rmse:.3f}\nBias = {bias:.3f}"
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white", alpha=0.7))
    
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname), dpi=300)
    plt.close()
    
    # Print metrics to console too
    print(f"{title}: R²={r2:.3f}, RMSE={rmse:.3f}, Bias={bias:.3f}")

# -----------------------------
# Generate plots
# -----------------------------
scatter_plot("Rn_inst_avg", "NETRAD", 
             "SEBAL Rn (3×3 avg, W/m²)", "Flux Tower Rn (W/m²)", 
             "Net Radiation (Rn): SEBAL avg vs Flux", "Rn_comparison_avg.png")

scatter_plot("G_inst_avg", "G_F_MDS", 
             "SEBAL G (3×3 avg, W/m²)", "Flux Tower G (W/m²)", 
             "Soil Heat Flux (G): SEBAL avg vs Flux", "G_comparison_avg.png")

scatter_plot("LE_inst_avg", "LE_F_MDS", 
             "SEBAL LE (3×3 avg, W/m²)", "Flux Tower LE (W/m²)", 
             "Latent Heat (LE): SEBAL avg vs Flux", "LE_comparison_avg.png")

scatter_plot("EF_inst_avg", "EF", 
             "SEBAL EF (3×3 avg)", "Flux Tower EF", 
             "Evaporative Fraction (EF): SEBAL avg vs Flux", "EF_comparison_avg.png")

# scatter_plot("Rn_inst_pix", "NETRAD", 
#              "SEBAL Rn (W/m²)", "Flux Tower Rn (W/m²)", 
#              "Net Radiation (Rn): SEBAL vs Flux", "Rn_comparison.png")

# scatter_plot("G_inst_pix", "G_F_MDS", 
#              "SEBAL G (W/m²)", "Flux Tower G (W/m²)", 
#              "Soil Heat Flux (G): SEBAL vs Flux", "G_comparison.png")

# scatter_plot("LE_inst_pix", "LE_F_MDS", 
#              "SEBAL LE (W/m²)", "Flux Tower LE (W/m²)", 
#              "Latent Heat (LE): SEBAL vs Flux", "LE_comparison.png")

# scatter_plot("EF_inst_pix", "EF", 
#              "SEBAL EF", "Flux Tower EF", 
#              "Evaporative Fraction (EF): SEBAL vs Flux", "EF_comparison.png")

print(f"✅ Plots + metrics saved in: {out_dir}")
