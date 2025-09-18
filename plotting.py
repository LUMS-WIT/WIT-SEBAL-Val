"""
Created on Jan 2025

validation of sebal soil moisture estimates
using WITSMS Network

@author: hamza rafique
"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as cl
import pandas as pd
import matplotlib.patches as mpatches
from config import OUTLIER_THRESHOLD

# setup the paths for generated results files
file_path_149039 = fr'.\validations\results\validations_149039_tw_0.xlsx'
file_path_150039 = fr'.\validations\results\validations_150039_tw_0.xlsx'

# Define the paramter for plotting on a raster
param = 'overlaps'  # 'overlaps', 'bias', 'mse', 'ubrmsd', 'p_rho', 's_rho'
display_extent = False  # True to display the extent box on the full map


# Function to define custom color maps
def get_status_colors():
    cmap = plt.get_cmap('Set3', 14)
    colors = [cmap(i) for i in range(cmap.N)]
    colors.insert(0, (0, 0.66666667, 0.89019608, 1.0))  # Special color
    colors.insert(0, (0.45882353, 0.08235294, 0.11764706, 1.0))
    return cl.ListedColormap(colors)

# Define color maps as per your mapping
_cclasses =  {
    'div_better': plt.get_cmap(
        'RdYlBu'
    ),  # diverging: 1 good, 0 special, -1 bad (pearson's R, spearman's rho')
    'div_worse': plt.get_cmap(
        'RdYlBu_r'
    ),  # diverging: 1 bad, 0 special, -1 good (difference of bias)
    'div_neutr':
    plt.get_cmap('RdYlGn'),  # diverging: zero good, +/- neutral: (bias)
    'seq_worse': plt.get_cmap(
        'YlGn_r'
    ),  # sequential: increasing value bad (p_R, p_rho, rmsd, ubRMSD, RSS)
    'seq_better': plt.get_cmap(
        'YlGn'),  # sequential: increasing value good (n_obs, STDerr)
    'qua_neutr':
    get_status_colors(),  # qualitative category with 2 forced colors
    'custom': plt.get_cmap(
        'viridis')
}

_colormaps = {
    'R': _cclasses['div_better'],
    'p_R': _cclasses['seq_worse'],
    'rho': _cclasses['div_better'],
    #'p_rho': _cclasses['seq_worse'],
    'p_rho': _cclasses['div_neutr'],
    's_rho': _cclasses['div_neutr'],
    'rmsd': _cclasses['seq_worse'],
    'bias': _cclasses['div_neutr'],
    # 'n_obs': _cclasses['seq_better'],
    'overlaps': _cclasses['seq_better'],  # orginal
    # 'ubrmsd': _cclasses['seq_worse'], # orginal
    #'overlaps': _cclasses['div_neutr'],
    'ubrmsd': _cclasses['seq_worse'],
    'mse': _cclasses['seq_worse'],
    'mse_corr': _cclasses['seq_worse'],
    'mse_bias': _cclasses['seq_worse'],
    'mse_var': _cclasses['seq_worse'],
    'RSS': _cclasses['seq_worse'],
    'tau': _cclasses['div_better'],
    'p_tau': _cclasses['seq_worse'],
    'snr': _cclasses['div_better'],
    'err_std': _cclasses['seq_worse'],
    'beta': _cclasses['div_neutr'],
    'status': _cclasses['qua_neutr'],
}


# Read the data
df1 = pd.read_excel(file_path_149039)
df2 = pd.read_excel(file_path_150039)

# Filter rows where metric data is available
metrics_columns = ['overlaps', 'bias', 'mse', 'ubrmsd', 'p_rho', 's_rho']
metric_names = {
    'overlaps': '# of Observations',
    'bias': 'Bias in m³/m³',
    'mse': 'Mean squared error in m³/m³',
    'ubrmsd': 'Unbiased root mean square deviation in m³/m³',
    'p_rho': "Pearson's r",
    's_rho': "Spearman's rho",
}
df1_filtered = df1.dropna(subset=metrics_columns)
df2_filtered = df2.dropna(subset=metrics_columns)

# Apply additional filters
df1_filtered_ = df1_filtered[
    (df1_filtered['p_rho'] >= OUTLIER_THRESHOLD) & (df1_filtered['s_rho'] >= OUTLIER_THRESHOLD) |
    ((df1_filtered['p_rho'] < OUTLIER_THRESHOLD) | (df1_filtered['s_rho'] < OUTLIER_THRESHOLD)) & (df1_filtered['overlaps'] > 10)
]

df2_filtered_ = df2_filtered[
    (df2_filtered['p_rho'] >= OUTLIER_THRESHOLD) & (df2_filtered['s_rho'] >= OUTLIER_THRESHOLD) |
    ((df2_filtered['p_rho'] < OUTLIER_THRESHOLD) | (df2_filtered['s_rho'] < OUTLIER_THRESHOLD)) & (df2_filtered['overlaps'] > 10)
]

# Combine and select required columns
df_final = pd.concat([df1_filtered_, df2_filtered_], ignore_index=True)
df_final = df_final[['latitude', 'longitude'] + metrics_columns]

print(f'Total Number of sites: {df_final["overlaps"].count()}')
print(f'Total Number of data points: {df_final["overlaps"].sum()}')

# Extract latitude, longitude, and parameter values
lats = df_final['latitude'].tolist()
lons = df_final['longitude'].tolist()
values = df_final[param].tolist()

# Calculate dynamic extent based on data points
margin = 0.5
extent = [
    min(lons) - margin, max(lons) + margin,
    min(lats) - margin, max(lats) + margin
]

if display_extent:

    # Plot 1: Entire map of Pakistan with the extent box
    fig1, ax1 = plt.subplots(figsize=(12, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    ax1.set_extent([60, 80, 20, 40], crs=ccrs.PlateCarree())  # Full map of Pakistan

    # Add features to the map
    ax1.add_feature(cfeature.COASTLINE)
    ax1.add_feature(cfeature.BORDERS, linestyle=':')
    ax1.add_feature(cfeature.STATES, linestyle=':')
    ax1.add_feature(cfeature.RIVERS)
    ax1.add_feature(cfeature.LAND, edgecolor='black', alpha=0.2)

    # Draw the extent box
    rect = mpatches.Rectangle(
        (extent[0], extent[2]),  # Bottom-left corner
        extent[1] - extent[0],  # Width
        extent[3] - extent[2],  # Height
        linewidth=2, edgecolor='red', linestyle='--', facecolor='none', transform=ccrs.PlateCarree()
    )
    ax1.add_patch(rect)

    # Add gridlines
    gl = ax1.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 12, 'color': 'gray'}
    gl.ylabel_style = {'size': 12, 'color': 'gray'}

    # Title for full map
    # ax1.set_title('Full Map of Pakistan with Highlighted Extent', fontsize=14)

# Plot 2: Zoomed-in view
fig2, ax2 = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
ax2.set_extent(extent, crs=ccrs.PlateCarree())  # Zoomed-in extent

# Add features to the zoomed-in map
states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none'
)

ax2.add_feature(cfeature.COASTLINE)
ax2.add_feature(cfeature.BORDERS)
ax2.add_feature(cfeature.STATES, linestyle=':')
ax2.add_feature(cfeature.RIVERS)
ax2.add_feature(states_provinces, edgecolor='gray')

# Add gridlines
gl2 = ax2.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl2.top_labels = False
gl2.right_labels = False
gl2.xlabel_style = {'size': 12, 'color': 'gray'}
gl2.ylabel_style = {'size': 12, 'color': 'gray'}

# Plot the scatter points
if param == 'overlaps':
    sc = ax2.scatter(lons, lats, c=values, cmap=_colormaps[param], marker='o', edgecolor='k', s=60, transform=ccrs.PlateCarree(), vmin=0, vmax=26)
elif param == 'bias' or param == 'mse':
    sc = ax2.scatter(lons, lats, c=values, cmap=_colormaps[param], marker='o', edgecolor='k', s=60, transform=ccrs.PlateCarree(), vmin=-0.025, vmax=0.025)
elif param == 'ubrmsd':
    sc = ax2.scatter(lons, lats, c=values, cmap=_colormaps[param], marker='o', edgecolor='k', s=60, transform=ccrs.PlateCarree(), vmin=0, vmax=0.04)
else:
    sc = ax2.scatter(lons, lats, c=values, cmap=_colormaps[param], marker='o', edgecolor='k', s=60, transform=ccrs.PlateCarree())


# Add a color bar
cbar = plt.colorbar(sc, ax=ax2, orientation='vertical')
cbar.set_label(metric_names.get(param, param), fontsize = 18)

# Add a title
# ax2.set_title(f'Scatter Plot of {param} metrics', fontsize=14)

# Show the plots
plt.show()