"""Spatial validation figures from the finalized row-specific workbooks.

Metrics are never recomputed after the correlation count filter. Display jitter
does not alter coordinates used by validation, folds or spatial diagnostics.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def load_series(validation_root, rows, temporal_win=0):
    frames = []
    for row in rows:
        path = Path(validation_root) / f'scaled/results/validations_{row}_tw_{temporal_win}.xlsx'
        frame = pd.read_excel(path, sheet_name='MetaData')
        frames.append(frame.assign(row=str(row)))
    return pd.concat(frames, ignore_index=True)


def eligible_correlations(series, min_pairs=6):
    return series.loc[series.overlaps.ge(min_pairs)].dropna(subset=['p_rho', 's_rho'])


def run(validation_root, output, rows, temporal_win=0, min_pairs=6,
        jitter=.05, seed=42, natural_earth=True):
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    series = load_series(validation_root, rows, temporal_win)
    specs = {
        'bias': ('Figure_10a', 'Bias [m³/m³]', 'RdYlGn', -.075, .025),
        'mse': ('Figure_10b', 'Mean squared error [m³/m³]²', 'YlGn_r', 0, .01),
        'ubrmsd': ('Figure_11', 'UbRMSD [m³/m³]', 'YlGn_r', 0, .04),
        'p_rho': ('Figure_12a', "Pearson's r", 'RdYlGn', -.4, 1),
        's_rho': ('Figure_12b', "Spearman's rho", 'RdYlGn', -.4, 1),
        'overlaps': ('matched_pair_map', 'Number of observations', 'YlGn', 0, 26),
    }
    for metric, (name, label, cmap, vmin, vmax) in specs.items():
        data = eligible_correlations(series, min_pairs) if metric in ('p_rho', 's_rho') else series
        data = data.dropna(subset=['latitude', 'longitude', metric])
        if data.empty:
            raise ValueError(f'No valid records for {metric}; correlation minimum is {min_pairs}.')
        # Reinitialize the original random sequence for every map.
        rng = np.random.RandomState(seed)
        lon = data.longitude.to_numpy() + rng.uniform(-jitter, jitter, len(data))
        lat = data.latitude.to_numpy() + rng.uniform(-jitter, jitter, len(data))
        extent = [data.longitude.min()-.5, data.longitude.max()+.5,
                  data.latitude.min()-.5, data.latitude.max()+.5]
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        if natural_earth:
            for feature in [cfeature.COASTLINE, cfeature.BORDERS, cfeature.STATES, cfeature.RIVERS]:
                ax.add_feature(feature)
            ax.add_feature(cfeature.NaturalEarthFeature('cultural', 'admin_1_states_provinces_lines',
                                                       '50m', facecolor='none'), edgecolor='gray')
        grid = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=.5, linestyle='--')
        grid.top_labels = grid.right_labels = False
        points = ax.scatter(lon, lat, c=data[metric], cmap=cmap, vmin=vmin, vmax=vmax,
                            edgecolor='k', s=35, transform=ccrs.PlateCarree())
        fig.colorbar(points, ax=ax).set_label(label, fontsize=18)
        fig.savefig(output / f'{name}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        data.to_csv(output / f'{name}_records.csv', index=False)
    corr = eligible_correlations(series, min_pairs)
    summary = dict(series=len(series), matched_pairs=int(series.overlaps.sum()),
                   coordinate_locations=len(series[['latitude', 'longitude']].drop_duplicates()),
                   correlation_min_pairs=min_pairs, eligible_correlation_series=len(corr),
                   spearman_exceeds_pearson=int(corr.s_rho.gt(corr.p_rho).sum()),
                   percent_spearman_exceeds_pearson=100*float(corr.s_rho.gt(corr.p_rho).mean()),
                   jitter_degrees=jitter, map_seed=seed, counting_unit='GPI–satellite-row series')
    (output / 'map_summary.json').write_text(json.dumps(summary, indent=2))
    return output
