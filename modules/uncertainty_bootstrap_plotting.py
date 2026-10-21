"""PNG-only publication plots. Edit PLOT_SETTINGS below for all figure styling."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D

# All label nouns, legend fonts and symbol sizes are centralized here.
# Matplotlib scatter sizes are marker AREAS in points squared.
PLOT_SETTINGS = {
    'font_family': 'DejaVu Serif', 'font_size': 9,
    'axis_label_size': 9, 'tick_label_size': 8, 'panel_title_size': 10,
    'legend_size': 9, 'annotation_size': 9,
    'parameter_legend_size': 7, 'spread_legend_size': 10,
    'relative_legend_size': 10, 'temporal_legend_size': 10,
    # Per-axis overrides; omitted labels use axis_label_size above.
    'label_sizes': {},  # Example: {'relative_label': 10, 'slope_label': 8}
    'bootstrap_marker_size': 10, 'selected_marker_size': 10,
    'central_marker_size': 10, 'soil_moisture_marker_size': 6,
    'central_marker': 'o', 'legend_marker_scale': 1.0,
    # Outline the three LEGEND symbols only, not the plotted points.
    'bootstrap_legend_edge_color': '#666666', 'bootstrap_legend_edge_width': .7,
    'selected_legend_edge_color': '#666666', 'selected_legend_edge_width': .7,
    'mean_legend_edge_color': '#666666', 'mean_legend_edge_width': .7,
    'mean_line_width': 1.05, 'ci_alpha': .4, 'dpi': 300,
    'figure5_size': (8, 3.65), 'figure6_size': (9, 3.6),
    'histogram_bins': 32, 'marginal_bank_bins': 28, 'marginal_subset_bins': 18,
    'bootstrap_label': 'Bootstrap Samples', 'selected_label': 'Propagated Samples',
    'mean_label': 'Mean', 'ci_label': '95% CI',
    'median_label': 'Median', 'p95_label': '95th percentile', 'correlation_label': r'$r$',
    'slope_label': r'Calibration slope, $\alpha$',
    'intercept_label': r'Calibration intercept, $\beta$',
    'uncertainty_label': r'Uncertainty $\sigma$ (m$^3$ m$^{-3}$)',
    'density_label': 'Probability Density',
    'soil_moisture_label': r'SEBAL Soil Moisture (m$^3$ m$^{-3}$)',
    'relative_label': 'Relative Uncertainty (%)',
    'date_label': 'Scene acquisition date',
    'mean_moisture_label': r'Mean soil moisture (m$^3$ m$^{-3}$)',
    # 'parameter_title': '(a) EF-calibration parameter distribution',
    # 'spread_title': '(b) Propagated soil-moisture uncertainty',
    # 'relative_title': '(a) Relative calibration uncertainty',
    # 'temporal_title': '(b) Combined acquisition-date series',
    'parameter_title': '(a)',
    'spread_title': '(b)',
    'relative_title': '(a)',
    'temporal_title': '(b)',
    'blue': '#38677D', 'orange': '#C06B2A', 'red': '#B53932', 'grey': '#B4B8BB',
    # Optional complete per-pixel evidence; these are PNG pages, never PDFs.
    'save_site_grid': False,
}


def make_publication_plots(bank, selected, pixels, means, central, output, settings=None):
    settings = {**PLOT_SETTINGS, **(settings or {})}
    s = settings
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    style = {'font.family': s['font_family'], 'font.size': s['font_size'],
             'axes.labelsize': s['axis_label_size'], 'axes.titlesize': s['panel_title_size'],
             'legend.fontsize': s['legend_size'], 'xtick.labelsize': s['tick_label_size'],
             'ytick.labelsize': s['tick_label_size'], 'axes.spines.top': False,
             'axes.spines.right': False}
    blue, orange, red, grey = [s[k] for k in ['blue', 'orange', 'red', 'grey']]
    files = []
    def save(fig, name):
        path = output/f'{name}.png'
        fig.savefig(path, dpi=s['dpi'])
        plt.close(fig)
        files.append(path)
    def labels(ax, x, y):
        ax.set_xlabel(s[x], fontsize=s['label_sizes'].get(x,s['axis_label_size']))
        ax.set_ylabel(s[y], fontsize=s['label_sizes'].get(y,s['axis_label_size']))
    with plt.rc_context(style):
        fig = plt.figure(figsize=s['figure5_size'], layout='constrained')
        grid = fig.add_gridspec(1, 2, width_ratios=[1, 1.05])
        left = grid[0].subgridspec(3, 3, height_ratios=[.65, 1, 1],
                                 width_ratios=[1, 1, .65], hspace=.04, wspace=.04)
        top = fig.add_subplot(left[0, :2])
        joint = fig.add_subplot(left[1:, :2], sharex=top)
        side = fig.add_subplot(left[1:, 2], sharey=joint)
        for ax, variable, orientation in [(top, 'alpha', 'vertical'), (side, 'beta', 'horizontal')]:
            ax.hist(bank[variable], bins=s['marginal_bank_bins'], density=True,
                    color=grey, alpha=.75, orientation=orientation)
            ax.hist(selected[variable], bins=s['marginal_subset_bins'], density=True,
                    histtype='step', color=orange, lw=1, orientation=orientation)
        top.set_title(s['parameter_title'], loc='left', fontsize=s['axis_label_size'])
        top.tick_params(axis='x', labelbottom=False, bottom=False)
        top.set_yticks([])
        side.tick_params(axis='y', labelleft=False, left=False)
        side.set_xticks([])
        joint.scatter(bank.alpha, bank.beta, s=s['bootstrap_marker_size'], color=grey,
                      alpha=.25, edgecolors='none', label=s['bootstrap_label'])
        joint.scatter(selected.alpha, selected.beta, s=s['selected_marker_size'], color=orange,
                      alpha=.55, edgecolors='none', label=s['selected_label'])
        joint.scatter([central['alpha']], [central['beta']], marker=s['central_marker'],
                      s=s['central_marker_size'], color=red, edgecolors='white',
                      linewidths=.6, label=s['mean_label'], zorder=5)
        joint.text(.04, .96, f"{s['correlation_label']} = {bank.alpha.corr(bank.beta):.3f}",
                   transform=joint.transAxes, va='top', fontsize=s['annotation_size'])
        labels(joint,'slope_label','intercept_label')
        joint.xaxis.set_major_locator(MaxNLocator(4))
        joint.yaxis.set_major_locator(MaxNLocator(4))
        handles, legend_labels = joint.get_legend_handles_labels()
        for index, color, size_key, edge_prefix, marker in [
            (0, grey, 'bootstrap_marker_size', 'bootstrap', 'o'),
            (1, orange, 'selected_marker_size', 'selected', 'o'),
            (2, red, 'central_marker_size', 'mean', s['central_marker']),
        ]:
            handles[index] = Line2D([], [], linestyle='None', marker=marker,
                                   markersize=np.sqrt(s[size_key]), markerfacecolor=color,
                                   markeredgecolor=s[f'{edge_prefix}_legend_edge_color'],
                                   markeredgewidth=s[f'{edge_prefix}_legend_edge_width'])
        joint.legend(handles, legend_labels, frameon=False, loc='lower left',
                     markerscale=s['legend_marker_scale'], fontsize=s['parameter_legend_size'])
        ax = fig.add_subplot(grid[1])
        upper = max(.005, float(np.ceil(pixels.ensemble_sd.max()/.005)*.005))
        ax.hist(pixels.ensemble_sd, bins=np.linspace(0, upper, s['histogram_bins']),
                density=True, color=blue, edgecolor='white', linewidth=.5)
        ax.axvline(pixels.ensemble_sd.median(), color=orange, lw=1.3,
                   label=f"{s['median_label']} = {pixels.ensemble_sd.median():.4f}")
        ax.axvline(pixels.ensemble_sd.quantile(.95), color=red, ls='--', lw=1.3,
                   label=f"{s['p95_label']} = {pixels.ensemble_sd.quantile(.95):.4f}")
        ax.set(xlim=(0, upper), title=s['spread_title'])
        labels(ax,'uncertainty_label','density_label')
        ax.legend(frameon=False, loc='upper right',fontsize=s['spread_legend_size'])
        ax.grid(axis='y', alpha=.15)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        save(fig, 'Figure_5')

        fig = plt.figure(figsize=s['figure6_size'], layout='constrained')
        grid = fig.add_gridspec(1, 2, width_ratios=[1, 1.65])
        ax = fig.add_subplot(grid[0])
        relative = 100*pixels.ensemble_sd/pixels.central_rzsm.where(pixels.central_rzsm.gt(0))
        ax.scatter(pixels.central_rzsm, relative, s=s['soil_moisture_marker_size'],
                   color=blue, alpha=.35, edgecolors='none')
        ax.axhline(relative.median(), color=red, ls=':', lw=1.5,
                   label=f"{s['median_label']} = {relative.median():.2f}%")
        ax.set(ylim=(0, max(float(relative.max()*1.07), 1)), title=s['relative_title'])
        labels(ax,'soil_moisture_label','relative_label')
        ax.legend(frameon=False, loc='upper right',fontsize=s['relative_legend_size'])
        ax.grid(axis='y', alpha=.15)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax = fig.add_subplot(grid[1])
        dates = pd.to_datetime(means.index)
        band = ax.fill_between(dates, means.p025, means.p975, color=blue,
                               alpha=s['ci_alpha'], label=s['ci_label'])
        line, = ax.plot(dates, means.original_mean, color=orange, lw=s['mean_line_width'],
                        label=s['mean_label'])
        ax.set(title=s['temporal_title'])
        labels(ax,'date_label','mean_moisture_label')
        ax.legend([line, band], [s['mean_label'], s['ci_label']], frameon=False, loc='upper left',fontsize=s['temporal_legend_size'])
        ax.grid(axis='y', alpha=.15)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        save(fig, 'Figure_6')

        if s['save_site_grid']:
            for swath, points in pixels.groupby('swath'):
                dates = sorted(means.index[means.swaths.str.split(';').apply(lambda a: str(swath) in a)])
                order = points.groupby('gpi').latitude.median().sort_values(ascending=False).index
                for start in range(0, len(order), 25):
                    fig, axes = plt.subplots(5, 5, figsize=(11.5, 9), sharex=True, sharey=True)
                    for ax, gpi in zip(axes.flat, order[start:start+25]):
                        d = points[points.gpi.eq(gpi)].set_index('date').reindex(dates)
                        dt = pd.to_datetime(dates)
                        ax.fill_between(dt, d.p025, d.p975, color=blue, alpha=.25)
                        ax.plot(dt, d.central_rzsm, color=orange, lw=.7)
                        ax.set_title(str(gpi))
                        ax.set_ylim(.04, np.ceil(pixels.p975.max()/.05)*.05)
                        ax.xaxis.set_major_locator(mdates.YearLocator())
                        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                    for ax in axes.flat[len(order[start:start+25]):]:
                        ax.set_visible(False)
                    fig.suptitle(f'Pointwise 95% calibration intervals: swath {swath}')
                    fig.tight_layout()
                    save(fig, f'pixel_intervals_{swath}_{start//25+1}')
    return files
