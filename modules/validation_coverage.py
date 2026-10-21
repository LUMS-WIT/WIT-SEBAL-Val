"""Audit final workbook counts and plot coverage at explicit counting levels."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config as cfg
ROOT = cfg.PROJECT_ROOT
OUT = Path(cfg.COVERAGE_OUTPUT)
SOURCE = Path(cfg.SCALING_OUTPUT_BASE) / 'scaled/results'
PREDICTIONS = Path(cfg.SCALING_OUTPUT_BASE) / 'heldout_predictions.csv'
ROWS = cfg.ROW_PATHS
LABELS = ['<5', '5–9', '>9']
COLORS = ['#B9BEC4', '#70A7C3', '#276384']


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(path.with_suffix('.' + extension), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def figures(data, folder, unit):
    """Data contains exactly one row per declared counting unit."""
    coverage = data.groupby('overlaps').size().rename('count').reset_index()
    coverage['eight_day_equivalent_days'] = coverage.overlaps * 8
    coverage.to_csv(folder / 'coverage_distribution.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.8), sharex=True)
    y = np.arange(len(coverage))
    maximum = coverage['count'].max()
    for ax, column, color, label in [
        (axes[0], 'overlaps', '#06AEB5', 'Number of overlapping pairs'),
        (axes[1], 'eight_day_equivalent_days', '#F17C70', 'Nominal 8-day-equivalent duration (days)')]:
        ax.barh(y, coverage['count'], height=.93, color=color, edgecolor='white', linewidth=.4)
        for yy, count in zip(y, coverage['count']):
            ax.text(count + maximum * .025, yy, str(count), va='center', fontsize=9)
        ax.set_yticks(y, coverage[column].astype(int).astype(str))
        ax.set_ylim(len(coverage) - .5, -.5)
        ax.set_xlim(0, maximum * 1.17)
        ax.set_xlabel(f'Number of {unit}')
        ax.set_ylabel(label)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.grid(axis='x', color='#E7EBEE', linewidth=.65)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(axis='y', length=0)
    fig.tight_layout(w_pad=3)
    save(fig, folder / 'horizontal_sampling_coverage')

    data = data.copy()
    data['group'] = pd.cut(data.overlaps, [-np.inf, 4, 9, np.inf], labels=LABELS)
    groups = data.groupby('group', observed=False)
    summary = groups.agg(count=('overlaps', 'size'), pairs=('overlaps', 'sum'),
                         median_spearman=('s_rho', 'median'))
    assert summary['count'].sum() == len(data) and summary.pairs.sum() == data.overlaps.sum()
    summary.to_csv(folder / 'three_group_summary.csv')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    ax = axes[0]
    bars = ax.bar(range(3), summary['count'], color=COLORS, width=.6)
    ax.set_ylim(0, summary['count'].max() * 1.28)
    for bar, count, pairs in zip(bars, summary['count'], summary.pairs):
        ax.text(bar.get_x() + bar.get_width() / 2, count + .6,
                f'{count} {unit}\n{pairs} pairs', ha='center', va='bottom', fontsize=10)
    ax.set_ylabel(f'Number of {unit}')
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    values = [groups.get_group(label).s_rho.dropna().to_numpy() for label in LABELS]
    ax = axes[1]
    boxes = ax.boxplot(values, positions=range(3), widths=.5, patch_artist=True,
                       showfliers=False, medianprops=dict(color='#111111', linewidth=1.7))
    for box, color in zip(boxes['boxes'], COLORS):
        box.set_facecolor(color)
        box.set_alpha(.55)
    rng = np.random.default_rng(17)
    for i, values_in_group in enumerate(values):
        ax.scatter(i + rng.uniform(-.14, .14, len(values_in_group)), values_in_group,
                   s=23, color='#263238', edgecolors='white', linewidths=.4, alpha=.75, zorder=3)
        ax.text(i, 1.08, f'n={len(values_in_group)}', ha='center', fontsize=10)
        ax.annotate(f'{np.median(values_in_group):.3f}', xy=(i + .25, np.median(values_in_group)),
                    xytext=(5, 0), textcoords='offset points', ha='left', va='center', fontsize=9,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.85, pad=1))
    ax.set_ylim(-1.08, 1.18)
    ax.set_xlim(-.55, 2.65)
    ax.set_yticks([-1, -.5, 0, .5, 1])
    ax.set_ylabel('Spearman ρ')
    ax.axhline(0, color='#666666', lw=.7)
    for ax in axes:
        ax.set_xticks(range(3), LABELS)
        ax.set_xlabel('Number of overlapping pairs')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
    fig.tight_layout(w_pad=3)
    save(fig, folder / 'coverage_spearman_three_groups')
    return summary


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    for row in ROWS:
        path = SOURCE / f'validations_{row}_tw_{cfg.TEMPORAL_WIN}.xlsx'
        frame = pd.read_excel(path, sheet_name='MetaData')
        summary = pd.read_excel(path, sheet_name='Summary')
        assert frame.gpi.is_unique
        assert frame.overlaps.sum() == summary.loc[summary.Metric.eq('Observations'), 'mean'].iloc[0]
        frames.append(frame.assign(row=row))
    series = pd.concat(frames, ignore_index=True)
    combined = pd.read_excel(SOURCE / f'validations_tw_{cfg.TEMPORAL_WIN}.xlsx', sheet_name='MetaData')
    # Final combined workbooks pool each GPI across swaths (66 GPI rows in
    # the paper); row-specific workbooks retain GPI–swath series (73 rows).
    assert combined.gpi.is_unique
    assert set(combined.gpi) == set(series.gpi)
    assert combined.overlaps.sum() == series.overlaps.sum()
    assert series.overlaps.sum() > 0
    predictions = pd.read_csv(PREDICTIONS, dtype={'row': str}, parse_dates=['date'])
    assert not predictions.duplicated(['gpi', 'row', 'date']).any()
    assert not predictions.duplicated(['gpi', 'date']).any()
    assert not predictions.duplicated(['latitude', 'longitude', 'date']).any()
    assert len(predictions) == series.overlaps.sum()
    # Verify underlying dates/values against every final workbook record before
    # using those predictions for physical-location correlation and date spans.
    for record in series.to_dict('records'):
        group = predictions[predictions.gpi.eq(record['gpi']) & predictions.row.eq(record['row'])]
        assert len(group) == record['overlaps']
        np.testing.assert_allclose(group[['latitude', 'longitude']],
                                   np.tile([record['latitude'], record['longitude']], (len(group), 1)), atol=1e-12)
        x, y = group.scaled.to_numpy(), group.observed.to_numpy()
        error = x - y
        expected = [error.mean(), np.mean(error ** 2), np.std(error),
                    pearsonr(x, y).statistic, spearmanr(x, y).statistic]
        np.testing.assert_allclose(expected, [record[k] for k in ['bias', 'mse', 'ubrmsd', 'p_rho', 's_rho']],
                                   rtol=1e-10, atol=1e-12)

    unique_gpis = series[['gpi', 'latitude', 'longitude', 'start_date', 'end_date', 'count']].drop_duplicates()
    assert unique_gpis.gpi.is_unique
    colocated = unique_gpis[unique_gpis.duplicated(['latitude', 'longitude'], keep=False)]
    colocated.sort_values(['latitude', 'longitude', 'gpi']).to_csv(OUT / 'colocated_gpis.csv', index=False)
    repeats = series[series.gpi.duplicated(keep=False)].sort_values(['gpi', 'row'])
    repeats.to_csv(OUT / 'repeated_gpis_by_satellite_row.csv', index=False)
    series.to_csv(OUT / 'reconciled_series.csv', index=False)

    site_records = []
    for site_id, ((lat, lon), group) in enumerate(predictions.groupby(['latitude', 'longitude']), 1):
        site_records.append(dict(site_id=site_id, latitude=lat, longitude=lon,
                                 gpis=';'.join(map(str, sorted(group.gpi.unique()))),
                                 satellite_rows=';'.join(sorted(group.row.unique())), overlaps=len(group),
                                 first_matched_date=group.date.min().date(), last_matched_date=group.date.max().date(),
                                 elapsed_span_days=(group.date.max() - group.date.min()).days,
                                 nominal_eight_day_equivalent_days=len(group) * 8,
                                 p_rho=pearsonr(group.scaled, group.observed).statistic,
                                 s_rho=spearmanr(group.scaled, group.observed).statistic))
    sites = pd.DataFrame(site_records)
    assert len(sites) == len(predictions[["latitude", "longitude"]].drop_duplicates()) and sites.overlaps.sum() == len(predictions)
    sites.to_csv(OUT / 'final_location_inventory.csv', index=False)
    gpirecords = []
    for gpi, group in predictions.groupby('gpi'):
        gpirecords.append(dict(gpi=gpi, latitude=group.latitude.iloc[0], longitude=group.longitude.iloc[0],
                               overlaps=len(group), s_rho=spearmanr(group.scaled, group.observed).statistic))
    gpis = pd.DataFrame(gpirecords)
    gpis.to_csv(OUT / 'final_gpi_inventory.csv', index=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'svg.fonttype': 'none'})
    site_summary = figures(sites, OUT, 'sites')
    for name, frame, unit in [('series', series, 'series'), ('gpi', gpis, 'GPIs')]:
        folder = OUT / name
        folder.mkdir(exist_ok=True)
        figures(frame, folder, unit)
    manifest = dict(source_workbooks={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE.glob('*.xlsx')},
                    predictions_sha256=hashlib.sha256(PREDICTIONS.read_bytes()).hexdigest(),
                    gpis=int(predictions.gpi.nunique()), exact_coordinate_sites=len(sites),
                    series=len(series), pairs=len(predictions),
                    counting_units='Physical coordinate sites, GPI records, and GPI–row series reported separately',
                    coverage_note='Pair count times eight is nominal equivalent duration, not continuous coverage')
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(site_summary.to_string())
    return OUT


if __name__ == '__main__':
    main()
