"""Check the study's reported aggregate results without exposing field records."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import config as cfg


def run():
    checks = []

    def check(name, actual, expected, tolerance):
        checks.append(dict(name=name, actual=float(actual), expected=expected,
                           tolerance=tolerance, passed=bool(np.isclose(actual, expected, rtol=0, atol=tolerance))))

    validation = Path(cfg.SCALING_OUTPUT_BASE)
    predictions = pd.read_csv(validation / 'heldout_predictions.csv')
    sites = pd.read_csv(validation / 'per_gpi_metrics.csv')
    check('matched_pairs', len(predictions), 445, 0)
    check('coordinate_locations', len(predictions[['latitude', 'longitude']].drop_duplicates()), 61, 0)
    for method, expected in [('raw', [-.1184, .0160, .0292]), ('scaled', [-.0194, .0034, .0329])]:
        medians = sites.loc[sites.method.eq(method), ['bias', 'MSE', 'ubRMSD']].median()
        for metric, reference in zip(medians.index, expected):
            check(f'{method}_median_{metric}', medians[metric], reference, .00005)
    ef = pd.read_csv(cfg.CALIBRATION_OUTPUT / 'evaluation/ef_tower_model_pairs.csv')
    means = ef.groupby('site')[['EF_tower', 'EF_model']].mean()
    check('site_mean_EF_RMSE', np.sqrt(np.mean((means.EF_tower-means.EF_model)**2)), .161, .0005)
    le = pd.read_csv(cfg.CALIBRATION_OUTPUT / 'ef_fit/EF_calibration_report.csv').set_index('metric')
    check('corrected_LE_RMSE', le.loc['RMSE', 'EF_after'], 65.28, .005)
    uncertainty = json.loads((cfg.UNCERTAINTY_BOOTSTRAP_OUTPUT / 'data/results.json').read_text())
    for key, value, tolerance in [('n_scenes', 192, 0), ('n_unique_pixel_date_cases', 5630, 0),
                                  ('median_sd', .0021, .00005),
                                  ('p95_sd', .0099, .00005),
                                  ('median_relative_sd_percent', 1.93, .005),
                                  ('median_scene_usr', .089, .0005)]:
        check(key, uncertainty[key], value, tolerance)
    maps = json.loads((cfg.VALIDATION_MAP_OUTPUT / 'map_summary.json').read_text())
    check('eligible_correlation_series', maps['eligible_correlation_series'], 35, 0)
    check('spearman_exceeds_pearson', maps['spearman_exceeds_pearson'], 22, 0)
    windows = pd.read_csv(cfg.VOLATILITY_OUTPUT / 'regional_window_concentration.csv')
    windows = windows[(windows.schedule == 'available_combined') &
                      (windows.support == 'at_least_one_increment') & (windows['quantile'] == .75)]
    for metric, values in [('AV_norm', [.320, .194, .172]), ('QV_norm', [.305, .202, .193])]:
        for region, value in zip(['South', 'Central', 'North'], values):
            actual = windows.loc[(windows.metric == metric) & (windows.region == region), 'window_fraction'].item()
            check(f'{region}_{metric}_upper_quartile_fraction', actual, value, .0005)
    seasonal = pd.read_csv(cfg.RICE_SEASONAL_OUTPUT / 'season_specific_rice_association.csv')
    primary = seasonal[(seasonal.window == 'manuscript_JunOct_vs_NovApr') &
                       (seasonal.season == 'Kharif_window') & (seasonal.metric == 'ubrmsd')]
    check('seasonal_rice_ubRMSD_spearman', primary.spearman_error_vs_rice_fraction.item(), .533, .0005)
    out = Path(cfg.OUTPUT_ROOT) / 'verification'
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(checks).to_csv(out / 'reported_results_check.csv', index=False)
    (out / 'summary.json').write_text(json.dumps(dict(passed=all(c['passed'] for c in checks), checks=checks), indent=2))
    failed = [c['name'] for c in checks if not c['passed']]
    if failed:
        raise ValueError(f'Study-reference differences: {failed}. See {out}; these checks target the original study inputs.')
    print(f'{len(checks)} reported-result checks passed.', flush=True)
    return out
