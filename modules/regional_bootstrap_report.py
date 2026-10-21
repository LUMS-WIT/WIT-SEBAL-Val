"""Publication figures from standalone SEBAL point ensembles, without sensor SM.

Recompute all intervals from realizations and reconcile with supplied summaries.
Retain every sensor record, but count a model pixel once per scene for spatial
summaries. Realizations remain coupled across points by their calibration pair.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


IDENTITY = ['scene', 'swath', 'date', 'excel_row', 'gpi']
PIXEL = ['scene', 'swath', 'date', 'pixel_row', 'pixel_col']
METRICS = ['ensemble_mean', 'ensemble_median', 'ensemble_sd', 'p025', 'p975',
           'interval_width', 'median_shift', 'relative_interval_width']


def sha256(path):
    with open(path, 'rb') as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_outputs(root):
    root = Path(root)
    summaries = sorted((root/'by_workbook').glob('*_point_summary.csv'))
    if not summaries:
        raise ValueError(f'No combined point summaries in {root / "by_workbook"}')
    summary_frames, realization_frames, sources = [], [], []
    for path in summaries:
        long_path = path.with_name(path.name.replace('_point_summary.csv', '_point_realizations.csv.gz'))
        if not long_path.exists():
            raise FileNotFoundError(long_path)
        summary_frames.append(pd.read_csv(path, dtype={'gpi': str, 'swath': str}))
        columns = IDENTITY + ['ensemble_order', 'replicate', 'alpha', 'beta', 'pixel_row',
                              'pixel_col', 'inside_raster', 'mask_valid', 'valid_rzsm', 'rzsm']
        realization_frames.append(pd.read_csv(long_path, usecols=columns,
                                             dtype={'gpi': str, 'swath': str}))
        sources.extend([path, long_path])
    summaries = pd.concat(summary_frames, ignore_index=True)
    realizations = pd.concat(realization_frames, ignore_index=True)
    if summaries.duplicated(IDENTITY).any() or realizations.duplicated(IDENTITY + ['ensemble_order']).any():
        raise ValueError('Duplicate scene/sensor or realization identities; do not pool overlapping exports.')
    for column in ['inside_raster', 'mask_valid', 'complete_ensemble']:
        if not summaries[column].isin([True, False]).all():
            raise ValueError(f'Summary boolean field {column} is incomplete or not boolean.')
    for column in ['inside_raster', 'mask_valid', 'valid_rzsm']:
        if not realizations[column].isin([True, False]).all():
            raise ValueError(f'Realization boolean field {column} is incomplete or not boolean.')
    identity_left = set(map(tuple, summaries[IDENTITY].to_numpy()))
    identity_right = set(map(tuple, realizations[IDENTITY].to_numpy()))
    if identity_left != identity_right:
        raise ValueError('Summary and realization case coverage differs.')
    return summaries, realizations, sources


def calculate_results(summaries, realizations):
    """Native-unit results. No rescaling, observation comparison or confidence tests."""
    records, daily, member_records, convergence = [], [], [], []
    errors = {name: 0. for name in METRICS}
    for scene, scene_summary in summaries.groupby('scene', sort=True):
        members = realizations[realizations.scene.eq(scene)].copy()
        expected_count = scene_summary.n_realizations.unique()
        if len(expected_count) != 1 or expected_count[0] < 2:
            raise ValueError(f'{scene}: inconsistent ensemble size.')
        count = int(expected_count[0])
        pair_records = members.groupby('ensemble_order')[['replicate', 'alpha', 'beta']].nunique()
        if pair_records.ne(1).any().any() or len(pair_records) != count:
            raise ValueError(f'{scene}: calibration pairs differ between points or member coverage differs.')
        valid_rows = []
        for _, row in scene_summary.iterrows():
            point = members[members.gpi.eq(row.gpi)].sort_values('ensemble_order')
            if len(point) != count:
                raise ValueError(f'{scene}/{row.gpi}: member count differs.')
            for column in ['pixel_row', 'pixel_col', 'inside_raster', 'mask_valid']:
                if not point[column].eq(row[column]).all():
                    raise ValueError(f'{scene}/{row.gpi}: {column} changes across realizations.')
            values = point.rzsm.to_numpy(float)
            finite = np.isfinite(values)
            expected_valid = finite & bool(row.mask_valid) & bool(row.inside_raster)
            if not np.array_equal(expected_valid, point.valid_rzsm.to_numpy(bool)):
                raise ValueError(f'{scene}/{row.gpi}: validity flags contradict values/masks.')
            complete = bool(expected_valid.all() and np.isfinite(row.central_rzsm))
            if complete != bool(row.complete_ensemble) or int(row.n_finite) != int(finite.sum()):
                raise ValueError(f'{scene}/{row.gpi}: summary completeness/finite count is incorrect.')
            if not complete:
                continue
            low, median, high = np.percentile(values, [2.5, 50, 97.5])
            central = float(row.central_rzsm)
            computed = dict(ensemble_mean=float(values.mean()), ensemble_median=float(median),
                            ensemble_sd=float(values.std(ddof=1)), p025=float(low), p975=float(high),
                            interval_width=float(high-low), median_shift=float(median-central),
                            relative_interval_width=float((high-low)/central) if central > 0 else np.nan)
            for name, value in computed.items():
                original = row[name]
                if not np.isclose(original, value, atol=1e-10, rtol=1e-9, equal_nan=True):
                    raise ValueError(f'{scene}/{row.gpi}: supplied {name} does not reproduce from members.')
                if np.isfinite(value):
                    errors[name] = max(errors[name], abs(float(original)-value))
            valid_rows.append(row.to_dict())
            valid_rows[-1].update(computed)
        if not valid_rows:
            daily.append(dict(scene=scene, swath=str(scene_summary.iloc[0].swath),
                              date=scene_summary.iloc[0].date, n_valid_sensor_cases=0,
                              n_unique_pixels=0, n_realizations=count))
            continue
        valid = pd.DataFrame(valid_rows)
        # Different sensor coordinates may still sample the same 30 m pixel.
        for _, duplicate in valid.groupby(['pixel_row', 'pixel_col']):
            if len(duplicate) > 1:
                for name in ['central_rzsm'] + METRICS:
                    if not np.allclose(duplicate[name], duplicate.iloc[0][name], atol=1e-10,
                                       rtol=1e-9, equal_nan=True):
                        raise ValueError(f'{scene}: sensors assigned to the same pixel have different {name}.')
        unique = valid.sort_values('gpi').drop_duplicates(PIXEL).copy()
        sensor_lists = valid.groupby(PIXEL, dropna=False).gpi.agg(lambda values: ';'.join(sorted(values)))
        unique['sensor_ids'] = [sensor_lists.loc[tuple(row[name] for name in PIXEL)] for _, row in unique.iterrows()]
        unique['n_sensor_ids_in_pixel'] = unique.sensor_ids.str.split(';').str.len()
        records.extend(unique.to_dict('records'))
        matrix = members[members.gpi.isin(unique.gpi)].pivot(index='ensemble_order', columns='gpi', values='rzsm').sort_index()
        baseline = unique.set_index('gpi').central_rzsm.reindex(matrix.columns)
        iqr = float(baseline.quantile(.75)-baseline.quantile(.25))
        central_sd = float(baseline.std(ddof=1)) if len(unique) > 1 else np.nan
        correlations = []
        for order, vector in matrix.iterrows():
            # Descriptive ranking, not a correlation with sensor observations.
            rho = float(vector.corr(baseline, method='spearman')) if len(unique) > 1 and baseline.nunique() > 1 and vector.nunique() > 1 else np.nan
            correlations.append(rho)
            parameter = members[members.ensemble_order.eq(order)].iloc[0]
            member_records.append(dict(scene=scene, swath=str(unique.iloc[0].swath), date=unique.iloc[0].date,
                                       ensemble_order=int(order), replicate=int(parameter.replicate),
                                       alpha=float(parameter.alpha), beta=float(parameter.beta),
                                       mean_rzsm=float(vector.mean()), spearman_vs_central=rho))
        shifts = matrix.sub(baseline, axis=1)
        common = shifts.mean(axis=1)
        total_variance = float(shifts.var(axis=0, ddof=1).mean())
        common_fraction = float(common.var(ddof=1)/total_variance) if total_variance > 0 else np.nan
        mean_sd = float(unique.ensemble_sd.mean())
        finite_correlations = np.asarray(correlations)[np.isfinite(correlations)]
        daily.append(dict(scene=scene, swath=str(unique.iloc[0].swath), date=unique.iloc[0].date,
                          n_valid_sensor_cases=len(valid), n_unique_pixels=len(unique), n_realizations=count,
                          central_min=float(baseline.min()), central_max=float(baseline.max()),
                          central_spatial_iqr=iqr, central_spatial_sd=central_sd,
                          mean_ensemble_sd=mean_sd, median_ensemble_sd=float(unique.ensemble_sd.median()),
                          median_interval_width=float(unique.interval_width.median()),
                          p95_interval_width=float(unique.interval_width.quantile(.95)),
                          max_interval_width=float(unique.interval_width.max()),
                          median_relative_width_percent=float(100*unique.relative_interval_width.median()),
                          median_signed_shift=float(unique.median_shift.median()),
                          n_negative_median_shifts=int(unique.median_shift.lt(0).sum()),
                          usr=mean_sd/iqr if iqr > 0 and len(unique) >= 2 else np.nan,
                          median_width_to_spatial_iqr=float(unique.interval_width.median()/iqr) if iqr > 0 else np.nan,
                          min_rank_correlation=float(finite_correlations.min()) if len(finite_correlations) else np.nan,
                          median_rank_correlation=float(np.median(finite_correlations)) if len(finite_correlations) else np.nan,
                          common_shift_variance_fraction=common_fraction,
                          min_member_rzsm=float(matrix.min().min()), max_member_rzsm=float(matrix.max().max())))
        checkpoints = sorted(set([n for n in [5, 10, 15, 20, 50, 100, 200, 500, 1000, count] if n <= count]))
        for n in checkpoints:
            data = matrix.iloc[:n].to_numpy(float)
            low, median, high = np.percentile(data, [2.5, 50, 97.5], axis=0)
            convergence.append(dict(scene=scene, swath=str(unique.iloc[0].swath), date=unique.iloc[0].date,
                                    n_realizations=n, median_interval_width=float(np.median(high-low)),
                                    mean_ensemble_sd=float(data.std(axis=0, ddof=1).mean()),
                                    median_p025=float(np.median(low)), median_p975=float(np.median(high))))
    return pd.DataFrame(records), pd.DataFrame(daily), pd.DataFrame(member_records), pd.DataFrame(convergence), errors


