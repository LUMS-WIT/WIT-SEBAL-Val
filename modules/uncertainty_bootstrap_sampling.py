"""Uncertainty-only acquisition resampling of the existing EF calibration data.

Refits quantify coefficient uncertainty; they do not replace the central
calibration or alter the original calibration module. This module never runs
regional SEBAL, changes validation results, or reads WITSMS measurements.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fit_line(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or not np.isfinite(np.r_[x, y]).all():
        raise ValueError("EF fitting requires at least three finite paired observations.")
    dx = x - x.mean()
    sxx = dx @ dx
    if sxx <= 0:
        raise ValueError("EF model values have no variation; cannot fit this sample.")
    alpha = float(dx @ (y - y.mean()) / sxx)
    beta = float(y.mean() - alpha * x.mean())
    residual = y - (alpha * x + beta)
    sst = np.sum((y - y.mean()) ** 2)
    return dict(alpha=alpha, beta=beta, n_pairs=len(x),
                ef_residual_rmse=float(np.sqrt(np.mean(residual ** 2))),
                r_squared=float(1 - np.sum(residual ** 2) / sst) if sst > 0 else None)


def reconstruct_pairs(source_root, tower_workbook, sites):
    """Reproduce original 5x5 means, 3-decimal merge, closure and clipping."""
    import rasterio
    from rasterio.warp import transform
    from rasterio.windows import Window

    source_root, tower_workbook = Path(source_root), Path(tower_workbook)
    records, sources = [], []
    for site, (lat, lon) in sites.items():
        tower = pd.read_excel(tower_workbook, sheet_name=f"{site}_EnergyFluxes")
        tower['date'] = pd.to_datetime(tower.date)
        numeric = tower.select_dtypes('number').columns
        tower[numeric] = tower[numeric].round(3)
        for scene in sorted((source_root / site).iterdir()):
            if not scene.is_dir():
                continue
            date = pd.Timestamp(scene.name.split('_')[3])
            match = tower.loc[tower.scene.eq(scene.name) & tower.date.eq(date)]
            if len(match) != 1:
                raise ValueError(f"Ambiguous/missing tower match: {site}/{scene.name}")
            rec = dict(site=site, scene=scene.name, date=date)
            for field, pattern in [('G_inst_avg', '*G_inst*.tif'),
                                   ('Rn_inst_avg', '*Rn_inst*.tif'),
                                   ('LE_inst_avg', '*LEinst*.tif')]:
                paths = list((scene/'Output_energy_balance').glob(pattern))
                if len(paths) != 1:
                    raise ValueError(f"Expected one {field} raster: {scene}")
                with rasterio.open(paths[0]) as src:
                    xx, yy = transform('EPSG:4326', src.crs, [lon], [lat])
                    row, col = src.index(xx[0], yy[0])
                    values = src.read(1, window=Window(col-2, row-2, 5, 5), masked=True)
                    values = values.filled(np.nan)
                    rec[field] = round(float(np.nanmean(values)), 3) if np.isfinite(values).any() else np.nan
                rec[field+'_source'] = str(paths[0].resolve())
                sources.append(paths[0])
            for field in ['NETRAD', 'G_F_MDS', 'H_F_MDS', 'LE_F_MDS']:
                rec[field] = float(match.iloc[0][field])
            records.append(rec)
    merged = pd.DataFrame(records)
    with np.errstate(divide='ignore', invalid='ignore'):
        merged['ef_model_before_clip'] = merged.LE_inst_avg / (merged.Rn_inst_avg-merged.G_inst_avg)
        merged['ef_tower'] = 1 / (1 + merged.H_F_MDS / merged.LE_F_MDS)
    available = merged.NETRAD-merged.G_F_MDS
    merged['ef_tower'] = merged.ef_tower.where(np.isfinite(available) & available.gt(1e-6)).clip(0, 1.8)
    merged['ef_model'] = merged.ef_model_before_clip.clip(0, 1.8)
    keep = merged[['NETRAD', 'G_F_MDS', 'LE_F_MDS']].notna().all(axis=1)
    keep &= merged.Rn_inst_avg.ge(0) & merged.LE_inst_avg.between(-400, 800)
    keep &= np.isfinite(merged[['ef_model', 'ef_tower']]).all(axis=1)
    merged['used_in_fit'] = keep
    pairs = merged.loc[keep].copy().reset_index(drop=True)
    pairs['acquisition_group'] = pairs.scene
    pairs['stratum'] = np.where(pairs.site.eq('Src'), 'Arizona_Src', 'Idaho_three_sites')
    provenance = dict(tower_workbook=str(tower_workbook.resolve()), tower_sha256=sha256(tower_workbook),
                      raster_sources=[dict(path=str(p.resolve()), sha256=sha256(p)) for p in sorted(set(sources))])
    return pairs, merged, provenance


def draw_bootstrap_sample(pairs, rng):
    chunks, draws = [], []
    for stratum, group in pairs.groupby('stratum', sort=True):
        ids = sorted(group.acquisition_group.unique())
        for slot, acquisition in enumerate(rng.choice(ids, size=len(ids), replace=True), 1):
            block = group.loc[group.acquisition_group.eq(acquisition)].copy()
            block['draw_slot'] = slot
            chunks.append(block)
            draws.append(dict(stratum=stratum, draw_slot=slot, acquisition_group=acquisition,
                              n_records=len(block)))
    return pd.concat(chunks, ignore_index=True), draws


def bootstrap_bank(pairs, repetitions=5000, seed=20261011):
    needed = ['site', 'stratum', 'acquisition_group', 'ef_model', 'ef_tower']
    if pairs.empty or not set(needed).issubset(pairs):
        raise ValueError(f"Calibration pairs must contain {needed}")
    if not isinstance(repetitions, int) or repetitions < 2:
        raise ValueError("At least two bootstrap repetitions are required.")
    if pairs.groupby('acquisition_group').stratum.nunique().gt(1).any():
        raise ValueError("An acquisition cannot span two bootstrap strata.")
    rng = np.random.default_rng(seed)
    records, scene_draws = [], []
    for rep in range(1, repetitions+1):
        sample, draws = draw_bootstrap_sample(pairs, rng)
        fit = fit_line(sample.ef_model, sample.ef_tower)
        records.append(dict(replicate=rep, **fit, unique_scenes=sample.acquisition_group.nunique()))
        scene_draws.extend(dict(replicate=rep, **draw) for draw in draws)
    return pd.DataFrame(records), pd.DataFrame(scene_draws)


def export_locations(source_csv):
    """Read identity/coordinates/swath ONLY, never the observation columns."""
    d = pd.read_csv(source_csv, usecols=['gpi', 'row', 'latitude', 'longitude'], dtype={'gpi': str, 'row': str})
    if d.isna().any().any() or not np.isfinite(d[['latitude', 'longitude']]).all().all():
        raise ValueError("Sensor identity and coordinates must be finite and complete.")
    if d.groupby('gpi')[['latitude', 'longitude']].nunique().gt(1).any().any():
        raise ValueError("A GPI has inconsistent coordinates.")
    if not d.latitude.between(-90, 90).all() or not d.longitude.between(-180, 180).all():
        raise ValueError("Sensor coordinates must be longitude/latitude in EPSG:4326.")
    loc = d[['gpi', 'latitude', 'longitude']].drop_duplicates().sort_values('gpi')
    loc['swaths'] = loc.gpi.map(d.groupby('gpi')['row'].agg(lambda a: ';'.join(sorted(set(a)))))
    return loc.reset_index(drop=True)


def convergence_table(bank):
    rows = []
    counts = sorted(set([n for n in [100, 200, 500, 1000, 2000, 5000, len(bank)] if n <= len(bank)]))
    for n in counts:
        sample = bank.iloc[:n]
        for variable, values in [('alpha', sample.alpha), ('beta', sample.beta)] + [
                (f'corrected_EF_at_{x:g}', np.clip(sample.alpha*x+sample.beta, 0, 1.8))
                for x in [.2, .5, 1, 1.5]]:
            q = np.percentile(values, [2.5, 50, 97.5])
            rows.append(dict(n_replicates=n, quantity=variable, p025=q[0], median=q[1], p975=q[2],
                             sd=float(np.std(values, ddof=1))))
    return pd.DataFrame(rows)


