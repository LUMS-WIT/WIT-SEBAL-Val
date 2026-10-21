"""Point-output ensembles using the unchanged downstream pySEBAL functions.

NumPy + standard library only; GDAL is needed only for actual raster projection
and optional reference checks. No sensor observation values are accepted.
"""
from pathlib import Path
import contextlib
import csv
import gzip
import hashlib
import io
import json
import os
import tempfile

import numpy as np

STATE_NAMES = (
    'LE_inst', 'rn_inst', 'g_inst', 'esat_24', 'eact_24', 'Rn_24',
    'Refl_rad_water', 'Lhv', 'ETpot_24', 'QC_Map', 'water_mask', 'vegt_cover',
    'Theta_sat_top', 'Theta_sat_sub', 'Theta_res_top', 'Theta_res_sub',
    'depl_factor', 'Field_Capacity', 'FPAR', 'Soil_moisture_wilting_point',
    'Light_use_extinction_factor', 'LAI', 'sl_es_24', 'Psychro_c',
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_coefficients(path):
    with open(path, newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 2:
        raise ValueError('Coefficient input must contain at least two refits.')
    records = []
    for row in rows:
        try:
            record = dict(replicate=int(row['replicate']), alpha=float(row['alpha']), beta=float(row['beta']))
        except (KeyError, ValueError) as exc:
            raise ValueError('Coefficient CSV requires replicate,alpha,beta columns.') from exc
        if not np.isfinite([record['alpha'], record['beta']]).all():
            raise ValueError('Coefficient pairs must be finite.')
        records.append(record)
    if len({r['replicate'] for r in records}) != len(records):
        raise ValueError('Coefficient replicate identifiers must be unique.')
    return records


def select_coefficients(bank, count=200, seed=20261012):
    if count is None:
        count = len(bank)
    if not isinstance(count, int) or not 2 <= count <= len(bank):
        raise ValueError(f'Request between 2 and {len(bank)} downstream realizations.')
    # A fixed permutation gives nested subsets when count is increased.
    order = np.random.default_rng(seed).permutation(len(bank))[:count]
    return [dict(ensemble_order=i+1, **bank[index]) for i, index in enumerate(order)]


def coefficient_subset_diagnostics(bank, selected):
    """Cheap comparison of selected coefficients/EF responses with the full bank."""
    result = []
    quantities = [('alpha', lambda p: p['alpha']), ('beta', lambda p: p['beta'])]
    quantities += [(f'corrected_EF_at_{x:g}', lambda p, x=x: np.clip(p['alpha']*x+p['beta'], 0, 1.8))
                   for x in [.2, .5, 1., 1.5]]
    for quantity, calculate in quantities:
        for label, pairs in [('full_bank', bank), ('selected_subset', selected)]:
            values = np.asarray([calculate(p) for p in pairs])
            low, median, high = np.percentile(values, [2.5, 50, 97.5])
            result.append(dict(quantity=quantity, sample=label, n=len(pairs), p025=float(low),
                               median=float(median), p975=float(high), sd=float(np.std(values, ddof=1))))
    return result


def read_locations(path):
    with open(path, newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    locations = []
    for row in rows:
        try:
            gpi = str(row['gpi']).strip()
            lat, lon = float(row['latitude']), float(row['longitude'])
        except (KeyError, ValueError) as exc:
            raise ValueError('Location CSV requires gpi,latitude,longitude; optional swaths column.') from exc
        if not gpi or not np.isfinite([lat, lon]).all() or not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError(f'Invalid EPSG:4326 location: {row}')
        swaths = [s.strip() for s in row.get('swaths', '').split(';') if s.strip()]
        locations.append(dict(gpi=gpi, latitude=lat, longitude=lon, swaths=swaths))
    if not locations or len({r['gpi'] for r in locations}) != len(locations):
        raise ValueError('Location CSV must contain one unique entry per GPI.')
    return locations


def project_locations(locations, swath, shape, transform, projector):
    """Floor the inverse affine transform: same containing-pixel convention as rasterio."""
    height, width = shape
    x0, a, b, y0, d, e = transform
    matrix = np.array([[a, b], [d, e]], float)
    if abs(np.linalg.det(matrix)) < 1e-15:
        raise ValueError('Raster geotransform is singular.')
    records = []
    for location in locations:
        if location['swaths'] and swath not in location['swaths']:
            continue
        xx, yy = projector(location['longitude'], location['latitude'])
        if not np.isfinite([xx, yy]).all():
            raise ValueError(f"Cannot project GPI {location['gpi']}")
        col_float, row_float = np.linalg.solve(matrix, [xx-x0, yy-y0])
        row, col = int(np.floor(row_float)), int(np.floor(col_float))
        records.append(dict(**location, pixel_row=row, pixel_col=col,
                            inside_raster=0 <= row < height and 0 <= col < width))
    return records


def gdal_projector(wkt):
    from osgeo import osr
    if not wkt:
        raise ValueError('Scene raster has no projection.')
    source, target = osr.SpatialReference(), osr.SpatialReference()
    source.ImportFromEPSG(4326)
    target.ImportFromWkt(wkt)
    if hasattr(source, 'SetAxisMappingStrategy'):
        source.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        target.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    projection = osr.CoordinateTransformation(source, target)
    return lambda lon, lat: projection.TransformPoint(lon, lat)[:2]


def downstream_root_zone(state, alpha, beta, functions):
    """Exact original call sequence, including raw-EF application and scene statistics.

    All scene-wide inputs remain available. Calc_Soil_Moisture mutates corrected
    EF; that array is new for each realization, so upstream state stays fixed.
    """
    _, ef = functions['Calc_instantaneous_ET_fraction_calibrations'](
        state['LE_inst'], state['rn_inst'], state['g_inst'], alpha, beta, True)
    eta, _ = functions['Calc_ETact'](state['esat_24'], state['eact_24'], ef,
                                   state['Rn_24'], state['Refl_rad_water'], state['Lhv'])
    etp = np.where(state['ETpot_24'] < eta, eta, state['ETpot_24'])
    trigger, total, first, stress, top, _ = functions['Calc_Soil_Moisture'](
        eta, ef, state['QC_Map'], state['water_mask'], state['vegt_cover'],
        state['Theta_sat_top'], state['Theta_sat_sub'], state['Theta_res_top'],
        state['Theta_res_sub'], state['depl_factor'], state['Field_Capacity'],
        state['FPAR'], state['Soil_moisture_wilting_point'])
    separated = functions['Separate_E_T'](
        state['Light_use_extinction_factor'], state['LAI'], etp,
        state['Theta_res_top'], state['Theta_res_sub'], state['Theta_sat_top'],
        state['Theta_sat_sub'], top, state['sl_es_24'], state['Psychro_c'], stress,
        state['vegt_cover'], eta, trigger, first, total)
    return separated[6]


def extract_float32(array, points):
    values = np.full(len(points), np.nan)
    for i, point in enumerate(points):
        if point['inside_raster']:
            value = array[point['pixel_row'], point['pixel_col']]
            # Match original GeoTIFF Float32 serialization and sentinel handling.
            values[i] = float(np.float32(value)) if np.isfinite(value) and value > -9999 else np.nan
    return values


def point_validity(state, points):
    valid, reasons = [], []
    shape = np.shape(state['LE_inst'])
    for point in points:
        reason = ''
        if not point['inside_raster']:
            reason = 'outside_raster'
        else:
            rr, cc = point['pixel_row'], point['pixel_col']
            for name in ['QC_Map', 'water_mask', 'vegt_cover']:
                value = np.broadcast_to(state[name], shape)[rr, cc]
                if not np.isfinite(value):
                    reason = 'nonfinite_mask_or_vegetation'
                    break
                if name == 'QC_Map' and value == 1:
                    reason = 'qc_cloud_or_snow'
                    break
                if name == 'water_mask' and value == 1:
                    reason = 'water'
                    break
                if name == 'vegt_cover' and value < 0:
                    reason = 'invalid_vegetation'
                    break
        valid.append(not reason)
        reasons.append(reason)
    return np.asarray(valid, bool), reasons


def summary_record(values, central, statically_valid):
    finite = np.isfinite(values)
    n = len(values)
    result = dict(n_realizations=n, n_finite=int(finite.sum()),
                  n_valid=int(finite.sum()) if statically_valid else 0,
                  central_rzsm=central if np.isfinite(central) else '',
                  ensemble_mean='', ensemble_median='', ensemble_sd='', p025='', p975='',
                  interval_width='', median_shift='', relative_interval_width='',
                  complete_ensemble=bool(statically_valid and finite.all() and np.isfinite(central)))
    # Do not make narrow intervals by silently dropping failing realizations.
    if result['complete_ensemble']:
        low, median, high = np.percentile(values, [2.5, 50, 97.5])
        result.update(ensemble_mean=float(np.mean(values)), ensemble_median=float(median),
                      ensemble_sd=float(np.std(values, ddof=1)), p025=float(low), p975=float(high),
                      interval_width=float(high-low), median_shift=float(median-central),
                      relative_interval_width=float((high-low)/central) if central > 0 else '')
    return result


def write_table(path, records):
    if not records:
        return
    with open(path, 'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


class EnsembleContext:
    def __init__(self, *, coefficients, locations, output, central_alpha=-.089,
                 central_beta=.343, reference_root=None):
        self.coefficients = coefficients
        self.locations = locations
        self.output = Path(output)
        self.central = (central_alpha, central_beta)
        self.reference_root = Path(reference_root) if reference_root else None

    def scene_directory(self, metadata):
        scene = metadata['scene']
        if Path(scene).name != scene or any(c in scene for c in '<>:"/\\|?*'):
            raise ValueError('Scene identity must be a safe folder name.')
        return self.output/'scenes'/metadata['swath']/f"row_{metadata['excel_row']:04d}_{scene}"

    def process_scene(self, state, functions, template, metadata):
        target = self.scene_directory(metadata)
        if target.exists():
            raise FileExistsError(f'Completed scene output already exists: {target}')
        shape = np.shape(state['LE_inst'])
        if len(shape) != 2 or shape != (template.RasterYSize, template.RasterXSize):
            raise ValueError('Scene array and georeferencing dimensions differ.')
        points = project_locations(self.locations, metadata['swath'], shape,
                                   template.GetGeoTransform(), gdal_projector(template.GetProjection()))
        if not points:
            raise ValueError(f"No configured sensors assigned to swath {metadata['swath']}")
        valid, reasons = point_validity(state, points)
        n = len(self.coefficients)
        samples = np.full((n, len(points)), np.nan)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.ensemble-', dir=target.parent) as temp:
            staging = Path(temp)/'complete'
            staging.mkdir()
            with contextlib.redirect_stdout(io.StringIO()), np.errstate(all='ignore'):
                central_map = downstream_root_zone(state, *self.central, functions)
                central = extract_float32(central_map, points)
                del central_map
            comparisons = self.compare_reference(central, metadata, points)
            if comparisons:
                write_table(staging/'central_reference_comparison.csv', comparisons)
                differences = [abs(r['difference']) for r in comparisons if isinstance(r['difference'], float)]
                if differences and max(differences) > 1e-6:
                    raise ValueError(f'Central point reproduction failed (max difference {max(differences):.6g}); no ensemble published.')
            fields = ['workbook', 'excel_row', 'scene', 'date', 'swath', 'ensemble_order',
                      'replicate', 'alpha', 'beta', 'gpi', 'latitude', 'longitude',
                      'pixel_row', 'pixel_col', 'inside_raster', 'mask_valid', 'exclusion_reason',
                      'rzsm', 'valid_rzsm']
            with gzip.open(staging/'point_realizations.csv.gz', 'wt', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for index, pair in enumerate(self.coefficients):
                    with contextlib.redirect_stdout(io.StringIO()), np.errstate(all='ignore'):
                        raster = downstream_root_zone(state, pair['alpha'], pair['beta'], functions)
                        samples[index] = extract_float32(raster, points)
                        del raster
                    for j, point in enumerate(points):
                        value = samples[index, j]
                        rec = dict(metadata, **pair, **{key: point[key] for key in ['gpi', 'latitude', 'longitude', 'pixel_row', 'pixel_col', 'inside_raster']},
                                   mask_valid=bool(valid[j]), exclusion_reason=reasons[j],
                                   rzsm=float(value) if np.isfinite(value) else '',
                                   valid_rzsm=bool(valid[j] and np.isfinite(value)))
                        writer.writerow({key: rec[key] for key in fields})
                    if (index+1) % 20 == 0 or index+1 == n:
                        print(f"  Row {metadata['excel_row']}: {index+1}/{n} downstream realizations", flush=True)
            summaries, convergence = [], []
            checkpoints = sorted(set([m for m in [50, 100, 200, 500, 1000, n] if m <= n]))
            for j, point in enumerate(points):
                identity = dict(metadata, **{key: point[key] for key in ['gpi', 'latitude', 'longitude', 'pixel_row', 'pixel_col', 'inside_raster']},
                                mask_valid=bool(valid[j]), exclusion_reason=reasons[j])
                summaries.append(dict(**identity, **summary_record(samples[:, j], central[j], valid[j])))
                for m in checkpoints:
                    convergence.append(dict(**identity, **summary_record(samples[:m, j], central[j], valid[j])))
            write_table(staging/'point_summary.csv', summaries)
            write_table(staging/'downstream_convergence.csv', convergence)
            manifest = dict(metadata, n_realizations=n, n_assigned_gpis=len(points),
                            n_inside_raster=sum(p['inside_raster'] for p in points),
                            n_mask_valid=int(valid.sum()), central_alpha=self.central[0], central_beta=self.central[1],
                            n_complete_uncertainty_gpis=sum(r['complete_ensemble'] for r in summaries),
                            reference_requested=self.reference_root is not None,
                            n_finite_reference_comparisons=sum(isinstance(r['difference'], float) for r in comparisons),
                            geotransform=list(template.GetGeoTransform()), projection_wkt=template.GetProjection(),
                            array_shape=list(shape), output_units='m3/m3', output_dtype='Float32, matching original raster writer',
                            spatial_support='Assigned sensor containing pixels; original full-scene statistics retained.',
                            scope='Conditional calibration parameter uncertainty. No WITSMS observations or member-specific rescaling.',
                            complete=True)
            (staging/'scene_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
            os.rename(staging, target)
        return target

    def compare_reference(self, central, metadata, points):
        if self.reference_root is None:
            return []
        from osgeo import gdal
        date = metadata['date'].replace('-', '_')
        folder = self.reference_root/metadata['swath']
        candidates = list(folder.glob(f'*Root_zone_moisture*{date}*.tif'))
        if len(candidates) != 1:
            raise ValueError(f'Expected one central reference raster for {metadata["date"]}: {folder}')
        dataset = gdal.Open(str(candidates[0]), gdal.GA_ReadOnly)
        if dataset is None:
            raise ValueError(f'Cannot open reference raster: {candidates[0]}')
        ref_points = project_locations(self.locations, metadata['swath'],
                                       (dataset.RasterYSize, dataset.RasterXSize), dataset.GetGeoTransform(),
                                       gdal_projector(dataset.GetProjection()))
        refs = {}
        band = dataset.GetRasterBand(1)
        nodata = band.GetNoDataValue()
        for point in ref_points:
            value = np.nan
            if point['inside_raster']:
                value = float(band.ReadAsArray(point['pixel_col'], point['pixel_row'], 1, 1)[0, 0])
                if nodata is not None and value == nodata:
                    value = np.nan
            refs[point['gpi']] = value
        result = []
        for point, value in zip(points, central):
            reference = refs.get(point['gpi'], np.nan)
            finite = np.isfinite([value, reference])
            if finite[0] != finite[1]:
                raise ValueError(f"Central/reference missingness differs at GPI {point['gpi']}")
            result.append(dict(gpi=point['gpi'], central_generated=float(value) if finite[0] else '',
                               central_reference=float(reference) if finite[1] else '',
                               difference=float(value-reference) if finite.all() else '', reference_path=str(candidates[0])))
        if not any(isinstance(record['difference'], float) for record in result):
            raise ValueError('No finite central/reference comparisons; reproduction is not established.')
        return result
