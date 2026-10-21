"""Study task catalog, input checks and result provenance. No scientific imports."""
from dataclasses import dataclass
from pathlib import Path
import importlib.util
import json
import hashlib
import time
import traceback
import config as cfg


@dataclass(frozen=True)
class Task:
    name: str
    description: str
    output: Path
    inputs: tuple = ()  # (path, glob pattern or None, producing task or None)
    packages: tuple = ()
    preparation: bool = False


def figure_sources():
    """Publication numbering, independent of historical filename conventions."""
    v = Path(cfg.SCALING_OUTPUT_BASE)
    tag = f"{cfg.VALIDATION_MEMBER}_{'_'.join(map(str, cfg.ROW_PATHS))}_tw_{cfg.TEMPORAL_WIN}"
    sources = {
        'Figure_3.png': (cfg.CALIBRATION_OUTPUT / 'evaluation/ef_site_means.png', 'calibration-plots'),
        'Figure_4.png': (cfg.CALIBRATION_OUTPUT / 'ef_fit/LE_after_EF.png', 'calibration-fit'),
        'Figure_5.png': (cfg.UNCERTAINTY_BOOTSTRAP_OUTPUT / 'figures/Figure_5.png', 'uncertainty'),
        'Figure_6.png': (cfg.UNCERTAINTY_BOOTSTRAP_OUTPUT / 'figures/Figure_6.png', 'uncertainty'),
        'Figure_7.png': (cfg.VALIDATION_MAP_OUTPUT / 'matched_pair_map.png', 'maps'),
        'Figure_8.png': (v / 'figs/error_metric_boxplots.png', 'validation'),
        'Figure_9a.png': (v / f'scaled/figs/ci_bias_{tag}.png', 'validation'),
        'Figure_9b.png': (v / f'scaled/figs/ci_ubrmsd_{tag}.png', 'validation'),
        'Supplement_MASD_MSSD.png': (cfg.VOLATILITY_OUTPUT / 'all_sensor_regional_volatility.png', 'volatility'),
    }
    for name in ['Figure_10a', 'Figure_10b', 'Figure_11', 'Figure_12a', 'Figure_12b']:
        sources[name+'.png'] = (cfg.VALIDATION_MAP_OUTPUT / (name+'.png'), 'maps')
    return sources


def tasks():
    """One place records which sources reproduce each final analysis."""
    v = Path(cfg.SCALING_OUTPUT_BASE)
    prediction = v / 'heldout_predictions.csv'
    books = tuple((v / f'scaled/results/validations_{r}_tw_{cfg.TEMPORAL_WIN}.xlsx', None, 'validation')
                  for r in cfg.ROW_PATHS)
    raw = ((Path(cfg.SCALING_WIT_SMS_PATH), '*.csv', None),) + tuple(
        (Path(cfg.SCALING_RASTER_BASE) / cfg.VALIDATION_MEMBER / str(r), '*Root_zone_moisture*.tif', None)
        for r in cfg.ROW_PATHS)
    return {
        t.name: t for t in [
            Task('climate', 'Climate-analog distances and classification table', cfg.CLIMATE_OUTPUT,
                 ((cfg.CLIMATE_INPUT, None, None),), ('numpy', 'pandas', 'scipy')),
            Task('tower-extract', 'Optional raw tower-to-overpass matching', cfg.CALIBRATION_TOWER_WORKBOOK,
                 packages=('pandas', 'xlsxwriter'), preparation=True),
            Task('calibration-extract', 'Optional model tower-footprint extraction', cfg.CALIBRATION_MODEL_WORKBOOK,
                 packages=('rasterio', 'pandas', 'xlsxwriter'), preparation=True),
            Task('calibration-merge', 'Merge the saved tower and model workbooks', cfg.CALIBRATION_MERGED,
                 ((cfg.CALIBRATION_MODEL_WORKBOOK, None, 'calibration-extract'),
                  (cfg.CALIBRATION_TOWER_WORKBOOK, None, 'tower-extract')), ('pandas', 'openpyxl')),
            Task('calibration-fit', 'EF calibration, parameters and before/after LE diagnostics',
                 cfg.CALIBRATION_OUTPUT / 'ef_fit', ((cfg.CALIBRATION_FIT_INPUT, None, 'uncertainty-inputs'),),
                 ('numpy', 'pandas', 'scipy', 'sklearn', 'matplotlib', 'xlsxwriter')),
            Task('calibration-plots', 'Overall and tower-level EF/LE evaluation figures',
                 cfg.CALIBRATION_OUTPUT / 'evaluation', ((cfg.CALIBRATION_MERGED, None, 'calibration-merge'),),
                 ('numpy', 'pandas', 'sklearn', 'matplotlib')),
            Task('validation', 'Five-fold validation, raw/scaled tables, error/CI/fold figures', v,
                 raw + (((cfg.SCALING_COHORT_FILE, None, None),) if cfg.SCALING_COHORT_FILE else ()),
                 ('numpy', 'pandas', 'scipy', 'osgeo', 'matplotlib', 'openpyxl', 'xlsxwriter')),
            Task('maps', 'Bias, MSE, ubRMSD and temporal-correlation maps', cfg.VALIDATION_MAP_OUTPUT,
                 books, ('numpy', 'pandas', 'matplotlib', 'cartopy', 'openpyxl')),
            Task('coverage', 'Matched-pair coverage at coordinate, GPI and swath levels', cfg.COVERAGE_OUTPUT,
                 books + ((prediction, None, 'validation'),), ('numpy', 'pandas', 'scipy', 'matplotlib', 'openpyxl')),
            Task('uncertainty-inputs', 'Optional reconstruction of bootstrap pairs and pixel locations',
                 cfg.UNCERTAINTY_CALIBRATION_INPUTS,
                 ((cfg.UNCERTAINTY_SOURCE_RASTER_ROOT, '*', None),
                  (cfg.UNCERTAINTY_SOURCE_TOWER_WORKBOOK, None, None),
                  (cfg.UNCERTAINTY_SOURCE_LOCATION_FILE, None, 'validation')),
                 ('numpy', 'pandas', 'rasterio', 'openpyxl'), True),
            Task('uncertainty-samples', 'Acquisition-group bootstrap coefficient bank', cfg.UNCERTAINTY_SAMPLE_OUTPUT,
                 ((cfg.UNCERTAINTY_CALIBRATION_INPUTS / 'calibration_pairs.csv', None, 'uncertainty-inputs'),),
                 ('numpy', 'pandas')),
            Task('uncertainty', 'Figures 5/6, intervals, USR and ensemble stability tables', cfg.UNCERTAINTY_BOOTSTRAP_OUTPUT,
                 ((cfg.UNCERTAINTY_REGIONAL_BANK, None, None),) + tuple(
                     (Path(p), 'run_manifest.json', None) for p in cfg.UNCERTAINTY_REGIONAL_INPUTS),
                 ('numpy', 'pandas', 'matplotlib')),
            Task('volatility', 'Unfiltered daily MASD/MSSD, regional windows and supplementary figure', cfg.VOLATILITY_OUTPUT,
                 raw + ((prediction, None, 'validation'),), ('numpy', 'pandas', 'osgeo', 'scipy', 'matplotlib')),
            Task('rice', 'Rice-map buffers, overlap tables and comparison figures', cfg.RICE_OUTPUT,
                 ((prediction, None, 'validation'),
                  (v / f'scaled/results/validations_tw_{cfg.TEMPORAL_WIN}.xlsx', None, 'validation'),
                  (cfg.RICE_ARCHIVE, None, None), (cfg.RICE_2024_RASTER, None, None)),
                 ('numpy', 'pandas', 'rasterio', 'matplotlib', 'PIL', 'openpyxl')),
            Task('rice-seasonal', 'Seasonal rice/error comparisons and sensitivity tables', cfg.RICE_SEASONAL_OUTPUT,
                 ((prediction, None, 'validation'), (cfg.RICE_OUTPUT / 'gpi_rice_overlap_2024.csv', None, 'rice'),
                  (v / f'scaled/results/validations_tw_{cfg.TEMPORAL_WIN}.xlsx', None, 'validation')),
                 ('numpy', 'pandas')),
            Task('figures', 'Collect numbered result figures in one folder', Path(cfg.OUTPUT_ROOT) / 'figures',
                 tuple((path, None, producer) for path, producer in figure_sources().values())),
            Task('verify', 'Check reported numerical results for the original study inputs', Path(cfg.OUTPUT_ROOT) / 'verification',
                 ((cfg.CLIMATE_OUTPUT / 'climate_analog_results.csv', None, 'climate'),
                  (prediction, None, 'validation'),
                  (v / 'per_gpi_metrics.csv', None, 'validation'),
                  (v / f'scaled/results/validations_tw_{cfg.TEMPORAL_WIN}.xlsx', None, 'validation'),
                  (cfg.CALIBRATION_OUTPUT / 'evaluation/ef_tower_model_pairs.csv', None, 'calibration-plots'),
                  (cfg.CALIBRATION_OUTPUT / 'ef_fit/EF_calibration_report.csv', None, 'calibration-fit'),
                  (cfg.UNCERTAINTY_BOOTSTRAP_OUTPUT / 'data/results.json', None, 'uncertainty'),
                  (cfg.UNCERTAINTY_BOOTSTRAP_OUTPUT / 'data/pixel_summary.csv', None, 'uncertainty'),
                  (cfg.UNCERTAINTY_BOOTSTRAP_OUTPUT / 'data/sampled_pixel_mean_intervals.csv', None, 'uncertainty'),
                  (cfg.VALIDATION_MAP_OUTPUT / 'map_summary.json', None, 'maps'),
                  (cfg.VOLATILITY_OUTPUT / 'regional_window_concentration.csv', None, 'volatility'),
                  (cfg.RICE_SEASONAL_OUTPUT / 'season_specific_rice_association.csv', None, 'rice-seasonal')),
                 ('numpy', 'pandas', 'scipy', 'openpyxl')),
            Task('export-source', 'Optional source-only release folder, without data or Git history',
                 Path(cfg.OUTPUT_ROOT) / 'public_source', preparation=True),
        ]
    }


def preflight(names, *, check_packages=True):
    """Check everything before running; planned producers may supply later inputs."""
    catalog, planned, errors = tasks(), set(), []
    for name in names:
        task = catalog[name]
        for path, pattern, producer in task.inputs:
            if producer in planned:
                continue
            path = Path(path)
            present = any(path.glob(pattern)) if pattern and path.is_dir() else path.is_file() if not pattern else False
            if not present:
                errors.append(f'{name}: missing {path}' + (f' (expected {pattern})' if pattern else '') +
                              (f'; run --task {producer} first or select it in this run' if producer else ''))
        if name in ('tower-extract', 'calibration-extract'):
            if not cfg.CALIBRATION_SITES:
                errors.append(f'{name}: configure CALIBRATION_SITES in config_local.py')
            for site in cfg.CALIBRATION_SITES:
                keys = ('flux_daily_path', 'flux_halfhourly_path', 'output_root') if name == 'tower-extract' else ('root_dir',)
                for key in keys:
                    if key not in site or not Path(site[key]).exists():
                        errors.append(f'{name}: missing {key} for {site.get("name", "unnamed site")}')
        if check_packages:
            for package in task.packages:
                if importlib.util.find_spec(package) is None:
                    errors.append(f'{name}: missing Python package {package}; use requirements.yml')
        planned.add(name)
    return errors


def ordered_tasks(names):
    """Order selected producers first; never add an expensive unrequested stage."""
    catalog = tasks()
    pending = [name for name in catalog if name in names]
    ordered = []
    while pending:
        ready = [name for name in pending if not any(producer in pending
                 for _, _, producer in catalog[name].inputs)]
        if not ready:
            raise ValueError('Cyclic dependencies in task catalog.')
        for name in ready:
            pending.remove(name)
            ordered.append(name)
    return ordered


def run(names):
    """Execute in catalog order and record each stage, including any failure."""
    catalog = tasks()
    if any('matplotlib' in catalog[name].packages for name in names):
        import matplotlib
        matplotlib.use('Agg')
    from workflows import study_workflows
    root = Path(cfg.OUTPUT_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / 'run_manifest.json'
    manifest = dict(started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    selected_tasks=names, tasks=[], status='running',
                    source_sha256={str(p.relative_to(cfg.PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for pattern in ['*.py', 'modules/*.py', 'workflows/*.py']
                                   for p in cfg.PROJECT_ROOT.glob(pattern)})
    for name in names:
        record = dict(name=name, output=str(catalog[name].output), status='running',
                      inputs=[dict(path=str(p), pattern=pattern,
                                   sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                   if Path(p).is_file() else None)
                              for p, pattern, _ in catalog[name].inputs])
        manifest['tasks'].append(record)
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f'\n[{name}] {catalog[name].description}', flush=True)
        start = time.monotonic()
        try:
            getattr(study_workflows, name.replace('-', '_'))()
        except Exception:
            record.update(status='failed', error=traceback.format_exc())
            manifest['status'] = 'failed'
            raise
        else:
            record['status'] = 'complete'
        finally:
            record['elapsed_seconds'] = round(time.monotonic()-start, 3)
            manifest_path.write_text(json.dumps(manifest, indent=2))
    manifest['status'] = 'complete'
    manifest['artifacts'] = {
        name: [str(p) for p in sorted(Path(catalog[name].output).rglob('*')) if p.is_file()]
        if Path(catalog[name].output).is_dir() else [str(catalog[name].output)] for name in names}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    # Keep previous runs and an accumulated per-stage artifact index, so
    # separate invocations do not hide results generated earlier.
    history = root / 'runs'
    history.mkdir(exist_ok=True)
    (history / f'{time.time_ns()}.json').write_text(json.dumps(manifest, indent=2))
    index_path = root / 'results_index.json'
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    for record in manifest['tasks']:
        index[record['name']] = dict(record, artifacts=manifest['artifacts'][record['name']])
    index_path.write_text(json.dumps(index, indent=2))
    print(f'\nResult index: {manifest_path}', flush=True)
    return manifest_path
