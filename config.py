"""Portable settings; private overrides belong in ignored config_local.py.

Paths are relative to this checkout, never to the working directory.
"""
from pathlib import Path
import runpy

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / 'dataset'
WIT_SMS_PATH = PROJECT_ROOT / 'witsms/processed/Nestle SMS/daily'
RASTER_BASE = DATA_ROOT / 'validations/rzsm'
OUTPUT_ROOT = PROJECT_ROOT / 'outputs/reproduction'
CALIBRATION_INPUT_ROOT = PROJECT_ROOT / 'calibrations'
CLIMATE_INPUT = PROJECT_ROOT / 'climate_analog/BariDoab_vs_FluxSites_Climate_v7.csv'
RICE_INPUT_ROOT = PROJECT_ROOT / 'outputs/rice_overlap_20261016'
UNCERTAINTY_ROOT = PROJECT_ROOT / 'uncertainty'
ROW_PATHS = ['149039', '150039']
VALIDATION_MEMBER = 'mean'
TEMPORAL_WIN = 0
SCALING_N_FOLDS = 5
SCALING_SHOW_PLOTS = SCALING_SAVE_SITE_PLOTS = False
SCALING_COHORT_FILE = PROJECT_ROOT / 'validation_inputs/private/five_fold_cohort.json'
CORRELATION_MIN_PAIRS = 6
MAP_JITTER_DEGREES = .05  # Display only; original map convention.
MAP_SEED = 42
MAP_NATURAL_EARTH = True  # Cartopy cache; first use may download basemaps.
UNCERTAINTY_BOOTSTRAP_REPLICATES = 5000
UNCERTAINTY_BOOTSTRAP_SEED = 20261011
# Public AmeriFlux tower coordinates, not private WITSMS field locations.
UNCERTAINTY_SOURCE_SITES = {
 'RIs': (43.1439, -116.736), 'Rms': (43.0645, -116.7486),
 'Rwf': (43.1207, -116.7231), 'Src': (31.9083, -110.8395),
}
CALIBRATION_SITES = []  # Optional raw-tower/footprint extraction; see example.
_local_file = PROJECT_ROOT / 'config_local.py'
_LOCAL = {k:v for k,v in runpy.run_path(str(_local_file)).items() if k.isupper()} if _local_file.exists() else {}
globals().update(_LOCAL)
if 'DATA_ROOT' in _LOCAL and 'RASTER_BASE' not in _LOCAL:
    RASTER_BASE = Path(DATA_ROOT) / 'validations/rzsm'

# Derived defaults follow overridden base directories.
SCALING_PROJECT_ROOT = PROJECT_ROOT
SCALING_WIT_SMS_PATH = WIT_SMS_PATH
SCALING_RASTER_BASE = RASTER_BASE
SCALING_OUTPUT_BASE = Path(OUTPUT_ROOT) / 'validation'
CLIMATE_OUTPUT = Path(OUTPUT_ROOT) / 'climate'
CALIBRATION_OUTPUT = Path(OUTPUT_ROOT) / 'calibration'
CALIBRATION_TOWER_WORKBOOK = Path(CALIBRATION_INPUT_ROOT) / 'landsat_meteo_energy_multisite.xlsx'
CALIBRATION_MODEL_WORKBOOK = Path(CALIBRATION_INPUT_ROOT) / 'sebal_extract_multisite.xlsx'
CALIBRATION_MERGED = CALIBRATION_OUTPUT / 'merged_calibration_multisite.csv'
CALIBRATION_CORRECTED = CALIBRATION_OUTPUT / 'ef_fit/merged_calibration_EF_corrected.csv'
VALIDATION_MAP_OUTPUT = Path(OUTPUT_ROOT) / 'validation_maps'
COVERAGE_OUTPUT = Path(OUTPUT_ROOT) / 'coverage'
VOLATILITY_OUTPUT = Path(OUTPUT_ROOT) / 'volatility'
RICE_OUTPUT = Path(OUTPUT_ROOT) / 'rice'
RICE_SEASONAL_OUTPUT = Path(OUTPUT_ROOT) / 'rice_seasonal'
RICE_ARCHIVE = Path(RICE_INPUT_ROOT) / 'SouthAsiaRice_Pakistan.zip'
RICE_2024_RASTER = Path(RICE_INPUT_ROOT) / 'Punjab-Rice10_2024.tif'
UNCERTAINTY_CALIBRATION_INPUTS = Path(UNCERTAINTY_ROOT) / 'bootstrap/inputs/calibration'
CALIBRATION_FIT_INPUT = UNCERTAINTY_CALIBRATION_INPUTS / 'calibration_pairs.csv'
UNCERTAINTY_SENSOR_LOCATIONS = Path(UNCERTAINTY_ROOT) / 'bootstrap/inputs/sensor_locations.csv'
UNCERTAINTY_SAMPLE_OUTPUT = Path(OUTPUT_ROOT) / 'uncertainty/samples'
UNCERTAINTY_BOOTSTRAP_OUTPUT = Path(OUTPUT_ROOT) / 'uncertainty'
UNCERTAINTY_REGIONAL_INPUTS = [Path(UNCERTAINTY_ROOT) / f'bootstrap/inputs/regional/{s}' for s in ROW_PATHS]
# Never replace the bank used by completed model runs with a newly refitted bank.
UNCERTAINTY_REGIONAL_BANK = Path(UNCERTAINTY_ROOT) / 'bootstrap/inputs/regional/coefficient_bank_used.csv'
UNCERTAINTY_SOURCE_RASTER_ROOT = Path(CALIBRATION_INPUT_ROOT) / 'SEBAL_out/calibrations/Calibrations'
UNCERTAINTY_SOURCE_TOWER_WORKBOOK = CALIBRATION_TOWER_WORKBOOK
UNCERTAINTY_SOURCE_LOCATION_FILE = SCALING_OUTPUT_BASE / 'heldout_predictions.csv'

# Compatibility aliases for shared helpers and archived workflows only.
COMBINE_VALIDATIONS = RESCALING = True
OUTLIER_THRESHOLD = -.47
UQ_MEMBERS = ['mean', 'lower', 'upper']
ROW_PATH = ROW_PATHS[-1]
SAVE_PLOT = SAVE_ALL_PLOTS = SHOW_ALL_PLOTS = False
SAVE_ALL_PLOTS_UQ = SHOW_ALL_PLOTS_UQ = False
VALIDATIONS_OUTPUT_BASE = Path(OUTPUT_ROOT) / 'legacy_validation'
VAL_BASE = VALIDATIONS_OUTPUT_BASE / 'validation_points'
FIG_BASE = VALIDATIONS_OUTPUT_BASE / 'figs'
RESULTS_BASE = VALIDATIONS_OUTPUT_BASE / 'results'
VALIDATION_FOLDER = VAL_BASE / VALIDATION_MEMBER / f'{ROW_PATH}_{TEMPORAL_WIN}'
MASTER_FOLDER = VAL_BASE / 'combine'
INPUT_FOLDER = MASTER_FOLDER
IMAGES_FOLDER = FIG_BASE / ROW_PATH
METADATA_FILE_PATH = VAL_BASE / f'metadata_{ROW_PATH}_tw_{TEMPORAL_WIN}.xlsx'
RASTER_FOLDER_PATH = Path(RASTER_BASE) / VALIDATION_MEMBER / ROW_PATH
OUTPUT_FILE = RESULTS_BASE / f'validations_tw_{TEMPORAL_WIN}.xlsx'
PLOT_OUTPUT_FILE = RESULTS_BASE / f'validations_tw_{TEMPORAL_WIN}.png'
UNCERTAINTY_LEGACY_RECORDS = Path(UNCERTAINTY_ROOT) / 'legacy/inputs/historical_uq_records.csv'
UNCERTAINTY_LEGACY_FIGURES = Path(OUTPUT_ROOT) / 'legacy_uncertainty/figures'
UQ_BASE = Path(UNCERTAINTY_ROOT) / 'legacy/inputs/validation_points'
FIG_BASE_UQ = UNCERTAINTY_LEGACY_FIGURES
RESULTS_BASE_UQ = Path(OUTPUT_ROOT) / 'legacy_uncertainty/data'
# Explicit per-task paths take precedence over the derived defaults.
globals().update(_LOCAL)
