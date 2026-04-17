from pathlib import Path

# rows to run (2 rows for your case, but can be 1)
ROW_PATHS = ["149039", "150039"]

# UQ members kept for FUTURE workflows (uncertainty etc.)
# Validation workflow will ONLY use VALIDATION_MEMBER
UQ_MEMBERS = ["mean", "lower", "upper"]

# member used by validations routine
VALIDATION_MEMBER = "mean"

# temporal matching
TEMPORAL_WIN = 0    # days for temporal window 3, 5,7
RESCALING = True
OUTLIER_THRESHOLD = -0.47

# plotting
SAVE_ALL_PLOTS = False          # save per-site time-series plots during overlaps
SHOW_ALL_PLOTS = False     # show box/CI plots for non-final runs

# =========================
# Base paths only
# =========================

# inputs
WIT_SMS_PATH = r"D:/SEBAL/datasets/witsms/processed/Nestle SMS/daily"
RASTER_BASE = r"D:/SEBAL/datasets/validation/LBDC_validations/rzsm"  # expects: {RASTER_BASE}/{member}/{row}/

# outputs validations
VAL_BASE = Path(r".\validations_Output\validation_points")  # expects: {VAL_BASE}/{member}/{row}_{tw}/
FIG_BASE = Path(r".\validations_Output\figs")
RESULTS_BASE = Path(r".\validations_Output\results")


# outputs UQ
UQ_BASE = Path(fr".\UQ_Output\validation_points")
FIG_BASE_UQ = Path(fr".\UQ_Output\figs")
RESULTS_BASE_UQ = Path(fr".\UQ_Output\results")

# TODO: plotting not working for UQ yet to save them
SAVE_ALL_PLOTS_UQ = True       
SHOW_ALL_PLOTS_UQ = False 


# =========================
# Volatility / inter-overpass diagnostics
# =========================
VOLATILITY_OUTPUT_BASE = Path(r".\volatility_Output")

# ---- Volatility / inter-overpass diagnostics ----
RUN_INTER_OVERPASS_DIAGNOSTICS = True

INTER_OVERPASS_RASTER_STATS = ["mean"]   # keep simple for now

VALIDATIONS_OUTPUT_BASE = Path(r".\validations_Output")
MIN_VALID_INCREMENTS = 2
EPSILON_MISSED_VARIATION = 1e-9

# Important correction:
# Only compute Missed / Missed_norm if the sensor path is complete
# from one model date to the next.
REQUIRE_COMPLETE_PATH_FOR_MISSED = True

# Whether to apply the same sensor calibration and model rescaling
# logic as the validation pipeline before diagnostics.
VOLATILITY_APPLY_SMS_CALIBRATION = True
VOLATILITY_APPLY_RESCALING = True

# Site-level correlation diagnostics computed directly from raw inputs
# using centered sensor windows around each model date.
VOLATILITY_TEMPORAL_WINDOWS = [0, 3, 5, 7]
VOLATILITY_MIN_PAIRS_CORR = 3

# Scatter plotting
VOLATILITY_SAVE_SCATTERS = True


# MEAN_DIR   = BASE_DIR / "mean"  / f"{ROW_PATH}_{TEMPORAL_WIN}"
# LOWER_DIR  = BASE_DIR / "lower" / f"{ROW_PATH}_{TEMPORAL_WIN}"
# UPPER_DIR  = BASE_DIR / "upper" / f"{ROW_PATH}_{TEMPORAL_WIN}"
# COMBINE_DIR = BASE_DIR / "combine" / f"{ROW_PATH}_{TEMPORAL_WIN}"

# ------------------------------- #
# STEP1 Variables 
# ------------------------------- #

ROW_PATH = '150039'
SAVE_PLOT = False
RESCALING = True
# TEMPORAL_WIN = 0   # days for temporal matching

# input datasets
WIT_SMS_PATH = 'D:/SEBAL/datasets/witsms/processed/Nestle SMS/daily'
RASTER_FOLDER_PATH = fr'D:/SEBAL/datasets/validation/LBDC_validations/rzsm/mean/{ROW_PATH}/'

# output
VALIDATION_FOLDER = fr'.\validations_UQ\validation_points\mean\{ROW_PATH}_{TEMPORAL_WIN}'
IMAGES_FOLDER = fr'.\validations_UQ\figs\{ROW_PATH}'
METADATA_FILE_PATH = fr'.\validations_UQ\validation_points\mean\metadata_{ROW_PATH}_tw_{TEMPORAL_WIN}.xlsx'

# ------------------------------- #
# STEP 1b Combining validations for UQ
# ------------------------------- #
# BASE_DIR = Path(fr".\UQ\validation_points")

# MEAN_DIR   = BASE_DIR / "mean"  / f"{ROW_PATH}_{TEMPORAL_WIN}"
# LOWER_DIR  = BASE_DIR / "lower" / f"{ROW_PATH}_{TEMPORAL_WIN}"
# UPPER_DIR  = BASE_DIR / "upper" / f"{ROW_PATH}_{TEMPORAL_WIN}"
# COMBINE_DIR = BASE_DIR / "combine" / f"{ROW_PATH}_{TEMPORAL_WIN}"

# ------------------------------- #
# STEP 2 variables
# ------------------------------- #

COMBINE_VALIDATIONS = True
# MASTER_FOLDER = fr'.\validations_UQ\validation_points\combine'
MASTER_FOLDER = fr'.\validations_UQ\validation_points\mean'
OUTLIER_THRESHOLD = -0.47  # threshold for pearson and spearman correlation


if COMBINE_VALIDATIONS:
    INPUT_FOLDER= MASTER_FOLDER
    OUTPUT_FILE = fr'.\validations_UQ\results\validations_tw_{TEMPORAL_WIN}.xlsx'
    PLOT_OUTPUT_FILE = fr'.\validations_UQ\results\validations_tw_{TEMPORAL_WIN}.png'
else:
    INPUT_FOLDER= VALIDATION_FOLDER
    OUTPUT_FILE = fr'.\validations_UQ\results\validations_{ROW_PATH}_tw_{TEMPORAL_WIN}.xlsx'
    PLOT_OUTPUT_FILE = fr'.\validations_UQ\results\validations_{ROW_PATH}_tw_{TEMPORAL_WIN}.png'