from pathlib import Path

# configuration file consisting of variables and paths

# ------------------------------- #
# STEP1 Variables 
# ------------------------------- #

ROW_PATH = '149039'
SAVE_PLOT = False
RESCALING = True
TEMPORAL_WIN = 0   # days for temporal matching

# input datasets
WIT_SMS_PATH = 'D:/SEBAL/datasets/witsms/processed/Nestle SMS/daily'
RASTER_FOLDER_PATH = fr'D:/SEBAL/datasets/validation/LBDC_validations/rzsm/upper/{ROW_PATH}/'

# output
VALIDATION_FOLDER = fr'.\validations_UQ\validation_points\upper\{ROW_PATH}_{TEMPORAL_WIN}'
IMAGES_FOLDER = fr'.\validations_UQ\figs\{ROW_PATH}'
METADATA_FILE_PATH = fr'.\validations_UQ\validation_points\upper\metadata_{ROW_PATH}_tw_{TEMPORAL_WIN}.xlsx'

# ------------------------------- #
# STEP 1b Combining validations for UQ
# ------------------------------- #
BASE_DIR = Path(fr".\validations_UQ\validation_points")

MEAN_DIR   = BASE_DIR / "mean"  / f"{ROW_PATH}_{TEMPORAL_WIN}"
LOWER_DIR  = BASE_DIR / "lower" / f"{ROW_PATH}_{TEMPORAL_WIN}"
UPPER_DIR  = BASE_DIR / "upper" / f"{ROW_PATH}_{TEMPORAL_WIN}"
COMBINE_DIR = BASE_DIR / "combine" / f"{ROW_PATH}_{TEMPORAL_WIN}"

# ------------------------------- #
# STEP 2 variables
# ------------------------------- #

COMBINE_VALIDATIONS = True
MASTER_FOLDER = fr'.\validations_UQ\validation_points\combine'
OUTLIER_THRESHOLD = -0.47  # threshold for pearson and spearman correlation


if COMBINE_VALIDATIONS:
    INPUT_FOLDER= MASTER_FOLDER
    OUTPUT_FILE = fr'.\validations_UQ\results\validations_tw_{TEMPORAL_WIN}.xlsx'
    PLOT_OUTPUT_FILE = fr'.\validations_UQ\results\validations_tw_{TEMPORAL_WIN}.png'
else:
    INPUT_FOLDER= VALIDATION_FOLDER
    OUTPUT_FILE = fr'.\validations_UQ\results\validations_{ROW_PATH}_tw_{TEMPORAL_WIN}.xlsx'
    PLOT_OUTPUT_FILE = fr'.\validations_UQ\results\validations_{ROW_PATH}_tw_{TEMPORAL_WIN}.png'