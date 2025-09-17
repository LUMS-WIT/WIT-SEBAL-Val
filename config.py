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
RASTER_FOLDER_PATH = fr'D:/SEBAL/datasets/validation/LBDC_validations/{ROW_PATH}/'

# output
VALIDATION_FOLDER = fr'.\validations\validation_points\{ROW_PATH}_{TEMPORAL_WIN}'
IMAGES_FOLDER = fr'.\validations\figs\{ROW_PATH}'
METADATA_FILE_PATH = fr'.\validations\metadata_{ROW_PATH}_tw_{TEMPORAL_WIN}.xlsx'

# ------------------------------- #
# STEP 2 variables
# ------------------------------- #

COMBINE_VALIDATIONS = True
MASTER_FOLDER = fr'.\validations\validation_points'

if COMBINE_VALIDATIONS:
    INPUT_FOLDER= MASTER_FOLDER
    OUTPUT_FILE = fr'.\validations\results\validations_tw_{TEMPORAL_WIN}.xlsx'
    PLOT_OUTPUT_FILE = fr'.\validations\results\validations_tw_{TEMPORAL_WIN}.png'
else:
    INPUT_FOLDER= VALIDATION_FOLDER
    OUTPUT_FILE = fr'.\validations\results\validations_{ROW_PATH}_tw_{TEMPORAL_WIN}.xlsx'
    PLOT_OUTPUT_FILE = fr'.\validations\results\validations_{ROW_PATH}_tw_{TEMPORAL_WIN}.png'