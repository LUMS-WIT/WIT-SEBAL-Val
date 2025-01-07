# configuration file consisting of variables and paths

# sebal_validation.py varaibles
ROW_PATH = '150039'
SAVE_PLOT = False
RESCALING = True
TEMPORAL_WIN = 0   # days for temporal matching
# PATTERN = "Top_soil_moisture"

# input datasets
WIT_SMS_PATH = 'D:/SEBAL/datasets/witsms/processed/Nestle SMS/daily'
RASTER_FOLDER_PATH = fr'D:/SEBAL/datasets/validation/LBDC_validations/{ROW_PATH}/'

# output
VALIDATION_FOLDER = fr'.\validations\validation_points\{ROW_PATH}_{TEMPORAL_WIN}'
IMAGES_FOLDER = fr'.\validations\figs\{ROW_PATH}'
METADATA_FILE_PATH = fr'.\validations\metadata_{ROW_PATH}_tw_{TEMPORAL_WIN}.xlsx'

# if row_path == '149039':
#     raster_file_path = r'D:\SEBAL\datasets\validation\LBDC_validations\149039\L9_Top_soil_moisture_30m_2023_06_23_174.tif'
# elif row_path =='150039':
#     raster_file_path = r'D:\SEBAL\datasets\validation\LBDC_validations\150039\L9_Top_soil_moisture_30m_2023_09_18_261.tif'
# shapefile_path = fr'D:\SEBAL\datasets\validation\LBDC_validations\shapefiles\{row_path}_out_extent_1.shp'