from pathlib import Path

# =========================
# Main switches
# =========================
RUN_VALIDATION = False
RUN_UQ = False
RUN_ENDPOINT_DIAGNOSTICS = True

# =========================
# Cases
# =========================
ROW_PATHS = ["149039", "150039"]
ENDPOINT_RASTER_STATS = ["mean"]

# =========================
# Raw inputs
# =========================
WIT_SMS_PATH = r"D:/SEBAL/datasets/witsms/processed/Nestle SMS/daily"
RASTER_BASE = r"D:/SEBAL/datasets/validation/LBDC_validations/rzsm"

# =========================
# Shared preprocessing
# =========================
RESCALING = True

# Apply same preprocessing logic as validation
ENDPOINT_APPLY_SMS_CALIBRATION = True
ENDPOINT_APPLY_RESCALING = True

# Site-level validation metrics used for linking back to correlation
ENDPOINT_TEMPORAL_WINDOWS = [0, 3, 5, 7]
ENDPOINT_MIN_PAIRS_CORR = 3

# Window-level controls
ENDPOINT_MIN_VALID_INCREMENTS = 2
ENDPOINT_EPS = 1e-9

# Outputs
ENDPOINT_OUTPUT_BASE = Path(r".\endpoint_diagnostics_Output")

# Plotting
ENDPOINT_SAVE_PLOTS = True