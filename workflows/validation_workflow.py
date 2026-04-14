from pathlib import Path

from config import (
    ROW_PATHS,
    RASTER_STATS,
    WIT_SMS_PATH,
    SAVE_PLOT,
    RESCALING,
    TEMPORAL_WIN,
)

from modules.validation_module import (
    build_validation_case_config,
    generate_overlaps_for_case,
    run_validations_for_case,
)


def run_validation():
    """
    Main validation workflow.

    Loops over all requested row paths and raster stats
    (e.g. mean/lower/upper), generates overlaps, then
    runs validations for each case independently.
    """

    for row_path in ROW_PATHS:
        for raster_stat in RASTER_STATS:
            case_cfg = build_validation_case_config(
                row_path=row_path,
                raster_stat=raster_stat,
                wit_sms_path=WIT_SMS_PATH,
                rescaling=RESCALING,
                save_plot=SAVE_PLOT,
                temporal_window=TEMPORAL_WIN,
            )

            print("\n---------------------------------------------")
            print(f"Validation case: row={row_path}, stat={raster_stat}")
            print("---------------------------------------------")
            print(f"WIT SMS path     : {case_cfg['wit_sms_path']}")
            print(f"Raster folder    : {case_cfg['raster_folder_path']}")
            print(f"Validation folder: {case_cfg['validation_folder']}")
            print(f"Temporal window  : {case_cfg['temporal_window']}")
            print(f"Rescaling        : {case_cfg['rescaling']}")

            generate_overlaps_for_case(case_cfg)
            run_validations_for_case(case_cfg)