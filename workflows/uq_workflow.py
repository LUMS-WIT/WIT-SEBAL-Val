from modules.uq_module import (
    build_uq_case_config,
    combine_uq_triplets,
    run_uq_summary_for_case,
)

from config import ROW_PATHS


def run_uq():
    """
    Main uncertainty quantification workflow.

    For now this is kept simple:
    1. Combine lower / mean / upper overlap files
    2. Run uncertainty summary analysis on the combined files

    It can later be expanded without touching main.py.
    """

    for row_path in ROW_PATHS:
        case_cfg = build_uq_case_config(row_path=row_path)

        print("\n---------------------------------------------")
        print(f"UQ case: row={row_path}")
        print("---------------------------------------------")
        print(f"Mean dir   : {case_cfg['mean_dir']}")
        print(f"Lower dir  : {case_cfg['lower_dir']}")
        print(f"Upper dir  : {case_cfg['upper_dir']}")
        print(f"Combine dir: {case_cfg['combine_dir']}")

        combine_uq_triplets(case_cfg)
        run_uq_summary_for_case(case_cfg)