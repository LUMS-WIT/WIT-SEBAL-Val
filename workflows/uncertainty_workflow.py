import config as cfg

from modules.uncertainty_module import (
    raster_folder,
    uq_member_dir,
    uq_combine_dir,
    uq_rows_combined_dir,
    uq_metadata_file,
    generate_uq_overlaps_for_member,
    combine_uq,
    combine_uq_rows,
    run_uq_analysis,
)

def main():
    tw = cfg.TEMPORAL_WIN
    row_a, row_b = cfg.ROW_PATHS

    # -------------------------
    # A) Per-row UQ (row1, row2)
    # -------------------------
    for row_path in (row_a, row_b):
        # Step 1: overlaps for mean/lower/upper
        for member in cfg.UQ_MEMBERS:
            out_dir = uq_member_dir(cfg.UQ_BASE, member, row_path, tw)
            meta_out = uq_metadata_file(cfg.UQ_BASE, member, row_path, tw)

            generate_uq_overlaps_for_member(
                row_path=row_path,
                member=member,
                wit_sms_path=cfg.WIT_SMS_PATH,
                raster_folder_path=raster_folder(cfg.RASTER_BASE, member, row_path),
                out_dir=out_dir,
                metadata_out=meta_out,
                temporal_win=tw,
                rescaling=cfg.RESCALING,
            )

        # Step 2: combine (mean/lower/upper -> combine)
        mean_dir = uq_member_dir(cfg.UQ_BASE, "mean", row_path, tw)
        lower_dir = uq_member_dir(cfg.UQ_BASE, "lower", row_path, tw)
        upper_dir = uq_member_dir(cfg.UQ_BASE, "upper", row_path, tw)
        combine_dir = uq_combine_dir(cfg.UQ_BASE, row_path, tw)

        combine_uq(
            mean_dir=mean_dir,
            lower_dir=lower_dir,
            upper_dir=upper_dir,
            combine_dir=combine_dir,
        )

        # Step 3: run uncertainty analysis for this row
        run_uq_analysis(
            input_folder=combine_dir,
            results_dir=cfg.RESULTS_BASE_UQ,
            output_basename=f"SEBAL_uncertainty_summary_{row_path}_tw_{tw}",
        )

    # -------------------------
    # B) Combined-rows UQ (row1+row2 together)
    # -------------------------
    combine_dir_a = uq_combine_dir(cfg.UQ_BASE, row_a, tw)
    combine_dir_b = uq_combine_dir(cfg.UQ_BASE, row_b, tw)

    combined_rows_dir = uq_rows_combined_dir(cfg.UQ_BASE, row_a, row_b, tw)
    combine_uq_rows(
        combine_dir_a=combine_dir_a,
        combine_dir_b=combine_dir_b,
        out_dir=combined_rows_dir,
    )

    run_uq_analysis(
        input_folder=combined_rows_dir,
        results_dir=cfg.RESULTS_BASE_UQ,
        output_basename=f"SEBAL_uncertainty_summary_tw_{tw}",
    )

if __name__ == "__main__":
    main()