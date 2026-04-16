import config as cfg

from modules.validation_module import (
    raster_folder,
    validation_folder,
    metadata_file,
    images_folder,
    figs_row_folder,
    results_file,
    list_validation_files,
    list_validation_files_multi,
    read_metadata_df,
    merge_metadata,
    generate_overlaps,
    run_validations_and_save_all,
)

def main():
    member = cfg.VALIDATION_MEMBER
    row_a, row_b = cfg.ROW_PATHS
    tw = cfg.TEMPORAL_WIN

    # Order you requested:
    # - overlaps row1 -> validate row1
    # - overlaps row2 -> validate row2
    # - validate combined

    # -------------------------
    # Row 1: overlaps -> validate
    # -------------------------
    generate_overlaps(
        row_path=row_a,
        member=member,
        wit_sms_path=cfg.WIT_SMS_PATH,
        raster_folder_path=raster_folder(cfg.RASTER_BASE, member, row_a),
        validation_folder_path=validation_folder(cfg.VAL_BASE, member, row_a, tw),
        images_folder_path=images_folder(cfg.FIG_BASE, member, row_a),
        metadata_file_path=metadata_file(cfg.VAL_BASE, member, row_a, tw),
        temporal_win=tw,
        rescaling=cfg.RESCALING,
        save_plot=cfg.SAVE_PLOT,
    )

    files_a = list_validation_files(validation_folder(cfg.VAL_BASE, member, row_a, tw))
    meta_a = read_metadata_df(metadata_file(cfg.VAL_BASE, member, row_a, tw))

    run_validations_and_save_all(
        tag=f"{member}_{row_a}",
        files=files_a,
        metadata_df=meta_a,
        output_xlsx=results_file(cfg.RESULTS_BASE, f"validations_{row_a}", tw),
        figs_out_dir=figs_row_folder(cfg.FIG_BASE, row_a),  # figs/149039
        outlier_threshold=cfg.OUTLIER_THRESHOLD,
        show_all_plots=cfg.SHOW_ALL_PLOTS,
        save_all_plots=cfg.SAVE_ALL_PLOTS,  # save box/CI plots; paired is always saved anyway
        temporal_win=tw,
    )

    # -------------------------
    # Row 2: overlaps -> validate
    # -------------------------
    generate_overlaps(
        row_path=row_b,
        member=member,
        wit_sms_path=cfg.WIT_SMS_PATH,
        raster_folder_path=raster_folder(cfg.RASTER_BASE, member, row_b),
        validation_folder_path=validation_folder(cfg.VAL_BASE, member, row_b, tw),
        images_folder_path=images_folder(cfg.FIG_BASE, member, row_b),
        metadata_file_path=metadata_file(cfg.VAL_BASE, member, row_b, tw),
        temporal_win=tw,
        rescaling=cfg.RESCALING,
        save_plot=cfg.SAVE_PLOT,
    )

    files_b = list_validation_files(validation_folder(cfg.VAL_BASE, member, row_b, tw))
    meta_b = read_metadata_df(metadata_file(cfg.VAL_BASE, member, row_b, tw))

    run_validations_and_save_all(
        tag=f"{member}_{row_b}",
        files=files_b,
        metadata_df=meta_b,
        output_xlsx=results_file(cfg.RESULTS_BASE, f"validations_{row_b}", tw),
        figs_out_dir=figs_row_folder(cfg.FIG_BASE, row_b),  # figs/150039
        outlier_threshold=cfg.OUTLIER_THRESHOLD,
        show_all_plots=cfg.SHOW_ALL_PLOTS,
        save_all_plots=cfg.SAVE_ALL_PLOTS,
        temporal_win=tw,
    )

    # -------------------------
    # Combined: validate over both folders
    # Combined plots should be directly in figs/ (not figs/mean)
    # -------------------------
    combined_files = list_validation_files_multi([
        validation_folder(cfg.VAL_BASE, member, row_a, tw),
        validation_folder(cfg.VAL_BASE, member, row_b, tw),
    ])
    combined_meta = merge_metadata(meta_a, meta_b)

    run_validations_and_save_all(
        tag=f"{member}_{row_a}_{row_b}",
        files=combined_files,
        metadata_df=combined_meta,
        output_xlsx=results_file(cfg.RESULTS_BASE, "validations", tw),  # validations_tw_0.xlsx
        figs_out_dir=cfg.FIG_BASE,  # figs/
        outlier_threshold=cfg.OUTLIER_THRESHOLD,
        show_all_plots=cfg.SHOW_ALL_PLOTS,
        save_all_plots=cfg.SAVE_ALL_PLOTS,
        temporal_win=tw,
    )

if __name__ == "__main__":
    main()