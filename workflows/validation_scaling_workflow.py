"""Five-fold, location-held-out scaling with the existing validation formats."""
import config as cfg


def main():
    # Lazy import keeps other workflows independent of this workflow's execution.
    from modules.validation_scaling_module import run_validation_scaling

    return run_validation_scaling(
        wit_sms_path=cfg.SCALING_WIT_SMS_PATH,
        raster_base=cfg.SCALING_RASTER_BASE,
        output_base=cfg.SCALING_OUTPUT_BASE,
        cohort_file=cfg.SCALING_COHORT_FILE,
        rows=cfg.ROW_PATHS,
        member=cfg.VALIDATION_MEMBER,
        temporal_win=cfg.TEMPORAL_WIN,
        n_folds=cfg.SCALING_N_FOLDS,
        show_plots=cfg.SCALING_SHOW_PLOTS,
        save_site_plots=cfg.SCALING_SAVE_SITE_PLOTS,
    )


if __name__ == "__main__":
    main()
