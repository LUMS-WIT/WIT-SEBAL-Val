"""Callable stages for the final paper; heavy imports happen only on execution."""
import config as cfg


def climate():
    from modules.climate_analog import run
    return run(cfg.CLIMATE_INPUT, cfg.CLIMATE_OUTPUT)


def tower_extract():
    from modules.tower_extraction import run
    return run(cfg.CALIBRATION_SITES, cfg.CALIBRATION_TOWER_WORKBOOK)


def calibration_extract():
    from modules.calibration_extraction import run
    return run(cfg.CALIBRATION_SITES, cfg.CALIBRATION_MODEL_WORKBOOK)


def calibration_merge():
    from modules.calibration_merge import run
    return run(cfg.CALIBRATION_MODEL_WORKBOOK, cfg.CALIBRATION_TOWER_WORKBOOK, cfg.CALIBRATION_MERGED)


def calibration_fit():
    from modules.calibration_fit import run
    # Fit the retained pre-calibration pairs, not the already calibrated model
    # workbook used for flux evaluation. The bootstrap uses these same pairs.
    return run(cfg.CALIBRATION_FIT_INPUT, cfg.CALIBRATION_OUTPUT / 'ef_fit')


def calibration_plots():
    from modules.calibration_plotting import run
    # Original energy-flux plots use the merged model/tower observations. The
    # fit stage separately saves the EF correction and before/after LE figures.
    return run(cfg.CALIBRATION_MERGED, cfg.CALIBRATION_OUTPUT / 'evaluation')


def validation():
    from workflows.validation_scaling_workflow import main
    return main()


def maps():
    from modules.validation_maps import run
    return run(cfg.SCALING_OUTPUT_BASE, cfg.VALIDATION_MAP_OUTPUT, cfg.ROW_PATHS,
               cfg.TEMPORAL_WIN, cfg.CORRELATION_MIN_PAIRS, cfg.MAP_JITTER_DEGREES,
               cfg.MAP_SEED, cfg.MAP_NATURAL_EARTH)


def coverage():
    from modules.validation_coverage import main
    return main()


def uncertainty_inputs():
    """Optional reconstruction from tower rasters, without running pySEBAL."""
    import json
    from modules.uncertainty_bootstrap_sampling import reconstruct_pairs, export_locations
    pairs, audit, provenance = reconstruct_pairs(cfg.UNCERTAINTY_SOURCE_RASTER_ROOT,
                                                 cfg.UNCERTAINTY_SOURCE_TOWER_WORKBOOK,
                                                 cfg.UNCERTAINTY_SOURCE_SITES)
    out = cfg.UNCERTAINTY_CALIBRATION_INPUTS
    out.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(out / 'calibration_pairs.csv', index=False)
    audit.to_csv(out / 'calibration_extraction_audit.csv', index=False)
    (out / 'source_manifest.json').write_text(json.dumps(provenance, indent=2))
    cfg.UNCERTAINTY_SENSOR_LOCATIONS.parent.mkdir(parents=True, exist_ok=True)
    export_locations(cfg.UNCERTAINTY_SOURCE_LOCATION_FILE).to_csv(cfg.UNCERTAINTY_SENSOR_LOCATIONS, index=False)
    return out


def uncertainty_samples():
    from workflows.uncertainty_bootstrap_workflow import run_uncertainty_bootstrap
    return run_uncertainty_bootstrap('samples')


def uncertainty():
    from workflows.uncertainty_bootstrap_workflow import run_uncertainty_bootstrap
    return run_uncertainty_bootstrap('plots')


def volatility():
    from modules.regional_volatility import main
    from modules.volatility_plotting import make_figure
    main()
    make_figure()
    return cfg.VOLATILITY_OUTPUT


def rice():
    from modules.rice_overlay import main, secondary, compare_maps
    main()
    secondary()
    compare_maps()
    return cfg.RICE_OUTPUT


def rice_seasonal():
    from modules.seasonal_rice import main
    return main()


def figures():
    import hashlib
    import json
    import shutil
    from pathlib import Path
    from workflows.catalog import figure_sources
    output = Path(cfg.OUTPUT_ROOT) / 'figures'
    output.mkdir(parents=True, exist_ok=True)
    index = {}
    for name, (source, producer) in figure_sources().items():
        shutil.copy2(source, output / name)
        index[name] = dict(source=str(source), task=producer,
                           sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    (output / 'figure_index.json').write_text(json.dumps(index, indent=2))
    return output


def verify():
    from modules.reproduction_checks import run
    return run()


def export_source():
    from modules.source_export import run
    return run(cfg.PROJECT_ROOT, cfg.OUTPUT_ROOT / 'public_source')
