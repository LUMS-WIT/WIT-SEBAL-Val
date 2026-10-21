"""Two independent modes: regenerate coefficients, or plot existing SEBAL ensembles."""
import config as cfg


def run_uncertainty_bootstrap(mode='plots', *, repetitions=None, seed=None, regional_inputs=None):
    from modules.uncertainty_bootstrap_module import generate_samples, generate_figures
    if mode == 'samples':
        return generate_samples(
            cfg.UNCERTAINTY_CALIBRATION_INPUTS, cfg.UNCERTAINTY_SAMPLE_OUTPUT,
            repetitions=cfg.UNCERTAINTY_BOOTSTRAP_REPLICATES if repetitions is None else repetitions,
            seed=cfg.UNCERTAINTY_BOOTSTRAP_SEED if seed is None else seed)
    if mode == 'plots':
        if repetitions is not None or seed is not None:
            raise ValueError('Refit count and seed apply to samples mode, not plots mode.')
        return generate_figures(
            regional_inputs or cfg.UNCERTAINTY_REGIONAL_INPUTS,
            cfg.UNCERTAINTY_REGIONAL_BANK, cfg.UNCERTAINTY_BOOTSTRAP_OUTPUT)
    raise ValueError("mode must be 'samples' or 'plots'")


main = run_uncertainty_bootstrap
