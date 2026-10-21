"""Run the final study analyses. pySEBAL is a separate program /.

Usage from the repository root (edit config_local.py once for your PC):
    python main.py --list                          # stages, inputs and outputs
    python main.py --all --check                   # check before a complete run
    python main.py --all                           # reproduce all final analyses
    python main.py --task validation               # raw + five-fold scaled results
    python main.py --task maps --task coverage     # use completed validation tables
    python main.py --task uncertainty-samples      # bootstrap coefficients only
    python main.py --task uncertainty              # figures 5/6 from model exports
    python main.py --task volatility               # MASD/MSSD tables and figure
    python main.py --task rice --task rice-seasonal # overlays and seasonal comparison
    python main.py --task climate                 # climate-analog table
    python main.py --task figures                 # collect numbered result figures
    python main.py --task verify                  # check the paper's numerical results
    python main.py --task calibration-merge --task calibration-fit --task calibration-plots

Optional preprocessing (saved private inputs normally make these unnecessary):
    python main.py --task tower-extract --task calibration-extract
    python main.py --task uncertainty-inputs

Optional source-only release folder (no data, local config or Git history):
    python main.py --task export-source

No arguments retain the original validation-only behavior. Individual tasks read
existing upstream results; select their producers too when regeneration is needed.
--all runs the final analyses in dependency order, using retained workbooks and
completed regional model ensembles. It never starts pySEBAL or copies figures to
manuscript folders. See extra code/SEBAL/BOOTSTRAP_README.md for model execution.
"""


def run_uncertainty_bootstrap(mode='plots', *, repetitions=None, seed=None, regional_inputs=None):
    """Compatibility API: samples refits coefficients; plots reports saved ensembles."""
    from workflows.uncertainty_bootstrap_workflow import run_uncertainty_bootstrap as workflow
    return workflow(mode, repetitions=repetitions, seed=seed, regional_inputs=regional_inputs)


def main(argv=None):
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--task', action='append', help='Select a stage; repeat for multiple stages. See --list.')
    modes.add_argument('--all', action='store_true', help='All final analyses, excluding optional raw preprocessing.')
    modes.add_argument('--uncertainty-bootstrap', choices=['samples', 'plots'], help='Compatibility alias for uncertainty tasks.')
    modes.add_argument('--uncertainty-legacy', action='store_true', help='Archived signed-sensitivity plots only; not final results.')
    parser.add_argument('--list', action='store_true', help='List stages, inputs and output paths without running them.')
    parser.add_argument('--check', action='store_true', help='Check inputs/dependencies without running; default selection: all.')
    parser.add_argument('--bootstrap-replicates', type=int, help='Refit count in compatibility samples mode; default 5000.')
    parser.add_argument('--bootstrap-seed', type=int, help='Seed in compatibility samples mode.')
    parser.add_argument('--bootstrap-results', type=Path, action='append', help='Compatibility plots-mode regional folder; repeat per swath.')
    args = parser.parse_args(argv)
    if args.list:
        from workflows.catalog import tasks
        for name, task in tasks().items():
            print(f'{name:22} {task.description}' + (' [optional preparation]' if task.preparation else ''))
            for path, pattern, producer in task.inputs:
                print(f'  input:  {path}' + (f'/{pattern}' if pattern else '') + (f' [from {producer}]' if producer else ''))
            print(f'  output: {task.output}')
        return 0
    mode = args.uncertainty_bootstrap
    if args.bootstrap_results and mode is None:
        mode = 'plots'
    if (mode or args.uncertainty_legacy) and (args.task or args.all or args.check):
        parser.error('Use --task uncertainty / uncertainty-samples with --check, or the compatibility options separately.')
    if args.uncertainty_legacy:
        if args.bootstrap_replicates is not None or args.bootstrap_seed is not None or args.bootstrap_results:
            parser.error('Bootstrap options do not apply to legacy plots.')
        import config as cfg
        from modules.uncertainty_bootstrap_module import plot_legacy_records
        return plot_legacy_records(cfg.UNCERTAINTY_LEGACY_RECORDS, cfg.UNCERTAINTY_LEGACY_FIGURES)
    if mode:
        if mode == 'samples' and args.bootstrap_results:
            parser.error('--bootstrap-results applies only to plots mode.')
        if mode == 'plots' and (args.bootstrap_replicates is not None or args.bootstrap_seed is not None):
            parser.error('Refit count and seed apply only to samples mode.')
        return run_uncertainty_bootstrap(mode, repetitions=args.bootstrap_replicates,
                                        seed=args.bootstrap_seed, regional_inputs=args.bootstrap_results)
    if args.bootstrap_replicates is not None or args.bootstrap_seed is not None:
        parser.error('Choose --uncertainty-bootstrap samples to set a refit count or seed.')
    if args.task or args.all or args.check:
        from workflows.catalog import tasks, preflight, ordered_tasks, run
        catalog = tasks()
        requested = args.task or [name for name, task in catalog.items() if not task.preparation]
        unknown = set(requested) - set(catalog)
        if unknown:
            parser.error(f'Unknown stages: {", ".join(sorted(unknown))}. Use --list.')
        selected = ordered_tasks(requested)
        errors = preflight(selected)
        if errors:
            parser.error('Input/dependency check failed:\n' + '\n'.join(f'  {e}' for e in errors))
        if args.check:
            print('Ready: ' + ', '.join(selected))
            return 0
        return run(selected)
    from workflows.validation_scaling_workflow import main as run_validation_scaling
    return run_validation_scaling()


if __name__ == '__main__':
    main()
