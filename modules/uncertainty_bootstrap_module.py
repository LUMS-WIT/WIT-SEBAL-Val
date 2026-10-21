"""Self-contained sample generation and verified joint-ensemble reporting."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from modules.uncertainty_bootstrap_sampling import bootstrap_bank, fit_line, convergence_table
from modules.regional_bootstrap_report import load_outputs, calculate_results, sha256


def generate_samples(inputs, output, *, repetitions=5000, seed=20261011):
    """Resample retained EF pairs; no GDAL, sensor observations or SEBAL execution."""
    inputs, output = Path(inputs), Path(output)
    pairs = pd.read_csv(inputs/'calibration_pairs.csv', float_precision='round_trip')
    print(f'Refitting {repetitions:,} joint calibration samples from {len(pairs)} retained pairs.', flush=True)
    bank, draws = bootstrap_bank(pairs, repetitions, seed)
    output.mkdir(parents=True, exist_ok=True)
    bank.to_csv(output/'coefficient_bank.csv', index=False)
    draws.to_csv(output/'bootstrap_acquisition_draws.csv', index=False)
    convergence_table(bank).to_csv(output/'coefficient_convergence.csv', index=False)
    settings = dict(workflow='uncertainty_bootstrap_samples', seed=seed, repetitions=repetitions,
        source_pairs=str((inputs/'calibration_pairs.csv').resolve()),
        source_pairs_sha256=sha256(inputs/'calibration_pairs.csv'), fitted_central=fit_line(pairs.ef_model, pairs.ef_tower),
        group_counts=pairs.groupby('stratum').acquisition_group.nunique().to_dict(),
        paired_parameter_correlation=float(bank.alpha.corr(bank.beta)),
        sampling='Resample acquisition groups within calibration-site strata; keep simultaneous site records together.',
        scope='Conditional coefficient uncertainty; fixed analog sites, preprocessing and other model inputs.',
        regional_bank_unchanged=True)
    (output/'manifest.json').write_text(json.dumps(settings, indent=2)+'\n')
    print(f'Samples saved in {output}; completed regional ensembles were not changed.', flush=True)
    return output


def sampled_pixel_means(pixels, realizations, scenes):
    """Take percentiles AFTER per-member means, preserving correlated calibration draws."""
    selected = realizations.merge(pixels[['scene','gpi']], on=['scene','gpi'], validate='many_to_one')
    if selected.groupby('ensemble_order')[['replicate','alpha','beta']].nunique().ne(1).any().any():
        raise ValueError('Members must retain identical coefficient pairs across scenes and swaths.')
    matrix = selected.groupby(['date','ensemble_order']).rzsm.mean().unstack('ensemble_order')
    if not matrix.notna().all().all():
        raise ValueError('Date-specific means have incomplete member coverage.')
    means = pd.DataFrame(dict(original_mean=pixels.groupby('date').central_rzsm.mean(),
        n_valid_pixels=pixels.groupby('date').size(),
        bootstrap_median=matrix.median(axis=1), p025=matrix.quantile(.025,axis=1), p975=matrix.quantile(.975,axis=1)))
    means = means.reindex(sorted(scenes.date.unique()))
    means['n_valid_pixels'] = means.n_valid_pixels.fillna(0).astype(int)
    means['swaths'] = scenes.groupby('date').swath.agg(lambda a: ';'.join(sorted(set(a.astype(str)))))
    return means


def generate_figures(regional_inputs, bank_path, output):
    from modules.uncertainty_bootstrap_plotting import make_publication_plots
    roots, bank_path, output = [Path(p) for p in regional_inputs], Path(bank_path), Path(output)
    frames, long_frames, sources, runs, selections = [], [], [bank_path], [], []
    for root in roots:
        summary, long, paths = load_outputs(root)
        run = json.loads((root/'run_manifest.json').read_text())
        completion = json.loads((root/'run_summary.json').read_text())
        if not completion['all_requested_complete'] or completion['n_failed']:
            raise ValueError(f'Incomplete regional run: {root}')
        frames.append(summary); long_frames.append(long); sources.extend(paths)
        runs.append(run); selections.append(pd.read_csv(root/'selected_coefficients.csv'))
        sources.extend([root/'run_manifest.json',root/'run_summary.json',root/'selected_coefficients.csv'])
    if not runs:
        raise ValueError('Provide at least one regional output folder.')
    for run, selection in zip(runs[1:], selections[1:]):
        for key in ['coefficients_sha256','n_realizations','subset_seed','central_alpha','central_beta','selected_replicates']:
            if run[key] != runs[0][key]:
                raise ValueError(f'Regional settings differ: {key}')
        if not selection.equals(selections[0]):
            raise ValueError('Selected coefficients/order differ between regional runs.')
    if sha256(bank_path) != runs[0]['coefficients_sha256']:
        raise ValueError('The preserved coefficient bank does not match the completed regional runs.')
    bank, selected = pd.read_csv(bank_path), selections[0]
    check = selected.merge(bank[['replicate','alpha','beta']],on='replicate',suffixes=('','_bank'),validate='one_to_one')
    if len(check)!=len(selected) or not np.allclose(check.alpha,check.alpha_bank,rtol=0,atol=1e-12) or not np.allclose(check.beta,check.beta_bank,rtol=0,atol=1e-12):
        raise ValueError('Selected pairs do not match the preserved bank.')
    summaries, realizations = pd.concat(frames,ignore_index=True), pd.concat(long_frames,ignore_index=True)
    if summaries.duplicated(['scene','gpi']).any() or realizations.duplicated(['scene','gpi','ensemble_order']).any():
        raise ValueError('Overlapping exports would count model cases more than once.')
    print('Recomputing intervals from individual realizations for both swaths...',flush=True)
    pixels, scenes, ranks, prefixes, errors = calculate_results(summaries,realizations)
    if pixels.empty:
        raise ValueError('No complete valid pixel ensembles to plot.')
    means = sampled_pixel_means(pixels,realizations,scenes)
    data_dir, figure_dir = output/'data', output/'figures'
    data_dir.mkdir(parents=True,exist_ok=True)
    for name, table in [('pixel_summary',pixels),('scene_results',scenes),('member_spatial_metrics',ranks),('prefix_convergence',prefixes)]:
        table.to_csv(data_dir/f'{name}.csv',index=False)
    means.to_csv(data_dir/'sampled_pixel_mean_intervals.csv',index_label='date')
    # Per-case convergence matters when an aggregate summary appears stable.
    valid=realizations.merge(pixels[['scene','gpi']],on=['scene','gpi'],validate='many_to_one')
    matrix=valid.pivot(index=['scene','gpi'],columns='ensemble_order',values='rzsm').sort_index(axis=1)
    full_width=np.ptp(np.percentile(matrix.to_numpy(),[2.5,97.5],axis=1),axis=0)
    stability=[]
    for n in sorted({n for n in [20,50,100,150,len(selected)] if n<=len(selected)}):
        a=matrix.iloc[:,:n].to_numpy()
        width=np.ptp(np.percentile(a,[2.5,97.5],axis=1),axis=0)
        stability.append(dict(n_members=n,median_width=float(np.median(width)),p95_width=float(np.percentile(width,95)),
            mean_sd=float(a.std(axis=1,ddof=1).mean()),
            fraction_width_change_above_10percent=float(np.mean(abs(width-full_width)/np.maximum(full_width,1e-12)>.1))))
    pd.DataFrame(stability).to_csv(data_dir/'point_prefix_stability.csv',index=False)
    swath_summary=[]
    for swath,p in pixels.groupby('swath'):
        s=summaries[summaries.swath.eq(swath)]
        swath_summary.append(dict(swath=swath,n_scenes=int(s.scene.nunique()),n_sensors=int(s.gpi.nunique()),
            n_valid_sensor_cases=int(s.complete_ensemble.sum()),n_unique_pixel_date_cases=len(p),
            n_displayed_pixels=int(p.gpi.nunique()),mean_sd=float(p.ensemble_sd.mean()),
            median_sd=float(p.ensemble_sd.median()),p95_sd=float(p.ensemble_sd.quantile(.95))))
    pd.DataFrame(swath_summary).to_csv(data_dir/'swath_summary.csv',index=False)
    relative = 100*pixels.ensemble_sd/pixels.central_rzsm.where(pixels.central_rzsm.gt(0))
    metrics = dict(n_scenes=len(scenes), n_sensor_date_cases=len(summaries),
        n_valid_sensor_date_cases=int(summaries.complete_ensemble.sum()),n_unique_pixel_date_cases=len(pixels),
        mean_sd=float(pixels.ensemble_sd.mean()), median_sd=float(pixels.ensemble_sd.median()),
        p95_sd=float(pixels.ensemble_sd.quantile(.95)), median_relative_sd_percent=float(relative.median()),
        median_interval_width=float(pixels.interval_width.median()),p95_interval_width=float(pixels.interval_width.quantile(.95)),
        median_scene_usr=float(scenes.usr.median()), n_realizations=len(selected),
        parameter_correlation=float(bank.alpha.corr(bank.beta)))
    (data_dir/'results.json').write_text(json.dumps(metrics,indent=2)+'\n')
    manifest = dict(workflow='uncertainty_bootstrap_plots',
        source_files=[dict(path=str(p.resolve()),sha256=sha256(p)) for p in sources],
        summary_reconciliation_max_absolute_errors=errors,
        scope='Conditional calibration-parameter uncertainty; native soil moisture; validation unchanged.',
        aggregation='Percentiles across coherent member means of unique complete pixels at each actual acquisition date.',
        noncoincident_swaths='Dates may contain different swaths/location sets; no contemporaneous regional mean is inferred.',
        original_run_reference_checks=[r.get('reference_root') for r in runs])
    (data_dir/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    central = {k:runs[0][f'central_{k}'] for k in ['alpha','beta']}
    files = make_publication_plots(bank,selected,pixels,means,central,figure_dir)
    caption5=f'''Figure 5. Joint EF-calibration parameter sampling and propagated soil-moisture uncertainty. (a) Joint coefficient distribution and marginal histograms of {len(bank):,} acquisition-group bootstrap refits; the identical {len(selected)} paired settings propagated through both swaths are highlighted. The red circle labeled Mean identifies the central calibration used for the validated product (alpha={central['alpha']}, beta={central['beta']}), not the bootstrap coefficient mean or median. (b) Probability-density histogram of ensemble sample SD at {len(pixels):,} complete unique pixel–date cases. Lines show the median and 95th percentile of SD across cases. Duplicate sensors within a model pixel are counted once per scene; masked/incomplete cases are excluded.'''
    caption6=f'''Figure 6. Relative and temporal effects of conditional EF-calibration parameter uncertainty. (a) Relative uncertainty is 100 times ensemble sample SD divided by positive central soil moisture. (b) The orange Mean line is the equal-weight mean of the central product across available sampled pixels at each acquisition date; the shaded 95% CI comprises pointwise 2.5th–97.5th percentiles across {len(selected)} per-member pixel means. Spatial means are calculated before percentiles and parameters remain paired across locations and dates. The interleaved swaths have noncoincident dates and variable available locations, so the trajectory is not a simultaneous regional or area-weighted mean; coverage counts and swath membership are recorded in sampled_pixel_mean_intervals.csv. Missing acquisitions remain gaps; display connections do not imply daily retrievals or a simultaneous band. Other input/model errors, sensor error and transfer discrepancy are not sampled. Subsequent validation remains an independent assessment of central-product accuracy.'''
    (output/'CAPTIONS.md').write_text(caption5+'\n\n'+caption6+'\n')
    (output/'RESULTS.md').write_text('# Bootstrap uncertainty results\n\n'+
        '\n'.join(f'- {key}: {value}' for key,value in metrics.items())+'\n\n'+caption5+'\n\n'+caption6+'\n')
    print(json.dumps(metrics,indent=2),flush=True)
    print(f'PNG figures: {figure_dir}',flush=True)
    return files


def plot_legacy_records(records_path, output):
    """Reproduce archived signed half-difference plots without importing GDAL."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records, output = pd.read_csv(records_path), Path(output)
    output.mkdir(parents=True,exist_ok=True)
    for name in ['legacy_signed_half_difference','legacy_relative_half_difference','sebal_sm']:
        if name not in records:
            raise ValueError(f'Legacy records missing {name}')
    fig, ax = plt.subplots(figsize=(6,4),layout='constrained')
    ax.hist(records.legacy_signed_half_difference,bins=50,density=True,alpha=.7)
    ax.set(xlabel='Signed Half-Difference (m³/m³)',ylabel='Probability Density')
    ax.grid(alpha=.2); fig.savefig(output/'legacy_distribution.png',dpi=300);plt.close(fig)
    fig, ax = plt.subplots(figsize=(6,4),layout='constrained')
    ax.scatter(records.sebal_sm,records.legacy_relative_half_difference,s=8,alpha=.4)
    ax.set(xlabel='SEBAL Soil Moisture (m³/m³)',ylabel='Relative Signed Half-Difference')
    ax.grid(alpha=.2);fig.savefig(output/'legacy_relative_uncertainty.png',dpi=300);plt.close(fig)
    return output
