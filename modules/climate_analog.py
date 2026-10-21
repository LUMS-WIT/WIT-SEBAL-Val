"""Reproduce the historical climate score; retain its formula transparently.

The legacy output column is named Mahalanobis_D, but the published score
combines sqrt(Q) with squared standardized aridity before another square
root. It is not a conventional five-variable Mahalanobis distance. Changing
that formula would change the study analysis, so it is not done here.
"""
from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pandas as pd
from scipy.stats import chi2

YEARS = range(1981, 2011)
VARIABLES = ('TempRange', 'PrecipWQ', 'VPD', 'Wind')
METHOD_NOTE = (
    'Historical reproduction: score = sqrt(sqrt(Q) + z_AI**2), where Q is '
    'the four-variable dynamic-climate quadratic form. The legacy column '
    'Mahalanobis_D is retained for compatibility. This is not a conventional '
    'Mahalanobis distance; chi-square classification is reproduced from the '
    'original code and requires methodological review before interpretation.'
)


def calculate(data):
    """Return the original scores and a formula/provenance diagnostic.

    Raw annual observations are required; summary and result-only CSVs are
    not interchangeable inputs. Earth Engine numeric column prefixes are
    accepted, but each variable/year must occur exactly once.
    """
    df = data.copy()
    if not {'Site', 'AI'}.issubset(df.columns):
        raise ValueError('Climate input requires Site, AI and annual fields for 1981–2010.')
    if df.Site.isna().any() or df.Site.duplicated().any() or (df.Site == 'Bari_Doab').sum() != 1:
        raise ValueError('Site names must be unique and include exactly one Bari_Doab reference.')
    columns = {}
    for variable in VARIABLES:
        columns[variable] = []
        for year in YEARS:
            matches = [c for c in df.columns if re.fullmatch(r'(?:\d+_)?' + variable + '_' + str(year), c)]
            if len(matches) != 1:
                raise ValueError(f'Expected one {variable}_{year} field; found {len(matches)}. Use the annual-input CSV.')
            columns[variable].append(matches[0])
    required = ['AI'] + [c for variable in VARIABLES for c in columns[variable]]
    numeric = df[required].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError('Climate input contains nonfinite annual observations or aridity values.')
    df[required] = numeric
    df = df.copy()  # Consolidate the many annual columns after numeric conversion.

    annual = np.stack([df[columns[v]].to_numpy(dtype=float) for v in VARIABLES], axis=-1)
    reference = np.flatnonzero(df.Site.eq('Bari_Doab'))[0]
    covariance = np.cov(annual[reference], rowvar=False)
    if np.linalg.matrix_rank(covariance) < len(VARIABLES):
        raise ValueError('Reference annual climate covariance is singular.')
    inverse = np.linalg.inv(covariance)
    delta = annual.mean(axis=1) - annual[reference].mean(axis=0)
    quadratic = np.maximum(np.einsum('ij,jk,ik->i', delta, inverse, delta), 0.)
    aridity_sd = df.AI.std(ddof=1)
    if not np.isfinite(aridity_sd) or aridity_sd <= 0:
        raise ValueError('Aridity values must have positive between-site standard deviation.')
    aridity_squared = ((df.AI.to_numpy() - df.iloc[reference].AI) / aridity_sd)**2
    score = np.sqrt(np.sqrt(quadratic) + aridity_squared)
    lower, upper = np.sqrt(chi2.ppf([.95, .9999], 5))
    result = df.assign(Mahalanobis_D=score, Analog_Class=np.select(
        [score <= lower, score <= upper],
        ['Acceptable Analog', 'Marginal Analog (2–4σ)'], default='Novel Climate (>4σ)'))
    diagnostic = dict(
        method_note=METHOD_NOTE,
        years=[min(YEARS), max(YEARS)], variables=list(VARIABLES),
        historical_thresholds=dict(lower=float(lower), upper=float(upper)),
        # A diagnostic alternative, not a replacement or a claimed full 5-D covariance fit.
        quadratic_plus_aridity_diagnostic=dict(zip(df.Site, np.sqrt(quadratic + aridity_squared).tolist())),
    )
    return result, diagnostic


def run(input_file, output_dir):
    input_file, output_dir = Path(input_file), Path(output_dir)
    result, diagnostic = calculate(pd.read_csv(input_file))
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / 'climate_analog_results.csv', index=False)
    diagnostic.update(input_file=str(input_file.resolve()), input_sha256=hashlib.sha256(input_file.read_bytes()).hexdigest())
    (output_dir / 'climate_run.json').write_text(json.dumps(diagnostic, indent=2) + '\n')
    print(result[['Site', 'Mahalanobis_D', 'Analog_Class']].sort_values('Mahalanobis_D').to_string(index=False))
    print('Method note: historical composite score retained; see climate_run.json.', flush=True)
    return output_dir
