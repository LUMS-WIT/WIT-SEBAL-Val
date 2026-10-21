"""Seasonal checks of the unchanged 445 held-out predictions and 2024 rice overlay.

Uses existing scaled predictions without fitting separate seasonal transformations.
June--October / November--April follows the manuscript; May is excluded from that
specific comparison. Alternate windows are descriptive sensitivity checks.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

import config as cfg
ROOT = cfg.PROJECT_ROOT
OUT = Path(cfg.RICE_SEASONAL_OUTPUT)

SOURCE = Path(cfg.SCALING_OUTPUT_BASE) / 'heldout_predictions.csv'
MAP = Path(cfg.RICE_OUTPUT) / 'gpi_rice_overlap_2024.csv'
WORKBOOK = Path(cfg.SCALING_OUTPUT_BASE) / f"scaled/results/validations_tw_{cfg.TEMPORAL_WIN}.xlsx"

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [SOURCE, MAP, WORKBOOK]}
    data = pd.read_csv(SOURCE, parse_dates=['date'])
    rice = pd.read_csv(MAP)
    assert not data.empty
    data = data.merge(rice[['gpi', 'high_mse', 'high_ubrmsd', 'rice_fraction_1000m', 'rice_class_at_point']], on='gpi', validate='many_to_one')
    data['error'] = data.scaled - data.observed
    data['rice_neighborhood_25pct'] = data.rice_fraction_1000m.ge(.25)
    CLUSTER = getattr(cfg, 'RICE_DIAGNOSTIC_GPIS', [])
    WINDOWS = {
        'manuscript_JunOct_vs_NovApr': ([6,7,8,9,10], [11,12,1,2,3,4]),
        'MayOct_vs_NovApr': ([5,6,7,8,9,10], [11,12,1,2,3,4]),
        'AprSep_vs_OctMar': ([4,5,6,7,8,9], [10,11,12,1,2,3]),
        'JunNov_vs_DecMay': ([6,7,8,9,10,11], [12,1,2,3,4,5]),
    }

    def metrics(d):
        e=d.error
        return pd.Series({'n':len(d), 'bias':e.mean(), 'mse':(e**2).mean(), 'ubrmsd':np.sqrt(((e-e.mean())**2).mean())})

    group_rows=[]
    site_rows=[]
    paired_rows=[]
    season_rank_rows=[]
    for window,(summer,winter) in WINDOWS.items():
        d=data.copy()
        d['season']=np.select([d.date.dt.month.isin(summer),d.date.dt.month.isin(winter)], ['Kharif_window','comparison_window'], default='excluded')
        d=d[d.season.ne('excluded')]
        sites=d.groupby(['gpi','season']).apply(metrics,include_groups=False).reset_index()
        sites=sites.merge(rice[['gpi','rice_fraction_1000m','high_mse','high_ubrmsd']],on='gpi',validate='many_to_one')
        sites['window']=window
        site_rows.append(sites)
        # Audit original metrics against the unchanged workbook.
        for g,all_site in data.groupby('gpi'):
            expected=rice.set_index('gpi').loc[g]
            actual=metrics(all_site)
            assert np.isclose(actual.mse,expected.mse)
            assert np.isclose(actual.ubrmsd,expected.ubrmsd)
        groups={
            'all':d.gpi.notna(),
            'rice_neighborhood_25pct':d.rice_neighborhood_25pct,
            'other_neighborhoods':~d.rice_neighborhood_25pct,
            'rice_pixel_at_sensor':d.rice_class_at_point.eq(1),
            'full_period_high_mse':d.high_mse,
            'full_period_high_ubrmsd':d.high_ubrmsd,
            'full_period_high_both':d.high_mse & d.high_ubrmsd,
            'rice_overlap_error_cluster':d.gpi.isin(CLUSTER),
        }
        for group,mask in groups.items():
            for season in ['Kharif_window','comparison_window']:
                sub=d[mask & d.season.eq(season)]
                group_rows.append({'window':window,'group':group,'season':season,'n_gpis':sub.gpi.nunique(),**metrics(sub).to_dict()})
        # Compare the same GPIs, requiring at least three pairs in each window.
        wide=sites.pivot(index='gpi',columns='season',values=['n','mse','ubrmsd'])
        wide = wide.reindex(columns=pd.MultiIndex.from_product(
            [['n', 'mse', 'ubrmsd'], ['Kharif_window', 'comparison_window']]))
        both=wide[(wide['n']['Kharif_window']>=3)&(wide['n']['comparison_window']>=3)]
        for group,ids in [('all',rice.gpi),('rice_neighborhood_25pct',rice.loc[rice.rice_fraction_1000m.ge(.25),'gpi'])]:
            sub=both[both.index.isin(ids)]
            for metric in ['mse','ubrmsd']:
                paired_rows.append({'window':window,'group':group,'metric':metric,'n_gpis':len(sub),'n_higher_in_kharif':int(sub[metric]['Kharif_window'].gt(sub[metric]['comparison_window']).sum()),'median_kharif':sub[metric]['Kharif_window'].median(),'median_comparison':sub[metric]['comparison_window'].median(),'median_within_gpi_difference':(sub[metric]['Kharif_window']-sub[metric]['comparison_window']).median()})
        # Recompute elevated errors within each season using GPIs with >=3 pairs.
        for season in ['Kharif_window','comparison_window']:
            sub=sites[sites.season.eq(season)&sites.n.ge(3)].copy()
            sub['rice_group']=sub.rice_fraction_1000m.ge(.25)
            for metric in ['mse','ubrmsd']:
                threshold=sub[metric].quantile(.75)
                high=sub[metric].ge(threshold)
                season_rank_rows.append({'window':window,'season':season,'metric':metric,'n_gpis':len(sub),'n_rice_neighborhoods':int(sub.rice_group.sum()),'n_high':int(high.sum()),'n_high_rice':int((high&sub.rice_group).sum()),'median_rice_group':sub.loc[sub.rice_group,metric].median(),'median_other_group':sub.loc[~sub.rice_group,metric].median(),'spearman_error_vs_rice_fraction':sub[[metric,'rice_fraction_1000m']].rank().corr().iloc[0,1]})

    group_frame=pd.DataFrame(group_rows)
    paired_frame=pd.DataFrame(paired_rows)
    rank_frame=pd.DataFrame(season_rank_rows)
    group_frame.to_csv(OUT/'pooled_seasonal_metrics.csv',index=False)
    pd.concat(site_rows).to_csv(OUT/'per_gpi_seasonal_metrics.csv',index=False)
    paired_frame.to_csv(OUT/'matched_gpi_comparison.csv',index=False)
    rank_frame.to_csv(OUT/'season_specific_rice_association.csv',index=False)
    data[data.gpi.isin(CLUSTER)].sort_values(['gpi','date']).to_csv(OUT/'rice_overlap_cluster_dates.csv',index=False)
    (OUT/'analysis_notes.json').write_text(json.dumps({'input_sha256':hashes,'windows':WINDOWS,'rice_neighborhood_definition':'At least 25% mapped rice within 1km; descriptive threshold, not crop identity at sensor','elevated_full_period_definition':'Network upper quartile from the original full-duration workbook','season_specific_min_pairs':3,'paired_min_pairs_per_season':3,'pooled_ubrmsd_caution':'Pooled residual variance includes differences in bias between GPIs; per-GPI seasonal metrics are also supplied','seasonality_caution':'Season, site, and year coverage differ. No causal attribution or significance test is made.'},indent=2))
    print('PRIMARY POOLED COMPARISON\n',group_frame[group_frame.window.eq('manuscript_JunOct_vs_NovApr')].to_string(index=False))
    print('\nPAIRED COMPARISONS\n',paired_frame.to_string(index=False))
    print('\nSEASON-SPECIFIC RICE ASSOCIATIONS\n',rank_frame.to_string(index=False))
    for p,h in hashes.items(): assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h

    return OUT
