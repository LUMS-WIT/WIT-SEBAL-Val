"""Unfiltered regional volatility using successive available acquisitions.

Read every daily sensor file. No performance, correlation-pair, site-record-length
or maximum-window-length exclusion. Require one observable consecutive-day
increment to estimate AV/QV; unestimable windows remain explicit missing values.
Report combined available acquisitions, combined paired acquisitions, legacy
per-swath acquisitions, and the whole daily record. No favorable subset chosen.
"""
from pathlib import Path
import sys,json,hashlib
import config as cfg
ROOT=cfg.PROJECT_ROOT
import numpy as np,pandas as pd
from osgeo import gdal,osr
from utils import SoilMoistureData,remove_nan_entries
from sms_calibration import sms_calibrations
from modules.sensor_data import to_daily_frame as _to_df

OUT=Path(cfg.VOLATILITY_OUTPUT)
REGIONS=['South','Central','North']

def region(lat):return 'South' if lat<30.2 else 'Central' if lat<=30.6 else 'North'

def extract():
    soil=SoilMoistureData(str(cfg.SCALING_WIT_SMS_PATH));soil.read_data()
    meta=pd.DataFrame(soil.get_metadata())
    meta['gpi']=meta.gpi.astype(int)
    for c in ['latitude','longitude']:meta[c]=meta[c].astype(float)
    assert meta.gpi.is_unique
    meta['region']=meta.latitude.map(region)
    daily=[]
    for g in meta.gpi:
        dates,vals=soil.get_soil_moisture_by_location(gpi=str(g))
        f=_to_df(*sms_calibrations(remove_nan_entries(dates,vals)),'observed')
        if len(f):daily.append(f.assign(gpi=g))
    daily=pd.concat(daily,ignore_index=True)
    assert not daily.duplicated(['gpi','date']).any()
    daily.to_csv(OUT/'sensor_daily_all.csv',index=False)
    mf=OUT/'model_available_all.csv'
    rows=[];gdal.UseExceptions()
    for swath in map(int, cfg.ROW_PATHS):
        paths=sorted((Path(cfg.SCALING_RASTER_BASE)/cfg.VALIDATION_MEMBER/str(swath)).glob('*Root_zone_moisture*.tif'))
        for path in paths:
            parts=path.stem.split('_');date=pd.Timestamp('-'.join(parts[-4:-1]))
            ds=gdal.Open(str(path));gt=ds.GetGeoTransform();assert gt[2]==gt[4]==0
            target=ds.GetSpatialRef();target.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            source=osr.SpatialReference();source.ImportFromEPSG(4326);source.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            tx=osr.CoordinateTransformation(source,target);band=ds.GetRasterBand(1);nd=band.GetNoDataValue()
            for a in meta.itertuples():
                x,y,_=tx.TransformPoint(a.longitude,a.latitude)
                px=int(np.floor((x-gt[0])/gt[1]));py=int(np.floor((y-gt[3])/gt[5]))
                if not(0<=px<ds.RasterXSize and 0<=py<ds.RasterYSize):continue
                v=float(band.ReadAsArray(px,py,1,1)[0,0])
                if not np.isfinite(v) or (nd is not None and v==nd):continue
                rows.append(dict(gpi=a.gpi,row=swath,date=date,raw=v))
            ds=None
        print(f'Extracted {len(paths)} raster dates for swath {swath}',flush=True)
    model=pd.DataFrame(rows);model.to_csv(mf,index=False)
    base=pd.read_csv(Path(cfg.SCALING_OUTPUT_BASE)/'heldout_predictions.csv',parse_dates=['date'],float_precision='round_trip')
    z=base.merge(model,on=['gpi','row','date'],suffixes=('','_fresh'),validate='one_to_one')
    assert len(z)==len(base)
    np.testing.assert_allclose(z.raw,z.raw_fresh,rtol=0,atol=1e-12)
    z=base.merge(daily,on=['gpi','date'],suffixes=('','_fresh'),validate='many_to_one')
    assert len(z)==len(base)
    np.testing.assert_allclose(z.observed,z.observed_fresh,rtol=0,atol=1e-14)
    return meta,daily,model

def interval(sensor,start,end):
    n_days=(end-start).days
    if n_days<=0:raise ValueError('Non-increasing dates')
    f=sensor[(sensor.date>=start)&(sensor.date<=end)]
    dates=f.date.to_numpy(dtype='datetime64[D]');x=f.observed.to_numpy()
    consecutive=np.diff(dates).astype('timedelta64[D]').astype(int)==1
    increments=np.diff(x)[consecutive];m=len(increments)
    return dict(start_date=start,end_date=end,gap_days=n_days,daily_observations=len(f),
                valid_increments=m,coverage=m/n_days,complete_path=bool(m==n_days),
                AV=float(sum(abs(increments))) if m else np.nan,QV=float(sum(increments**2)) if m else np.nan,
                AV_norm=float(np.mean(abs(increments))) if m else np.nan,QV_norm=float(np.mean(increments**2)) if m else np.nan)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta,daily,model=extract();window_rows=[];site_rows=[]
    modelgroups={g:a for g,a in model.groupby('gpi')};dailygroups={g:a.sort_values('date') for g,a in daily.groupby('gpi')}
    for a in meta.itertuples():
        sensor=dailygroups.get(a.gpi,pd.DataFrame(columns=['date','observed']))
        sat=modelgroups.get(a.gpi,pd.DataFrame(columns=['date','row','raw']))
        dates=sorted(sat.date.unique())
        paired=sorted(set(dates)&set(sensor.date))
        if len(sensor):
            x=sensor.observed.to_numpy();dt=np.diff(sensor.date.to_numpy(dtype='datetime64[D]')).astype('timedelta64[D]').astype(int)
            inc=np.diff(x)[dt==1]
        else:inc=np.array([])
        site_rows.append(dict(gpi=a.gpi,latitude=a.latitude,longitude=a.longitude,region=a.region,
                              QC_daily_records=len(sensor),record_start=sensor.date.min(),record_end=sensor.date.max(),
                              available_overpasses=len(dates),paired_overpasses=len(paired),in_SEBAL_coverage=bool(len(dates)),
                              full_record_increments=len(inc),full_record_AV_rate=float(np.mean(abs(inc))) if len(inc) else np.nan,
                              full_record_QV_rate=float(np.mean(inc**2)) if len(inc) else np.nan))
        if sensor.empty or sat.empty:continue
        schedules=[('available_combined','combined',dates),('paired_combined','combined',paired)]
        schedules.extend(('available_per_swath',str(row),sorted(f.date.unique())) for row,f in sat.groupby('row'))
        for schedule,swath,ds in schedules:
            for start,end in zip(ds[:-1],ds[1:]):
                start,end=pd.Timestamp(start),pd.Timestamp(end)
                # Preserve boundary windows overlapping the daily monitoring period.
                if end<sensor.date.min() or start>sensor.date.max():continue
                rec=interval(sensor,start,end)
                rec.update(gpi=a.gpi,region=a.region,schedule=schedule,row=swath)
                window_rows.append(rec)
    w=pd.DataFrame(window_rows);w.to_csv(OUT/'all_windows.csv',index=False)
    inventory=pd.DataFrame(site_rows);inventory.to_csv(OUT/'all_sensor_inventory.csv',index=False)
    summaries=[];scores=[];concentrations=[];thresholds=[]
    for schedule in ['available_combined','paired_combined','available_per_swath']:
        for support in ['at_least_one_increment','legacy_two_increments','coverage_80pct','complete_path']:
            x=w[w.schedule.eq(schedule)].copy()
            x=x[x.valid_increments>=1]
            if support=='legacy_two_increments':x=x[x.valid_increments>=2]
            elif support=='coverage_80pct':x=x[x.coverage>=.8]
            elif support=='complete_path':x=x[x.complete_path]
            groups=[]
            for gpi,b in x.groupby('gpi'):
                rec=dict(gpi=gpi,region=b.region.iloc[0],schedule=schedule,support=support,windows=len(b),
                         increments=int(b.valid_increments.sum()),AV_rate=b.AV.sum()/b.valid_increments.sum(),
                         QV_rate=b.QV.sum()/b.valid_increments.sum(),median_window_AV=b.AV_norm.median(),
                         median_window_QV=b.QV_norm.median(),median_coverage=b.coverage.median(),
                         median_gap=b.gap_days.median(),max_gap=b.gap_days.max())
                groups.append(rec)
            b=pd.DataFrame(groups)
            if b.empty:continue
            scores.append(b.copy())
            for metric in ['AV_rate','QV_rate']:
                for q in [.75,.9]:
                    cutoff=b[metric].quantile(q);b['high']=b[metric]>cutoff
                    thresholds.append(dict(schedule=schedule,support=support,kind='site',metric=metric,quantile=q,threshold=cutoff))
                    for reg,c in b.groupby('region'):
                        summaries.append(dict(schedule=schedule,support=support,metric=metric,quantile=q,region=reg,
                                              GPIs=len(c),windows=int(c.windows.sum()),increments=int(c.increments.sum()),
                                              median_site_value=c[metric].median(),mean_site_value=c[metric].mean(),
                                              high_sites=int(c.high.sum()),high_site_fraction=c.high.mean(),
                                              median_site_windows=c.windows.median(),median_site_increments=c.increments.median(),
                                              median_site_coverage=c.median_coverage.median()))
            for metric in ['AV_norm','QV_norm']:
                for q in [.75,.9]:
                    cutoff=x[metric].quantile(q);x['high']=x[metric]>cutoff
                    thresholds.append(dict(schedule=schedule,support=support,kind='window',metric=metric,quantile=q,threshold=cutoff))
                    for reg,c in x.groupby('region'):
                        concentrations.append(dict(schedule=schedule,support=support,metric=metric,quantile=q,region=reg,
                                                   GPIs=c.gpi.nunique(),windows=len(c),high_windows=int(c.high.sum()),
                                                   window_fraction=c.high.mean(),equal_GPI_fraction=c.groupby('gpi').high.mean().mean()))
    # All QC daily observations, including days outside the acquisition span.
    full=inventory[inventory.in_SEBAL_coverage&(inventory.full_record_increments>=1)].copy()
    for metric in ['AV','QV']:
        field=f'full_record_{metric}_rate'
        for q in [.75,.9]:
            cutoff=full[field].quantile(q);full['high']=full[field]>cutoff
            thresholds.append(dict(schedule='full_daily_record',support='all_QC',kind='site',metric=metric+'_rate',quantile=q,threshold=cutoff))
            for reg,c in full.groupby('region'):
                summaries.append(dict(schedule='full_daily_record',support='all_QC',metric=metric+'_rate',quantile=q,region=reg,
                                      GPIs=len(c),windows=np.nan,increments=int(c.full_record_increments.sum()),
                                      median_site_value=c[field].median(),mean_site_value=c[field].mean(),high_sites=int(c.high.sum()),
                                      high_site_fraction=c.high.mean(),median_site_increments=c.full_record_increments.median()))
    score=pd.concat(scores,ignore_index=True);score.to_csv(OUT/'site_volatility_scores.csv',index=False)
    summary=pd.DataFrame(summaries);summary.to_csv(OUT/'regional_site_summary.csv',index=False)
    pd.DataFrame(concentrations).to_csv(OUT/'regional_window_concentration.csv',index=False)
    pd.DataFrame(thresholds).to_csv(OUT/'thresholds.csv',index=False)
    manifest=dict(sensor_files_read=len(meta),sensor_sites_with_SEBAL_coverage=int(inventory.in_SEBAL_coverage.sum()),
                  sensor_sites_with_QC_records=int((inventory.QC_daily_records>0).sum()),baseline_pairs_verified=len(pd.read_csv(Path(cfg.SCALING_OUTPUT_BASE)/"heldout_predictions.csv")),site_filters='No correlation, paired-count or record-length filter',
                  minimum_increment_for_estimation=1,maximum_window_gap=None,
                  schedules=['combined available acquisitions','combined jointly matched acquisitions','legacy per-swath acquisitions','full daily record'],
                  QC_limits=[.12,.90],region_boundaries=[30.2,30.6],
                  primary_site_score='Sum of window AV or QV divided by sum of observed consecutive-day increments',
                  baseline_sha256=hashlib.sha256((Path(cfg.SCALING_OUTPUT_BASE)/'heldout_predictions.csv').read_bytes()).hexdigest())
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2),flush=True)
    print('PRIMARY SITE RESULTS\n',summary.query('schedule=="available_combined" and support=="at_least_one_increment" and quantile==.75').to_string(index=False),flush=True)
    print('PAIRED AND FULL RECORD\n',summary.query('(schedule=="paired_combined" and support=="at_least_one_increment" or schedule=="full_daily_record") and quantile==.75').to_string(index=False),flush=True)

if __name__=='__main__':main()
