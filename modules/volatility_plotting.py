"""Publication boxplots of MASD and MSSD, without point or flier markers."""
from pathlib import Path
import hashlib,json,shutil
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import config as cfg
OUT=Path(cfg.VOLATILITY_OUTPUT)

def make_figure(a=None):
    if a is None:
        data=pd.read_csv(OUT/'site_volatility_scores.csv',float_precision='round_trip')
        a=data[(data.schedule=='available_combined')&(data.support=='at_least_one_increment')]
    # AV_rate and QV_rate are already means of absolute and squared increments.
    # MSSD is QV_rate itself: do not square MASD and do not take a square root.
    plt.rcParams.update({'font.size':12,'axes.labelsize':14,'axes.titlesize':14,
                         'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(10,4.6))
    regions=['South','Central','North']
    labels=[r'Mean absolute successive difference'+'\n'+r'(m$^3$/m$^3$)',
            r'Mean squared successive difference'+'\n'+r'[(m$^3$/m$^3$)$^2$]']
    for ax,field,label,panel in zip(axes,['AV_rate','QV_rate'],labels,['(a)','(b)']):
        groups=[a.loc[a.region.eq(reg),field].to_numpy() for reg in regions]
        ax.boxplot(groups,widths=.5,showfliers=False,patch_artist=False,
                   medianprops={'color':'black','linewidth':1.7},
                   boxprops={'color':'black','linewidth':1.1},
                   whiskerprops={'color':'black','linewidth':1.1},
                   capprops={'color':'black','linewidth':1.1})
        ax.set_xticks([1,2,3],[f'{r}\nn = {len(v)}' for r,v in zip(regions,groups)])
        ax.set_ylabel(label)
        ax.set_title(panel,loc='left',y=1.08,pad=3)
        ax.grid(axis='y',alpha=.2)
        ax.spines[['top','right']].set_visible(False)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    fig.tight_layout(w_pad=2.0)
    for ext in ['png','pdf','svg']:
        path=OUT/f'all_sensor_regional_volatility.{ext}'
        fig.savefig(path,dpi=400,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    medians=a.groupby('region').agg(records=('gpi','size'),MASD=('AV_rate','median'),MSSD=('QV_rate','median')).reindex(regions)
    medians.to_csv(OUT/'MASD_MSSD_regional_medians.csv')
    manifest=json.loads((OUT/'manifest.json').read_text())
    manifest['figure_metrics']={'a':'MASD = AV_rate','b':'MSSD = QV_rate (no square root)',
                                'markers':'No individual points or outlier markers',
                                'boxplot':'Quartiles, median and whiskers within 1.5 IQR; all values used in summaries'}
    manifest['code_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.py')}
    manifest['figure_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('all_sensor_regional_volatility.*')}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(medians.to_string())

if __name__=='__main__':make_figure()
