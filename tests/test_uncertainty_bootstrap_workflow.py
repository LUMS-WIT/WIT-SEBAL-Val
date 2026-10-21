import json
import tempfile
import unittest
import sys
import types
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import main
from modules.uncertainty_bootstrap_module import sampled_pixel_means, generate_samples
from modules.uncertainty_bootstrap_plotting import make_publication_plots


class WorkflowTests(unittest.TestCase):
    def test_cli_dispatch_and_invalid_mode_options(self):
        with patch('main.run_uncertainty_bootstrap') as run:
            main.main(['--uncertainty-bootstrap','samples','--bootstrap-replicates','12'])
            run.assert_called_once_with('samples',repetitions=12,seed=None,regional_inputs=None)
        with patch('main.run_uncertainty_bootstrap') as run:
            main.main(['--uncertainty-bootstrap','plots'])
            run.assert_called_once_with('plots',repetitions=None,seed=None,regional_inputs=None)
        with self.assertRaises(SystemExit):
            main.main(['--uncertainty-bootstrap','plots','--bootstrap-replicates','12'])

    def test_default_still_calls_existing_validation_workflow(self):
        called=[]
        stub=types.SimpleNamespace(main=lambda:called.append('validation_scaling'))
        with patch.dict(sys.modules,{'workflows.validation_scaling_workflow':stub}):
            main.main([])
        self.assertEqual(called,['validation_scaling'])

    def test_joint_pixel_means_do_not_pool_spatial_variability(self):
        pixels=pd.DataFrame([dict(scene='s',date='2021-01-01',gpi='a',central_rzsm=.1),
                             dict(scene='s',date='2021-01-01',gpi='b',central_rzsm=.2)])
        records=[]
        for order,change in enumerate([-.01,0,.01],1):
            for gpi,base,sign in [('a',.1,1),('b',.2,-1)]:
                records.append(dict(scene='s',date='2021-01-01',gpi=gpi,ensemble_order=order,
                    replicate=order,alpha=order,beta=1-order,rzsm=base+sign*change))
        scenes=pd.DataFrame([dict(date='2021-01-01',swath='149039'),dict(date='2021-01-02',swath='150039')])
        means=sampled_pixel_means(pixels,pd.DataFrame(records),scenes)
        self.assertAlmostEqual(means.iloc[0].original_mean,.15)
        self.assertAlmostEqual(means.iloc[0].p975-means.iloc[0].p025,0)
        self.assertEqual(means.iloc[1].n_valid_pixels,0)
        self.assertTrue(np.isnan(means.iloc[1].p025))
        bad=pd.DataFrame(records);bad.loc[0,'alpha']=99
        with self.assertRaisesRegex(ValueError,'identical coefficient pairs'):
            sampled_pixel_means(pixels,bad,scenes)

    def test_sampling_is_reproducible_without_changing_source_pairs(self):
        records=[]
        for stratum,sites in [('Idaho',['one','two','three']),('Arizona',['four'])]:
            for scene in range(4):
                for i,site in enumerate(sites):
                    x=.12+.21*scene+.03*i
                    records.append(dict(site=site,stratum=stratum,acquisition_group=f'{stratum}_{scene}',
                                        ef_model=x,ef_tower=.32-.1*x+.004*scene*i))
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);inputs=root/'inputs';inputs.mkdir()
            pair_file=inputs/'calibration_pairs.csv';pd.DataFrame(records).to_csv(pair_file,index=False)
            original=pair_file.read_bytes()
            generate_samples(inputs,root/'one',repetitions=10,seed=9)
            generate_samples(inputs,root/'two',repetitions=10,seed=9)
            self.assertEqual((root/'one/coefficient_bank.csv').read_bytes(),(root/'two/coefficient_bank.csv').read_bytes())
            self.assertEqual(pair_file.read_bytes(),original)
            self.assertTrue(json.loads((root/'one/manifest.json').read_text())['regional_bank_unchanged'])

    def test_plots_are_png_only(self):
        bank=pd.DataFrame({'alpha':[-.2,-.1,0,.1],'beta':[.5,.4,.3,.2]})
        pixels=pd.DataFrame({'central_rzsm':[.1,.12,.15],'ensemble_sd':[.001,.002,.003]})
        means=pd.DataFrame({'original_mean':[.12,.14],'p025':[.115,.135],'p975':[.125,.145]},
                           index=['2020-01-01','2021-01-01'])
        with tempfile.TemporaryDirectory() as folder:
            files=make_publication_plots(bank,bank.iloc[:2],pixels,means,
                                        {'alpha':-.1,'beta':.4},folder,{'dpi':40})
            self.assertEqual({p.name for p in files},{'Figure_5.png','Figure_6.png'})
            self.assertEqual({p.suffix for p in Path(folder).iterdir()},{'.png'})


if __name__=='__main__':unittest.main()
