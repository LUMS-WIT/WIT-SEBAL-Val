# Local inputs

All paths can be overridden in ignored `config_local.py`. `main.py --list` shows
resolved input/output paths, including upstream stages. Inputs are read in place.

- **Sensors:** the existing WITSMS daily CSV format used by
  `utils.SoilMoistureData`: metadata and daily rows are parsed from each sensor
  file. Preserve filenames and headers; `sms_calibration.py` retains study QC.
- **Regional rasters:** `RASTER_BASE/mean/{149039,150039}/`, with filenames
  containing `Root_zone_moisture` and ending in the original
  `..._YYYY_MM_DD_{swath}.tif` convention. Preserve CRS, geotransform and masks.
- **Cohort:** private JSON with `description`, `selection` and a `gpis` list.
  Keep the original file to reproduce the fixed cohort. This is distinct from
  the unfiltered all-sensor cohort used for daily volatility.
- **Climate:** `CLIMATE_INPUT` contains `Site`, `AI`, and annual descriptors
  ending in `{TempRange,PrecipWQ,VPD,Wind}_{1981..2010}`, including `Bari_Doab`.
  `calibrations/climate_inputs_gee.js` retains the Earth Engine preprocessing;
  supply your boundary asset in the Code Editor. This external export is not
  automatically downloaded by the Python workflow.
  The historical climate-score calculation is retained to reproduce the study
  table. Its additional square root differs from the standard Mahalanobis
  formula; `climate_run.json` records this discrepancy and an alternative
  diagnostic. The historical chi-square classification requires methodological
  review and must not be treated as a newly verified statistical interpretation.
- **Model/tower workbooks:** `sebal_extract_multisite.xlsx` has a sheet per site;
  `landsat_meteo_energy_multisite.xlsx` has `{site}_EnergyFluxes` sheets. Matching
  uses `scene` and `date`. Model columns include `Rn_inst_avg`, `G_inst_avg`,
  `LE_inst_avg`, and optional `EF_inst_avg`; tower columns include `NETRAD`,
  `G_F_MDS`, `H_F_MDS`, `LE_F_MDS`.
- **Calibration fit:** `CALIBRATION_FIT_INPUT` points to retained pre-calibration
  pairs, with the same flux columns. This is **not** the calibrated-model
  workbook used for EF/LE evaluation. The latter's merged table is generated
  by `calibration-merge`. Do not substitute it for the pre-calibration fit data.
- **Bootstrap pairs:** `site`, `stratum`, `acquisition_group`, `ef_model`,
  `ef_tower` in `calibration_pairs.csv`. Preserve the supplied groups, filtering
  and seed. `uncertainty-inputs` can reconstruct them from the original tower
  raster folders and export pixel locations; it does not run a model.
- **Regional ensembles:** preserve the entire exported swath folders, including
  `run_manifest.json`, `run_summary.json`, `selected_coefficients.csv`, and
  `exports/` summary/realization tables. Keep the matching original bank as
  `coefficient_bank_used.csv`. Checks reject incomplete or mismatched runs.
- **Rice:** retain the 2021 SARice ZIP and 2024 Punjab raster. Circular buffers
  retain original resolutions and area weights. Map sources are
  [SARice](https://zenodo.org/records/20923150) and
  [Punjab Rice10](https://zenodo.org/records/18468250). The 2024 map is a regional
  cross-check outside the validation period, not crop identity on a sensor date.

Optional `tower-extract` and `calibration-extract` use `CALIBRATION_SITES` (see
the example config). They prepare the workbooks from raw flux CSVs and saved
tower model scenes. They are omitted from `--all` because the workbooks are
retained inputs. The model itself remains a separate program.

Uncertainty statistics describe conditional EF coefficient uncertainty, not
total input/structural uncertainty. Validation retains its original parametric
CI formulas and their independence assumptions. Neither a six-pair correlation
threshold nor spatial holding-out guarantees precise site-level correlations.
