# Separate pySEBAL runner

This program stays outside `main.py`. Its workbooks, raster inputs, locations
and generated outputs are private and ignored by Git. Original model author
headers are preserved.

First generate coefficients from the repository root:

```bash
python main.py --task uncertainty-samples
```

Then run the model explicitly in your GDAL/pySEBAL environment, using available
local paths inside the workbook:

```bash
python "extra code/SEBAL/Run_py3_bootstrap.py" --excel "extra code/SEBAL/InputEXCEL_LBDC_149039.xlsx" --coefficients outputs/reproduction/uncertainty/samples/coefficient_bank.csv --locations uncertainty/bootstrap/inputs/sensor_locations.csv --n-realizations 200 --subset-seed 42 --output uncertainty/bootstrap/inputs/regional/149039 --preflight
```

Remove `--preflight` to run. Use `--start-row 2 --end-row 2` for a one-scene check.
Run the other swath with its own workbook and output directory. Use a new output
directory when preserving existing ensembles; `--resume` requires identical
inputs, settings and code. Publication runs used subset seed 42.

Preserve the exact bank associated with both completed swaths as
`uncertainty/bootstrap/inputs/regional/coefficient_bank_used.csv`. Keep all run
manifests, selected coefficients and `exports/` tables. Do not combine a newly
refitted bank with archived model results. Override the reporting paths in
`config_local.py` if retaining multiple bank/run combinations.

Finally run `python main.py --task uncertainty` to recreate Figures 5/6 and
uncertainty tables. Full-scene downstream statistics are retained, while only
sampled-pixel realizations are exported. `--reference-root` can compare against
saved central rasters. This standalone runner uses explicit arguments and does
not read `config_local.py`.
