# WIT-SEBAL-Val

SEBAL soil-moisture calibration, bootstrap uncertainty and WITSMS validation for
Bari Doab. Final analyses run through `main.py`; pySEBAL stays separate.

## Set up once

```bash
conda env create -f requirements.yml
conda activate sebal-val
```

Copy `config_local.example.py` to `config_local.py` if your paths differ from
`config.py`. On your existing PC, keep the private data where they are and set
those paths in the local file. The same commands work on Windows and macOS.

Private inputs are **not included in Git**. Retain these on your PC:

| Analysis | Local inputs |
|---|---|
| Validation and daily variability | WITSMS daily CSVs; `rzsm/mean/{swath}/*.tif`; `validation_inputs/private/five_fold_cohort.json` |
| Calibration | Two workbooks under `calibrations/`; retained `calibration_pairs.csv` under `uncertainty/bootstrap/inputs/calibration/` |
| Climate comparison | `CLIMATE_INPUT` CSV |
| Uncertainty figures | Completed regional ensembles and their original coefficient bank under `uncertainty/bootstrap/inputs/regional/` |
| Rice diagnostics | `SouthAsiaRice_Pakistan.zip` and `Punjab-Rice10_2024.tif` in `RICE_INPUT_ROOT` |

The configured defaults preserve this checkout's existing inputs. A new public
checkout needs access to those inputs; it cannot reproduce field results from
source code alone. [Input formats](docs/inputs.md) describe what each stage reads.

## Reproduce

Run from the repository root:

```bash
python main.py --list
python main.py --all --check
python main.py --all
```

`--check` reports missing files/packages before anything runs. `--all` runs the
final analyses from retained inputs, excluding optional raw preprocessing.
Results go to `outputs/reproduction/`; existing dated results stay in place.
`results_index.json` lists generated artifacts, and `runs/` retains run records.

For individual stages, use `python main.py --task NAME`:

| NAME | Results |
|---|---|
| `climate` | Analog-distance table |
| `calibration-merge`, `calibration-fit`, `calibration-plots` | Calibration tables, coefficients and EF/LE figures |
| `validation` | Unscaled and five-fold rescaled metrics, error/CI/fold figures |
| `maps`, `coverage` | Spatial metric maps and sample-coverage figures |
| `uncertainty-samples`, `uncertainty` | Bootstrap bank; uncertainty figures and summaries |
| `volatility` | MASD/MSSD tables and supplementary figure |
| `rice`, `rice-seasonal` | Rice overlays and seasonal error diagnostics |
| `figures` | Numbered result figures together in one folder |
| `verify` | Check the reported numerical results against the study reference |

Repeat `--task` to run several stages; selected producers run first. Otherwise,
upstream results must already exist. `python main.py --help` includes examples
and optional raw extraction commands. No arguments still run validation only.

pySEBAL is never launched by `main.py`. Use its
[separate instructions](extra%20code/SEBAL/BOOTSTRAP_README.md) to regenerate
regional model ensembles, then run `--task uncertainty`. New bootstrap samples
do not replace the coefficient bank associated with completed model runs.

## Source layout and checks

`modules/` contains calculations/plotting; `workflows/` connects stages and
checks inputs. Superseded drivers and session files are retained locally under
ignored `private_archive/`; old outputs and private data are also ignored.
Publish source through Git, not a ZIP of the working folder.

`python main.py --task export-source` prepares `outputs/reproduction/public_source/`
without private files or Git history. Use that folder for a fresh public
repository: the current repository's older history still contains data workbooks.

```bash
python -m unittest discover -s tests -v
```

The fixed validation cohort reproduces the study's historical selection;
held-out rescaling does not remove that selection limitation. Set
`SCALING_COHORT_FILE = None` to evaluate all eligible records in a new analysis.

Citation: Hamza Rafique and Abubakr Muhammad (2026), *Calibration and validation
of field scale soil moisture estimates from an Energy Balance Model for the
data-scarce Indus River Basin*, submitted to Journal of Hydrology: Regional Studies.
