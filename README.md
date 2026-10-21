# WIT-SEBAL-Val

A research workflow for calibrating and evaluating root-zone soil moisture
derived from the **Surface Energy Balance Algorithm for Land (SEBAL)** in
Bari Doab, Indus River Basin, Pakistan.

The framework combines climate-analog calibration, bootstrap propagation of
calibration-parameter uncertainty, and field evaluation against the WITSMS
soil-moisture monitoring network. It supports regional assessment of soil-moisture
magnitudes, temporal agreement and spatial differences in model performance.

## Capabilities

- Climate-analog assessment and evaporative-fraction calibration against flux-tower observations.
- Location-held-out, five-fold statistical rescaling with separate unscaled and rescaled validation results.
- Bootstrap uncertainty summaries, spatial error maps and temporal-correlation diagnostics.
- Daily soil-moisture variability and seasonal rice-cover/error analyses.
- Numbered result figures and checks against the accompanying study's reported results.

## Installation

```bash
conda env create -f requirements.yml
conda activate sebal-val
```

Run commands from the repository root. Copy `config_local.example.py` to
`config_local.py` and configure the paths for your datasets. Local configuration,
observations, model inputs and generated results are excluded from Git.

## Usage

```bash
python main.py --list
python main.py --task validation --check
python main.py --task validation
python main.py --task maps --task coverage
```

`--list` shows every stage and its required inputs and output paths. `--check`
checks requirements without executing an analysis. Repeat `--task` to select
multiple stages; selected upstream stages run first.

| Stages | Purpose |
|---|---|
| `climate` | Climate-analog comparison |
| `calibration-merge`, `calibration-fit`, `calibration-plots` | Tower/model calibration and energy-flux evaluation |
| `validation`, `maps`, `coverage` | Validation metrics, spatial maps and sampling coverage |
| `uncertainty-samples`, `uncertainty` | Bootstrap coefficients and reporting of completed model ensembles |
| `volatility`, `rice`, `rice-seasonal` | Daily variability and crop-related error diagnostics |
| `figures`, `verify` | Collect result figures and check the study's reported values |

With all study inputs available, `python main.py --all` reproduces the analysis
workflow. `verify` compares results with the original study and is intended for
those inputs. For new datasets, set `SCALING_COHORT_FILE = None` to evaluate all
eligible records and select the relevant stages.

Results are written to `outputs/reproduction/` by default. `results_index.json`
indexes generated artifacts; `runs/` records completed executions.

## Data and model requirements

Field observations and full model inputs are not bundled with this repository.
See [input formats and requirements](docs/inputs.md) for the sensor records,
rasters, calibration tables and ensemble exports needed by each analysis.

The pySEBAL implementation remains under `extra code/SEBAL/` and runs separately
from `main.py`. Follow its [execution instructions](extra%20code/SEBAL/BOOTSTRAP_README.md)
to generate regional ensembles before reporting their uncertainty. The analysis
quantifies uncertainty in fitted calibration parameters, conditional on the
selected sites and other model inputs.

## Repository structure

```text
main.py                  Workflow entry point
config.py                Shared settings and default paths
config_local.example.py  Local configuration template
modules/                 Calculations and plotting
workflows/               Stage orchestration and input checks
extra code/SEBAL/        Standalone model implementation
docs/                    Input and model-execution documentation
tests/                   Scientific and workflow regression checks
```

```bash
python -m unittest discover -s tests -v
```

## Citation

If you use this workflow, please cite the accompanying manuscript:

```bibtex
@unpublished{rafique2026soilmoisture,
  author = {Rafique, Hamza and Muhammad, Abubakr},
  title  = {Calibration and validation of field scale soil moisture estimates from an Energy Balance Model for the data-scarce Indus River Basin},
  year   = {2026},
  note   = {Manuscript submitted to Journal of Hydrology: Regional Studies}
}
```
