# SEBAL Soil Moisture estimates validation 

This repository contains the code for Validation of SEBAL soil moisture estimates using WIT SMS Network over Central Punjab.

## Installation and Setup
To run the models and scripts in this repository, ensure your system meets the following requirements:

### Prerequisites
- Python 3.8 or higher
- Input Dataset(s)
  - Soil moisture raster maps
  - WITSMS-Network Dataset

1. **Clone the Repository**
   ```bash
   git clone https://github.com/LUMS-WIT/WIT-SEBAL-Val.git
   cd WIT-SEBAL-Val

2. **Install Dependencies using conda**
   ```bash
   conda env create -f requirements.yml
   conda activate sebal-val

## Usage

Run the project entry point:

```bash
python main.py
```

`main.py` currently does not accept command-line arguments.  
Workflow selection is controlled directly in the file by enabling/disabling function calls.

---

## Workflow Selection (`main.py`)

In `main.py`, uncomment the workflow you want to run and keep others commented (unless you intentionally want sequential execution):

- `run_validation()`
- `run_validation_scaling()` (currently enabled)
- `run_uncertainty()`
- `run_endpoint_diagnostics_workflow()`

Example pattern:

```python
if __name__ == "__main__":
    run_validation()
    run_uncertainty()
    run_endpoint_diagnostics_workflow()
```

---

## Five-fold validation with independent rescaling

`python main.py` currently runs `run_validation_scaling()`. To run a different
workflow, change only the calls in `main.py`. The other workflows and their
configuration settings retain their existing behavior.

The `SCALING_*` settings in `config.py` configure the new workflow separately.
It reads fresh raw rasters and sensor CSVs, looks up sensors by GPI, applies the
existing sensor quality-control and temporal-matching rules, and produces both
unscaled and scaled results regardless of the legacy `RESCALING` switch.
`ROW_PATHS`, `VALIDATION_MEMBER`, and `TEMPORAL_WIN` select the inputs as before.

The default cohort is explicitly listed in `validation_inputs/five_fold_cohort.json`.
It reproduces the agreed historical cohort: 66 GPIs, 61 exact-coordinate
locations and currently 445 matched pairs. This cohort was previously selected
using correlation scores; five-fold calibration does not remove that inherited
selection bias. Set `SCALING_COHORT_FILE = None` to use all GPIs with matched data.
There is no additional filtering by correlation performance.

Five folds are equal-location-count geographic strips along the leading
coordinate principal axis. Co-located GPIs and all dates/satellite rows of a GPI
stay together. For each fold, the other four provide the pooled training mean
and population standard deviation. Their affine transform is applied only to
the held-out fold. Every pair is validated once. There is no spatial buffer.

Results are written to `outputs/validation_scaling/`:

- `scaled/` and `unscaled/`: the usual `validation_points/<member>/<row>_<tw>/`,
  metadata workbooks, `results/validations_<row>_tw_<tw>.xlsx`, combined
  `results/validations_tw_<tw>.xlsx`, and `figs/` with paired scatters, metric box
  plots and bias/ubRMSD CI plots. `MetaData` and `Summary` sheets retain their
  original names and columns. All aggregate plots are saved automatically.
- `figs/error_metric_boxplots.png` (also PDF/SVG): paired per-GPI bias, MSE and
  ubRMSD distributions, with median labels and no title or footer text.
- `figs/five_fold_results.png` (also PDF/SVG): geographic folds and fold RMSEs.
- Raw pairs, held-out predictions, fold assignments, per-fold/per-GPI metrics,
  `summary.json`, `REPORT.md`, and an input/protocol manifest.

The combined workbook contains one record per GPI/satellite-row series, without
the legacy many-to-many GPI merge. Currently these are 73 series, representing
66 GPIs and 445 pairs. The comparison box plot combines each GPI's satellite
rows, giving 66 values per method. Workbook Summary correlations retain the
legacy Fisher-z mean; the JSON report also distinguishes arithmetic site means
and pooled held-out correlations. Do not interchange these summaries.

Bias CIs in the new workflow use the actual bias as their centre, correcting a
wrong argument in the legacy exporter without changing other workflows. The
existing parametric CI assumptions are retained; they do not account for
calibration uncertainty or spatial/temporal dependence.

`SCALING_SHOW_PLOTS` controls display; `SCALING_SAVE_SITE_PLOTS` additionally
saves per-series matched time-series plots. A complete run replaces only this
workflow's own managed outputs. Failed runs preserve the last successful run;
manually added files in its output directory prevent replacement.

Run the focused checks with:

```bash
MPLBACKEND=Agg python -m unittest discover -s tests -v
```


## Citation
If you use this project in your research, please cite:
## Citation

```bibtex
@preprint{rafique2026soilmoisture,
  author       = {Hamza Rafique and Abubakr Muhammad},
  title        = {Calibration and validation of field scale soil moisture estimates from an Energy Balance Model for the data-scarce Indus River Basin},
  year         = {2026},
  note         = {Preprint, submitted to Journal of Hydrology: Regional Studies},
}
