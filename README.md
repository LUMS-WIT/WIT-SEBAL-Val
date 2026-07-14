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

