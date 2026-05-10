from workflows.validation_workflow import main as run_validation
from workflows.uncertainty_workflow import main as run_uncertainty
from workflows.volatility_workflow import main as run_volatility
from workflows.endpoint_workflow import main as run_endpoint_diagnostics_workflow

if __name__ == "__main__":
    # run_validation()
    # run_uncertainty()
    # run_volatility()
    run_endpoint_diagnostics_workflow()