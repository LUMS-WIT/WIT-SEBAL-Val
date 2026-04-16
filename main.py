from workflows.validation_workflow import main as run_validation
from workflows.uncertainty_workflow import main as run_uncertainty

if __name__ == "__main__":
    run_validation()
    run_uncertainty()