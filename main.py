from workflows.run_uq import run_uq_workflow
from workflows.run_validation import run_validation_workflow

if __name__ == "__main__":
    if RUN_UQ:
        run_uq_workflow()
    if RUN_VALIDATION:
        run_validation_workflow()