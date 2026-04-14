from workflows.validation_workflow import run_validation
from workflows.uq_workflow import run_uq
from config import RUN_UQ, RUN_VALIDATION


def main():
    
    if RUN_UQ:
        print("\n=============================================")
        print("Running uncertainty quantification workflow")
        print("=============================================")
        run_uq()

    if RUN_VALIDATION:
        print("\n=============================================")
        print("Running validation workflow")
        print("=============================================")
        run_validation()


if __name__ == "__main__":
    main()