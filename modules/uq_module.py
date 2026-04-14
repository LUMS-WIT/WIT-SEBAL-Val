from pathlib import Path

import pandas as pd

from utils import (
    sebal_uncertainty_analysis,
    compute_statistics,
    plot_uncertainty_boxplot,
    compute_coverage_probability,
    uncertainty_signal_ratio,
    plot_uncertainty_distribution,
    plot_gpi_uncertainty,
    plot_relative_uncertainty_vs_sm,
    plot_coverage_probability,
)


def build_uq_case_config(row_path):
    """
    Build all paths for one UQ case.

    Adjust this only if your folder structure changes.
    """
    base_dir_results = Path("./validations_UQ/results")
    base_dir_results.mkdir(parents=True, exist_ok=True)

    mean_dir = Path(f"./validations/mean/{row_path}/tw_0")
    lower_dir = Path(f"./validations/lower/{row_path}/tw_0")
    upper_dir = Path(f"./validations/upper/{row_path}/tw_0")
    combine_dir = Path(f"./validations_UQ/combined/{row_path}")
    combine_dir.mkdir(parents=True, exist_ok=True)

    output_excel = base_dir_results / f"SEBAL_uncertainty_summary_{row_path}.xlsx"

    return {
        "row_path": row_path,
        "mean_dir": mean_dir,
        "lower_dir": lower_dir,
        "upper_dir": upper_dir,
        "combine_dir": combine_dir,
        "output_excel": output_excel,
    }


def combine_uq_triplets(case_cfg):
    """
    Combine mean / lower / upper overlap files into one folder.
    Equivalent to your legacy combine_UQ() logic.
    """
    mean_dir = case_cfg["mean_dir"]
    lower_dir = case_cfg["lower_dir"]
    upper_dir = case_cfg["upper_dir"]
    combine_dir = case_cfg["combine_dir"]

    combine_dir.mkdir(parents=True, exist_ok=True)

    for mean_file in mean_dir.glob("*.xlsx"):
        fname = mean_file.name
        lower_file = lower_dir / fname
        upper_file = upper_dir / fname

        if not lower_file.exists() or not upper_file.exists():
            print(f"Skipping {fname} (missing lower/upper file)")
            continue

        df_mean = pd.read_excel(mean_file)
        df_lower = pd.read_excel(lower_file)
        df_upper = pd.read_excel(upper_file)

        if not (
            df_mean["Timestamp"].equals(df_lower["Timestamp"])
            and df_mean["Timestamp"].equals(df_upper["Timestamp"])
        ):
            print(f"Timestamp mismatch in {fname}, skipping")
            continue

        df_combined = df_mean.copy()
        df_combined["sebal_sm_l"] = df_lower["sebal_sm"]
        df_combined["sebal_sm_u"] = df_upper["sebal_sm"]

        if pd.api.types.is_datetime64_any_dtype(df_combined["Timestamp"]):
            df_combined["Timestamp"] = df_combined["Timestamp"].dt.date

        out_file = combine_dir / fname
        df_combined.to_excel(out_file, index=False)
        print(f"Saved combined UQ file: {out_file}")


def run_uq_summary_for_case(case_cfg):
    """
    Run the uncertainty summary analysis on combined files.
    Equivalent to the UQ block inside legacy validations().
    """
    input_folder = case_cfg["combine_dir"]

    uq_dict, uq_values = sebal_uncertainty_analysis(str(input_folder))
    uq_stats = compute_statistics(uq_dict)

    print("------------- SEBAL model uncertainty ----------------")
    print(uq_stats)

    plot_uncertainty_boxplot(uq_dict)

    coverage = compute_coverage_probability(uq_values)
    usr = uncertainty_signal_ratio(uq_values)

    print("Coverage Probability at 95% confidence interval:", coverage)
    print("Uncertainty Signal Ratio:", usr)

    uq_stats_df = pd.DataFrame(uq_stats).T
    uq_stats_df.index.name = "Metric"
    uq_stats_df["USR"] = usr
    uq_stats_df["CoverageProbability95"] = coverage

    uq_stats_df.to_excel(case_cfg["output_excel"])

    plot_uncertainty_distribution(uq_values)
    plot_gpi_uncertainty(uq_values)
    plot_relative_uncertainty_vs_sm(uq_values)
    plot_coverage_probability(uq_values)

    print("UQ summary saved to", case_cfg["output_excel"])