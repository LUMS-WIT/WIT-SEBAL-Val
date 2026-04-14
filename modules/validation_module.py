import os
from pathlib import Path

import pandas as pd

from utils import (
    SoilMoistureData,
    SebalSoilMoistureData,
    remove_nan_entries,
    save_to_excel,
    save_metadata,
    save_to_plot,
    validations_gpi_adv,
    compute_statistics,
    plot_box_and_whiskers,
    plot_metric_with_ci,
    plot_paired,
)

from scaling import scaling, temporal_matching
from sms_calibration import sms_calibrations

from config import (
    OUTLIER_THRESHOLD,
    COMBINE_VALIDATIONS,
)


def build_validation_case_config(
    row_path,
    raster_stat,
    wit_sms_path,
    rescaling,
    save_plot,
    temporal_window,
):
    """
    Build all paths for one validation case.

    Adjust only this function if your folder structure changes.
    """
    raster_folder_path = Path(
        f"D:/SEBAL/datasets/validation/LBDC_validations/rzsm/{raster_stat}/{row_path}"
    )

    validation_folder = Path(
        f"./validations/{raster_stat}/{row_path}/tw_{temporal_window}"
    )

    images_folder = validation_folder / "images"
    metadata_file_path = validation_folder / f"metadata_{row_path}_{raster_stat}_tw_{temporal_window}.xlsx"
    output_file = Path(f"./validations/results/validations_{raster_stat}_{row_path}_tw_{temporal_window}.xlsx")
    plot_output_file = Path(f"./validations/results/plots_{raster_stat}_{row_path}_tw_{temporal_window}.png")

    validation_folder.mkdir(parents=True, exist_ok=True)
    images_folder.mkdir(parents=True, exist_ok=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    return {
        "row_path": row_path,
        "raster_stat": raster_stat,
        "wit_sms_path": wit_sms_path,
        "raster_folder_path": raster_folder_path,
        "validation_folder": validation_folder,
        "images_folder": images_folder,
        "metadata_file_path": metadata_file_path,
        "output_file": output_file,
        "plot_output_file": plot_output_file,
        "rescaling": rescaling,
        "save_plot": save_plot,
        "temporal_window": temporal_window,
    }


def generate_overlaps_for_case(case_cfg):
    """
    Generate overlap files for a single case.
    Equivalent to your legacy generate_overlaps() logic,
    but isolated from main execution.
    """
    soil_moisture_data = SoilMoistureData(case_cfg["wit_sms_path"])
    soil_moisture_data.read_data()

    metadata = soil_moisture_data.get_metadata()
    raster_data = SebalSoilMoistureData(
        str(case_cfg["raster_folder_path"]),
        pattern="Root_zone_moisture",
    )

    validation_metadata = []

    print(
        f"Performing temporal matching with temporal window "
        f"of {case_cfg['temporal_window']} days"
    )

    for _metadata in metadata:
        lat = float(_metadata["latitude"])
        lon = float(_metadata["longitude"])

        csv_dates, csv_values = soil_moisture_data.get_soil_moisture_by_location(lat, lon)
        raster_dates, raster_values = raster_data.get_data(lat, lon)

        if raster_dates is None or raster_values is None:
            continue

        csv_dates, csv_values = remove_nan_entries(csv_dates, csv_values)
        raster_dates, raster_values = remove_nan_entries(raster_dates, raster_values)

        ref_data = (csv_dates, csv_values)         # WIT SMS
        test_data = (raster_dates, raster_values)  # SEBAL

        ref_data = sms_calibrations(ref_data)

        if case_cfg["rescaling"]:
            test_data, _ = scaling(test_data, ref_data)
            csv_dates, csv_values = ref_data
            raster_dates, raster_values = test_data

        common_dates, ref_values, test_values, overlap_count = temporal_matching(
            test_data,
            ref_data,
            case_cfg["temporal_window"],
        )

        _metadata_copy = dict(_metadata)
        _metadata_copy["overlaps"] = overlap_count

        if overlap_count > 0:
            validation_metadata.append(_metadata_copy)

            data_to_save = list(zip(common_dates, ref_values, test_values))
            headers = ["Timestamp", "wit_sm", "sebal_sm"]

            file_name = (
                f"sebal_{case_cfg['row_path']}_{case_cfg['raster_stat']}"
                f"_witgpi_{_metadata['gpi']}_lat_{lat}_lon_{lon}.xlsx"
            )
            out_file = case_cfg["validation_folder"] / file_name
            save_to_excel(data_to_save, str(out_file), headers)

            if case_cfg["save_plot"]:
                img_name = (
                    f"sebal_{case_cfg['row_path']}_{case_cfg['raster_stat']}"
                    f"_witgpi_{_metadata['gpi']}_lat_{lat}_lon_{lon}.png"
                )
                img_file = case_cfg["images_folder"] / img_name
                save_to_plot(
                    csv_dates,
                    csv_values,
                    raster_dates,
                    raster_values,
                    lat,
                    lon,
                    str(img_file),
                )

    print("Total number of overlapping in-situ sensors:", len(validation_metadata))
    save_metadata(validation_metadata, str(case_cfg["metadata_file_path"]))


def run_validations_for_case(case_cfg):
    """
    Run site-level metrics + summary generation for one case.
    Equivalent to your legacy validations() block.
    """
    input_folder = case_cfg["validation_folder"]

    metrics_dict, num_of_obs, paired_values = validations_gpi_adv(
        str(input_folder),
        threshold=OUTLIER_THRESHOLD,
    )

    stats_results = compute_statistics(metrics_dict)

    print("------------- results for gpi based metrics ----------------")
    print("Number of Observations N:", num_of_obs)
    print(stats_results)

    plot_box_and_whiskers(metrics_dict, str(case_cfg["plot_output_file"]), False)
    plot_metric_with_ci(metrics_dict, metric="ubrmsd")
    plot_metric_with_ci(metrics_dict, metric="bias")

    df = pd.read_excel(case_cfg["metadata_file_path"])
    df["gpi"] = df["gpi"].astype(str)

    metrics_df = pd.DataFrame(metrics_dict)
    merged_df = pd.merge(df, metrics_df, on="gpi", how="left")

    summary_data = {
        "Metric": ["bias", "mse", "ubrmsd", "p_rho", "s_rho"],
        "mean": [
            stats_results["bias"]["mean"],
            stats_results["mse"]["mean"],
            stats_results["ubrmsd"]["mean"],
            stats_results["p_rho"]["mean"],
            stats_results["s_rho"]["mean"],
        ],
        "median": [
            stats_results["bias"]["median"],
            stats_results["mse"]["median"],
            stats_results["ubrmsd"]["median"],
            stats_results["p_rho"]["median"],
            stats_results["s_rho"]["median"],
        ],
        "IQR": [
            stats_results["bias"]["IQR"],
            stats_results["mse"]["IQR"],
            stats_results["ubrmsd"]["IQR"],
            stats_results["p_rho"]["IQR"],
            stats_results["s_rho"]["IQR"],
        ],
    }

    summary_df = pd.DataFrame(summary_data)

    observations_df = pd.DataFrame(
        {
            "Metric": ["Observations"],
            "mean": [num_of_obs],
            "median": [""],
            "IQR": [""],
        }
    )

    summary_df = pd.concat([observations_df, summary_df], ignore_index=True)

    with pd.ExcelWriter(case_cfg["output_file"], engine="xlsxwriter") as writer:
        if not COMBINE_VALIDATIONS:
            merged_df.to_excel(writer, sheet_name="MetaData", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

    print("Validations saved to", case_cfg["output_file"])
    plot_paired(paired_values, stats_results)