import os
import shutil
from utils import copy_tif_files

def clean_subfolders(root_dir, keep_folder="Output_energy_balance"):
    """
    Removes all subfolders except one (keep_folder) from each main folder inside root_dir.
    """
    # Loop through main folders (A, B, C, ...)
    for main_folder in os.listdir(root_dir):
        main_path = os.path.join(root_dir, main_folder)

        # Skip if it's not a folder
        if not os.path.isdir(main_path):
            continue  

        # Loop through subfolders inside each main folder
        for subfolder in os.listdir(main_path):
            sub_path = os.path.join(main_path, subfolder)

            if os.path.isdir(sub_path) and subfolder != keep_folder:
                print(f"Deleting: {sub_path}")
                shutil.rmtree(sub_path)  # Delete the folder and its contents

    print("Cleanup complete ✅")

# # Example usage
# root_directory = r"D:\SEBAL\datasets\SEBAL_out\calibrations\Rms"  # <-- change this to your root path
# clean_subfolders(root_directory)

import os
import shutil

def delete_target_folders(root, target_name="Output_evapotranspiration"):
    deleted = []
    for dirpath, dirnames, _ in os.walk(root):
        if target_name in dirnames:
            full_path = os.path.join(dirpath, target_name)
            shutil.rmtree(full_path)
            deleted.append(full_path)
    return deleted


if __name__ == "__main__":
    # root_dir = r"D:\SEBAL\datasets\SEBAL_out\LBDC_exp\150039"

    # deleted_folders = delete_target_folders(root_dir)

    # print("\nDeleted folders:\n")
    # for f in deleted_folders:
    #     print(f)

    # print(f"\nTotal folders deleted: {len(deleted_folders)}")

    # Example usage
    ROW_PATH = '149039'
    source = fr'D:/SEBAL/datasets/SEBAL_out/LBDC_exp/upper/{ROW_PATH}/'
    destination = fr'D:/SEBAL/datasets/validation/LBDC_validations/rzsm/upper/{ROW_PATH}/'
    pattern = 'Root_zone_moisture'

    copy_tif_files(source, destination, pattern)