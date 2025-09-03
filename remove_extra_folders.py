import os
import shutil

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

# Example usage
root_directory = r"D:\SEBAL\datasets\SEBAL_out\calibrations\Rms"  # <-- change this to your root path
clean_subfolders(root_directory)
