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
    root_dir = r"D:\SEBAL\datasets\SEBAL_out\LBDC_exp\150039"

    deleted_folders = delete_target_folders(root_dir)

    print("\nDeleted folders:\n")
    for f in deleted_folders:
        print(f)

    print(f"\nTotal folders deleted: {len(deleted_folders)}")

