import os
import shutil


def copy_all_flair_for_bet_extraction(src_dir, dest_dir):
    for patient_id in os.listdir(src_dir):
        flair_path = os.path.join(src_dir, patient_id, "FLAIR.nii.gz")
        if os.path.exists(flair_path):
            dest_flair_path = os.path.join(dest_dir, f"subject_{int(patient_id):03d}_FLAIR.nii.gz")
            if not os.path.exists(dest_flair_path):
                shutil.copy2(flair_path, dest_flair_path)
                print(f"Copied {flair_path} to {dest_flair_path}")
            else:
                print(f"File already exists: {dest_flair_path}")
        else:
            print(f"FLAIR image not found for patient {patient_id} in {src_dir}")


if __name__ == "__main__":
    flair_src_dir = "./dataset/registered"
    flair_dest_dir = "./dataset/brain_mask_input"
    copy_all_flair_for_bet_extraction(flair_src_dir, flair_dest_dir)
