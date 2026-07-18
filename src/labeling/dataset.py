import os
import shutil
import nibabel as nib
import numpy as np

def create_unlabeled_dataset(reg_dir, bet_dir, output_dir, copy_only=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    for patient_id in os.listdir(reg_dir):
        reg_patient_dir = os.path.join(reg_dir, patient_id)
        bet_patient_file = os.path.join(bet_dir, f"subject_{int(patient_id):03d}_FLAIR_bet.nii.gz")
        if os.path.exists(reg_patient_dir) and os.path.exists(bet_patient_file):
            dest_patient_dir = os.path.join(output_dir, patient_id)
            if not os.path.exists(dest_patient_dir):
                os.makedirs(dest_patient_dir, exist_ok=True)
            for file_name in os.listdir(reg_patient_dir):
                src_file_path = os.path.join(reg_patient_dir, file_name)
                dest_file_path = os.path.join(dest_patient_dir, f"subject_{int(patient_id):03d}_{file_name.strip().split('.')[0].split('_')[0]}.nii.gz")

                if copy_only:
                    shutil.copy2(src_file_path, dest_file_path)
                else:
                    data = nib.load(src_file_path)
                    data = nib.as_closest_canonical(data)   # Converts to RAS
                    img = data.get_fdata().astype(np.float32)
                    affine =data.affine
                    nib.save(
                        nib.Nifti1Image(img.astype(np.float32), affine),
                        dest_file_path
                    )
            if copy_only:
                shutil.copy2(bet_patient_file, os.path.join(dest_patient_dir, f"subject_{int(patient_id):03d}_brain_mask.nii.gz"))
            else:
                data = nib.load(bet_patient_file)
                data = nib.as_closest_canonical(data)   # Converts to RAS
                img = data.get_fdata().astype(np.uint8)
                affine =data.affine
                nib.save(
                    nib.Nifti1Image(img.astype(np.float32), affine),
                    os.path.join(dest_patient_dir, f"subject_{int(patient_id):03d}_brain_mask.nii.gz")
                )
            print(f"processed data for patient {patient_id} to {dest_patient_dir}")
        else:
            print(f"Missing data for patient {patient_id}: reg_dir or bet_file does not exist.")
        
if __name__ == "__main__":
    reg_dir = "./dataset/registered"
    bet_dir = "./dataset/brain_masks"
    output_dir = "./dataset/unlabeled_dataset"
    create_unlabeled_dataset(reg_dir, bet_dir, output_dir)
