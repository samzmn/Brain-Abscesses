import os
import nibabel as nib
import numpy as np

def replace_missing_t1c(dataset_root):
    for patient_id in os.listdir(dataset_root):
        patient_dir = os.path.join(dataset_root, patient_id)
        if not os.path.isdir(patient_dir):
            continue

        t1c_path = os.path.join(patient_dir, f"subject_{int(patient_id):03d}_T1C.nii.gz")
        t1_path = os.path.join(patient_dir, f"subject_{int(patient_id):03d}_T1.nii.gz")

        if not os.path.exists(t1c_path) and os.path.exists(t1_path):
            print(f"Replacing missing T1C with T1 for patient {patient_id}")
            data = nib.load(t1_path)
            img = data.get_fdata().astype(np.float32)
            affine =data.affine
            nib.save(nib.Nifti1Image(np.zeros(img.shape, dtype=np.float32), affine), t1c_path)

if __name__ == "__main__":
    dataset_root = "./dataset/unlabeled_dataset"
    replace_missing_t1c(dataset_root)
