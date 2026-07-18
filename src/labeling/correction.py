import os
import shutil
import nibabel as nib
import numpy as np

unlabeled_path = "./dataset/unlabeled_dataset"
final_path = "./dataset/final_labeled_dataset"
out = "./dataset/final_labeled_dataset_2"

for patient_id in os.listdir(final_path):
    os.makedirs(os.path.join(out, patient_id), exist_ok=True)
    shutil.copy2(
        os.path.join(unlabeled_path, patient_id, f"subject_{int(patient_id):03d}_ADC.nii.gz"),
        os.path.join(out, patient_id, f"subject_{int(patient_id):03d}_ADC.nii.gz")
    )
    shutil.copy2(
        os.path.join(unlabeled_path, patient_id, f"subject_{int(patient_id):03d}_FLAIR.nii.gz"),
        os.path.join(out, patient_id, f"subject_{int(patient_id):03d}_FLAIR.nii.gz")
    )
    shutil.copy2(
        os.path.join(unlabeled_path, patient_id, f"subject_{int(patient_id):03d}_T1.nii.gz"),
        os.path.join(out, patient_id, f"subject_{int(patient_id):03d}_T1.nii.gz")
    )
    shutil.copy2(
        os.path.join(unlabeled_path, patient_id, f"subject_{int(patient_id):03d}_T2.nii.gz"),
        os.path.join(out, patient_id, f"subject_{int(patient_id):03d}_T2.nii.gz")
    )
    shutil.copy2(
        os.path.join(unlabeled_path, patient_id, f"subject_{int(patient_id):03d}_T1C.nii.gz"),
        os.path.join(out, patient_id, f"subject_{int(patient_id):03d}_T1C.nii.gz")
    )

    data = nib.load(os.path.join(final_path, patient_id, "Segmentations.nii.gz"))
    data = nib.as_closest_canonical(data)   # Converts to RAS
    img = data.get_fdata().astype(np.uint8)
    affine =data.affine
    nib.save(
        nib.Nifti1Image(img.astype(np.float32), affine),
        os.path.join(out, patient_id, "Segmentations.nii.gz")
    )
