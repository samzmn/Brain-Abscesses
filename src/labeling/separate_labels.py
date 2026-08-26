import os
import shutil
from pathlib import Path
import nibabel as nib
import numpy as np

def separate_brain_lesion(
        root_dir = "./dataset/final_labeled_dataset",
        out_dir = "./dataset/",
):
    for patient_id in os.listdir(root_dir):
        os.makedirs(os.path.join(out_dir, "brain_labeled_dataset", patient_id), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "lesion_labeled_dataset", patient_id), exist_ok=True)
        for file in os.listdir(os.path.join(root_dir, patient_id)):
            if file.startswith("Segmentation"):
                seg_data = nib.load(os.path.join(root_dir, patient_id, file))
                seg, affine = seg_data.get_fdata().astype(np.uint8), seg_data.affine
                # brain_seg = np.where(seg == 1, seg, 0)
                # print(np.unique(brain_seg), np.count_nonzero(brain_seg == 1))
                lesion_seg = np.where(seg == 1, 0, seg)
                # print(np.unique(lesion_seg), lesion_seg.shape, np.count_nonzero(lesion_seg == 2), np.count_nonzero(lesion_seg == 3), np.count_nonzero(lesion_seg == 4))
                lesion_seg = np.where(lesion_seg == 2, 1, lesion_seg)
                lesion_seg = np.where(lesion_seg == 3, 2, lesion_seg)
                lesion_seg = np.where(lesion_seg == 4, 3, lesion_seg)
                # print(np.unique(lesion_seg), lesion_seg.shape, np.count_nonzero(lesion_seg == 1), np.count_nonzero(lesion_seg == 2), np.count_nonzero(lesion_seg == 3))
                # nib.save(
                #     nib.Nifti1Image(brain_seg, affine),
                #     os.path.join(out_dir, "brain_labeled_dataset", patient_id, "Segmentations.nii.gz")
                # )
                shutil.copy2(
                    os.path.join("./dataset/labeled_dataset", patient_id, f"subject_{int(patient_id):03d}_brain_mask.nii.gz"),
                    os.path.join(out_dir, "brain_labeled_dataset", patient_id, "Segmentations.nii.gz")
                )
                nib.save(
                    nib.Nifti1Image(lesion_seg, affine),
                    os.path.join(out_dir, "lesion_labeled_dataset", patient_id, "Segmentations.nii.gz")
                )
            else:
                shutil.copy2(
                    os.path.join(root_dir, patient_id, file),
                    os.path.join(out_dir, "brain_labeled_dataset", patient_id, file)
                )
                shutil.copy2(
                    os.path.join(root_dir, patient_id, file),
                    os.path.join(out_dir, "lesion_labeled_dataset", patient_id, file)
                )


if __name__ == "__main__":
    separate_brain_lesion()