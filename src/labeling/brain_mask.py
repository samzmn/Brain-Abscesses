import os
import time
import datetime
import nibabel
import numpy as np
import ants
import HD_BET

def create_brain_mask(image_path: str, threshold: float = 0.):
    # Load the image using nibabel
    img = nibabel.load(image_path)
    data = img.get_fdata()

    # Create a brain mask by thresholding the image
    # brain_mask = data > (threshold * np.max(data))
    brain_mask = np.where(data > (threshold * np.max(data)), 1, 0)
    # Save the brain mask as a new NIfTI file
    brain_mask_img = nibabel.Nifti1Image(brain_mask.astype(np.uint8), img.affine)
    nibabel.save(brain_mask_img, os.path.dirname(image_path) + "/brain_mask.nii.gz")

def register_brain_to_flair(brain_path, flair_path, out_path):
    start_time = time.time()
    print("Registering Brian to FLAIR...")

    fixed = ants.image_read(flair_path)
    moving = ants.image_read(brain_path)

    reg = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform="Affine"
    )

    registered = reg["warpedmovout"]
    
    ants.image_write(
        registered,
        out_path
    )

    end_time = time.time() - start_time
    end_time = str(datetime.timedelta(seconds=int(end_time)))
    print(f"Total registration time: {end_time}.")

def main(brain_path: str="./dataset/masked", out_path="./dataset/registered", id_range=(240,240)):
    for patient_id in range(*id_range):
        if not os.path.exists(f'{brain_path}/{patient_id}'):
            continue

if __name__ == "__main__":
    # Example usage
    brain_path = "./dataset/masked/240/FLAIR_brain.nii.gz"
    flair_path = "./dataset/registered/240/FLAIR.nii.gz"
    registered_path = os.path.dirname(brain_path) + "/FLAIR_brain_registered.nii.gz"
    out_path = os.path.dirname(brain_path) + "/brain_mask.nii.gz"

    register_brain_to_flair(brain_path, flair_path, out_path=registered_path)
    create_brain_mask(registered_path, threshold=0.)
