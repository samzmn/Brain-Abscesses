import os
import shutil
import time
import datetime
import ants
from typing import Tuple


def register_seq_to_seq(seq: ants.ANTsImage, target_seq: ants.ANTsImage, out_path: str, transform_type: str="Affine"):
    start_time = time.time()
    reg = ants.registration(
        fixed=target_seq,
        moving=seq,
        type_of_transform=transform_type
    )
    registered = reg["warpedmovout"]
    ants.image_write(registered, out_path)

    end_time = time.time() - start_time
    end_time = str(datetime.timedelta(seconds=int(end_time)))
    print(f"registration completed in {end_time}.")


def register_seq_to_mni(patient_id: str, 
                            in_path: str="./dataset/nifti", 
                            out_path: str="./dataset/registered"):
    t1 = ants.image_read(f"{in_path}/{patient_id}/T1.nii.gz")
    t2 = ants.image_read(f"{in_path}/{patient_id}/T2.nii.gz")
    adc = ants.image_read(f"{in_path}/{patient_id}/ADC.nii.gz")
    flair = ants.image_read(f"{in_path}/{patient_id}/FLAIR.nii.gz")
    t1c = None
    try:
        t1c = ants.image_read(f"{in_path}/{patient_id}/T1C.nii.gz")
    except:
        print("     T1C does not exists!")

    sequences = {"T1":t1, "T2":t2, "ADC":adc, "FLAIR":flair, "T1C":t1c}
    fixed = ants.image_read(f"dataset/average305_t1_tal_lin.nii")

    for seq_name, seq in sequences.items():
        print(f"Registering {seq_name}...")
        register_seq_to_seq(seq, fixed,
            out_path=f"{out_path}/{patient_id}/{seq_name}_registered_to_mni.nii.gz"
        )


def register_all_seq_to_mni(in_path: str="./dataset/nifti", out_path: str="./dataset/registered", id_range: Tuple[int,int]=(240,240)):
    start_time = time.time()
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)

    for patient_id in range(*id_range):
        if not os.path.exists(f'{in_path}/{patient_id}'):
            continue
        print(f"\nProcessing Patient {patient_id}")
        if not os.path.exists(f"{out_path}/{patient_id}"):
            os.makedirs(f"{out_path}/{patient_id}")
        register_seq_to_mni(patient_id, in_path, out_path)

    end_time = time.time() - start_time
    end_time = str(datetime.timedelta(seconds=int(end_time)))
    print(f"Total registration time: {end_time}.")


if __name__ == "__main__":
    in_path = "./dataset/valid_nifti"
    out_path = "./dataset/registered_mni"
    register_all_seq_to_mni(in_path, out_path, id_range=(3, 4))
