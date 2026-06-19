import os
import time
import datetime
import ants

patient_id = "240"
in_path = "./dataset/nifti"
out_path = "./dataset/registered"

def register_all_seq_to_flair(in_path: str="./dataset/nifti", out_path="./dataset/registered", id_range=(240,240)):
    start_time = time.time()
    for patient_id in range(*id_range):
        if not os.path.exists(f'{in_path}/{patient_id}'):
            continue
        # Register T2 to FLAIR -------------------------------------------------
        t2_time = time.time()
        print("Registering T2 to FLAIR...")

        fixed = ants.image_read(f"{in_path}/{patient_id}/FLAIR.nii.gz")
        moving = ants.image_read(f"{in_path}/{patient_id}/T2.nii.gz")

        reg = ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform="Affine"
        )

        registered = reg["warpedmovout"]

        ants.image_write(
            registered,
            f"{out_path}/{patient_id}/T2_registered_to_FLAIR.nii.gz"
        )

        t2_time = time.time() - t2_time
        t2_time = str(datetime.timedelta(seconds=int(t2_time)))
        print(f"T2 to FLAIR registration completed in {t2_time}.")

        # Register T1 to FLAIR --------------------------------------------------
        t1_time = time.time()
        print("Registering T1 to FLAIR...")

        fixed = ants.image_read(f"{in_path}/{patient_id}/FLAIR.nii.gz")
        moving = ants.image_read(f"{in_path}/{patient_id}/T1.nii.gz")

        reg = ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform="Affine"
        )

        ants.image_write(
            reg["warpedmovout"],
            f"{out_path}/{patient_id}/T1_registered_to_FLAIR.nii.gz"
        )

        t1_time = time.time() - t1_time
        t1_time = str(datetime.timedelta(seconds=int(t1_time)))
        print(f"T1 to FLAIR registration completed in {t1_time}.")

        # Register ADC to FLAIR ------------------------------------------------------
        adc_time = time.time()
        print("Registering ADC to FLAIR...")

        fixed = ants.image_read(f"{in_path}/{patient_id}/FLAIR.nii.gz")
        moving = ants.image_read(f"{in_path}/{patient_id}/dADC.nii.gz")

        reg = ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform="Affine"
        )

        ants.image_write(
            reg["warpedmovout"],
            f"{out_path}/{patient_id}/ADC_registered_to_FLAIR.nii.gz"
        )

        adc_time = time.time() - adc_time
        adc_time = str(datetime.timedelta(seconds=int(adc_time)))
        print(f"ADC to FLAIR registration completed in {adc_time}.")

        end_time = time.time() - start_time
        end_time = str(datetime.timedelta(seconds=int(end_time)))
        print(f"Total registration time: {end_time}.")
