import os
import time
import datetime
import ants

patient_id = "240"
in_path = "./dataset/nifti"
out_path = "./dataset/registered"

def register_all_seq_to_one(patient_id: str, 
                            in_path: str="./dataset/nifti", 
                            out_path="./dataset/registered", 
                            fixed_seq="FLAIR"):
    # Register T2 -------------------------------------------------
    if fixed_seq != "T2":
        t2_time = time.time()
        print("Registering T2...")

        fixed = ants.image_read(f"{in_path}/{patient_id}/{fixed_seq}.nii.gz")
        moving = ants.image_read(f"{in_path}/{patient_id}/T2.nii.gz")

        reg = ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform="Affine"
        )

        registered = reg["warpedmovout"]

        ants.image_write(
            registered,
            f"{out_path}/{patient_id}/T2_registered_to_{fixed_seq}.nii.gz"
        )

        t2_time = time.time() - t2_time
        t2_time = str(datetime.timedelta(seconds=int(t2_time)))
        print(f"T2 to {fixed_seq} registration completed in {t2_time}.")

    # Register T1 --------------------------------------------------
    if fixed_seq != "T1":
        t1_time = time.time()
        print("Registering T1...")

        fixed = ants.image_read(f"{in_path}/{patient_id}/{fixed_seq}.nii.gz")
        moving = ants.image_read(f"{in_path}/{patient_id}/T1.nii.gz")

        reg = ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform="Affine"
        )

        ants.image_write(
            reg["warpedmovout"],
            f"{out_path}/{patient_id}/T1_registered_to_{fixed_seq}.nii.gz"
        )

        t1_time = time.time() - t1_time
        t1_time = str(datetime.timedelta(seconds=int(t1_time)))
        print(f"T1 to {fixed_seq} registration completed in {t1_time}.")

    # Register ADC ------------------------------------------------------
    if fixed_seq != "ADC":
        adc_time = time.time()
        print("Registering ADC...")

        fixed = ants.image_read(f"{in_path}/{patient_id}/{fixed_seq}.nii.gz")
        moving = ants.image_read(f"{in_path}/{patient_id}/ADC.nii.gz")

        reg = ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform="Affine"
        )

        ants.image_write(
            reg["warpedmovout"],
            f"{out_path}/{patient_id}/ADC_registered_to_{fixed_seq}.nii.gz"
        )

        adc_time = time.time() - adc_time
        adc_time = str(datetime.timedelta(seconds=int(adc_time)))
        print(f"ADC to {fixed_seq} registration completed in {adc_time}.")

    # Register FALIR ------------------------------------------------------
    if fixed_seq != "FLAIR":
        flair_time = time.time()
        print("Registering FLAIR...")

        fixed = ants.image_read(f"{in_path}/{patient_id}/{fixed_seq}.nii.gz")
        moving = ants.image_read(f"{in_path}/{patient_id}/FLAIR.nii.gz")

        reg = ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform="Affine"
        )

        ants.image_write(
            reg["warpedmovout"],
            f"{out_path}/{patient_id}/FLAIR_registered_to_{fixed_seq}.nii.gz"
        )

        flair_time = time.time() - flair_time
        flair_time = str(datetime.timedelta(seconds=int(flair_time)))
        print(f"FLAIR to {fixed_seq} registration completed in {flair_time}.")

    # Register T1C ------------------------------------------------------
    if fixed_seq != "T1C":
        t1c_time = time.time()
        print("Registering FLAIR...")

        fixed = ants.image_read(f"{in_path}/{patient_id}/{fixed_seq}.nii.gz")
        moving = ants.image_read(f"{in_path}/{patient_id}/T1C.nii.gz")

        reg = ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform="Affine"
        )

        ants.image_write(
            reg["warpedmovout"],
            f"{out_path}/{patient_id}/T1C_registered_to_{fixed_seq}.nii.gz"
        )

        t1c_time = time.time() - t1c_time
        t1c_time = str(datetime.timedelta(seconds=int(t1c_time)))
        print(f"T1C to {fixed_seq} registration completed in {t1c_time}.")

def register_all_seq_to_flair(in_path: str="./dataset/nifti", out_path="./dataset/registered", id_range=(240,240)):
    start_time = time.time()
    for patient_id in range(*id_range):
        if not os.path.exists(f'{in_path}/{patient_id}'):
            continue
        register_all_seq_to_one(patient_id, in_path, out_path, "FLAIR")

    end_time = time.time() - start_time
    end_time = str(datetime.timedelta(seconds=int(end_time)))
    print(f"Total registration time: {end_time}.")

if __name__ == "__main__":
    register_all_seq_to_one("237", "./dataset/nifti", "./dataset/registered", fixed_seq="T2")
