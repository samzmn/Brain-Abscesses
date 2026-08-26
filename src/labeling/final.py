import os
import shutil


def copy_all(
        src_dir="./dataset/final_labeled_dataset", 
        preds_dir="./outputs/test_results",
        dest_dir="./dataset/final_1"
):
    os.makedirs(dest_dir, exist_ok=True)
    for patient_id in os.listdir(src_dir):
        os.makedirs(os.path.join(dest_dir, patient_id), exist_ok=True)
        for file in os.listdir(os.path.join(src_dir, patient_id)):
            shutil.copy2(os.path.join(src_dir, patient_id, file), os.path.join(dest_dir, patient_id, file))
        shutil.copy2(os.path.join(preds_dir, f"subject_{patient_id}_pred_out.nii.gz"), os.path.join(dest_dir, patient_id, f"subject_{patient_id}_pred_out.nii.gz"))

if __name__ == "__main__":
    copy_all()
