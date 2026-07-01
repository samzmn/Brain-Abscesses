import os
import shutil

def create_dataset(reg_dir, bet_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=False)
    for patient_id in os.listdir(reg_dir):
        reg_patient_dir = os.path.join(reg_dir, patient_id)
        bet_patient_file = os.path.join(bet_dir, f"subject_{int(patient_id):03d}_FLAIR_bet.nii.gz")
        if os.path.exists(reg_patient_dir) and os.path.exists(bet_patient_file):
            dest_patient_dir = os.path.join(output_dir, patient_id)
            if not os.path.exists(dest_patient_dir):
                os.makedirs(dest_patient_dir, exist_ok=False)
            for file_name in os.listdir(reg_patient_dir):
                src_file_path = os.path.join(reg_patient_dir, file_name)
                dest_file_path = os.path.join(dest_patient_dir, f"subject_{int(patient_id):03d}_{file_name.strip().split('_')[0]}.nii.gz")
                shutil.copy2(src_file_path, dest_file_path)
            shutil.copy2(bet_patient_file, os.path.join(dest_patient_dir, f"subject_{int(patient_id):03d}_brain_mask.nii.gz"))
            print(f"Copied data for patient {patient_id} to {dest_patient_dir}")
        else:
            print(f"Missing data for patient {patient_id}: reg_dir or bet_file does not exist.")
        
if __name__ == "__main__":
    reg_dir = "./dataset/registered"
    bet_dir = "./dataset/brain_masks"
    output_dir = "./dataset/unlabeled_dataset"
    create_dataset(reg_dir, bet_dir, output_dir)
