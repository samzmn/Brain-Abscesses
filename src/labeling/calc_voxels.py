import os
import numpy as np
import nibabel as nib
from openpyxl import Workbook

dataset_dir = "./dataset/final_labeled_dataset"
out_excel = "./dataset/voxel_counts_auto.xlsx"


# ==========================
# Configuration
# ==========================

# Label definitions
LABELS = {
    1: "Brain Voxels",
    2: "Edema Voxels",
    3: "Abscess Voxels",
    4: "Ring Voxels",
}

# ==========================
# Create Excel workbook
# ==========================
wb = Workbook()
ws = wb.active
ws.title = "Voxel Counts"

# Header
headers = [
    "Patient ID",
    "Brain Voxels",
    "Edema Voxels",
    "Abscess Voxels",
    "Ring Voxels",
    "Total Labeled Voxels",
    "Physical Voxel Volume"
]
ws.append(headers)

# ==========================
# Iterate over patients
# ==========================
patient_folders = sorted(
    [
        f for f in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, f))
    ],
    key=lambda x: int(x) if x.isdigit() else x
)

for patient_id in patient_folders:

    seg_path = os.path.join(dataset_dir, patient_id, "Segmentations.nii.gz")

    if not os.path.exists(seg_path):
        print(f"Skipping {patient_id}: Segmentations.nii.gz not found.")
        continue

    # Load segmentation
    seg = nib.load(seg_path)
    seg_data = np.asarray(seg.get_fdata(), dtype=np.uint8)

    # Count voxels
    counts = {
        label: int(np.count_nonzero(seg_data == label))
        for label in LABELS
    }

    total = sum(counts.values())

    voxel_volume_mm3 = np.prod(seg.header.get_zooms()[:3])

    brain_volume_ml = counts[1] * voxel_volume_mm3 / 1000
    edema_volume_ml = counts[2] * voxel_volume_mm3 / 1000
    abscess_volume_ml = counts[3] * voxel_volume_mm3 / 1000
    ring_volume_ml = counts[4] * voxel_volume_mm3 / 1000

    # Write row
    ws.append([
        patient_id,
        counts[1],
        counts[2],
        counts[3],
        counts[4],
        total,
        voxel_volume_mm3
    ])

    print(f"Processed patient {patient_id}")

# ==========================
# Save Excel
# ==========================
wb.save(out_excel)

print(f"\nSaved results to {out_excel}")