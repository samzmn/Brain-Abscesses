import json
from pathlib import Path

def create_json(
    dataset_root = Path("./dataset/final_labeled_dataset"),
    output_json = Path("./jsons/valid.json"),
    # Order of modalities expected by the model
    modalities = [
        "FLAIR",
        "T1C",
        "T1",
        "T2",
        # "ADC",
    ]
):
    valid_list = []

    # Sort folders numerically
    subject_dirs = sorted(
        [d for d in dataset_root.iterdir() if d.is_dir()],
        key=lambda x: int(x.name)
    )

    for subject_dir in subject_dirs:

        image_files = []
        missing = False

        for modality in modalities:
            matches = list(subject_dir.glob(f"*_{modality}.nii.gz"))

            if len(matches) != 1:
                print(f"Skipping {subject_dir.name}: missing {modality}")
                missing = True
                break

            # relative path
            image_files.append(str(matches[0].relative_to(dataset_root)).replace("\\", "/"))

        if missing:
            continue

        label_name = "Segmentations.nii.gz"
        label_path = subject_dir / label_name

        if not label_path.exists():
            print(f"Skipping {subject_dir.name}: missing label")
            continue

        valid_list.append({
            "image": image_files,
            "label": str(label_path.relative_to(dataset_root)).replace("\\", "/")
        })

    # -----------------------------------------------------------------------------

    json_dict = {
        "valid": valid_list[-2:-1],
    }

    with open(output_json, "w") as f:
        json.dump(json_dict, f, indent=4)

    print(f"Saved {len(valid_list)} Validation samples to {output_json}")
