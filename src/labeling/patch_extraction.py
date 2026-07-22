import os
import random
from pathlib import Path
from typing import Any, Union, List, Tuple, Dict
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

import train_json
import valid_json

class SpatialPad:
    def __init__(
        self,
        spatial_size,
        mode="constant",
        constant_values=0,
    ):
        self.spatial_size = tuple(spatial_size)
        self.mode = mode
        self.constant_values = constant_values

    def _compute_pad_width(self, spatial_shape):
        """
        Compute symmetric padding.

        Example:
            current = 70
            target  = 96

            total_pad = 26
            before = 13
            after = 13
        """

        pad_width = []

        for current, target in zip(
            spatial_shape,
            self.spatial_size,
        ):
            if current >= target:
                pad_width.append((0, 0))
                continue

            total_pad = target - current

            before = total_pad // 2
            after = total_pad - before

            pad_width.append((before, after))

        return pad_width

    def _pad(self, img):
        if img.ndim == 3:
            # (H,W,D)
            spatial_shape = img.shape

            pad_width = self._compute_pad_width(
                spatial_shape
            )

        else:
            raise ValueError(
                f"Unsupported shape {img.shape}"
            )

        padded = np.pad(
            img,
            pad_width=pad_width,
            mode=self.mode,
            constant_values=self.constant_values,
        )
    
        return padded, pad_width

    def __call__(self, data, affine):
        padded, pad_width = self._pad(data)

        new_affine = affine.copy()
        pad_before = np.array([p[0] for p in pad_width], dtype=np.float64)
        new_affine[:3, 3] -= new_affine[:3, :3] @ pad_before

        return padded, new_affine


def data_files(basedir) -> Dict[str, Dict[str, Union[str, Dict[str, str]]]]:
    patients = dict()
    for patient_id in os.listdir(basedir):
        if not os.path.isdir(os.path.join(basedir, patient_id)):
            raise ValueError(f"Expected {patient_id} to be a directory")
        patient = dict()
        modalities = dict()
        for file_name in os.listdir(os.path.join(basedir, patient_id)):
            if "Segmentation" in file_name:
                patient["label"] = os.path.join(basedir, patient_id, file_name)
            elif "T1." in file_name:
                modalities["t1"] = os.path.join(basedir, patient_id, file_name)
            elif "T2" in file_name:
                modalities["t2"] = os.path.join(basedir, patient_id, file_name)
            elif "T1C" in file_name:
                modalities["t1c"] = os.path.join(basedir, patient_id, file_name)
            elif "FLAIR" in file_name:
                modalities["flair"] = os.path.join(basedir, patient_id, file_name)
            elif "ADC" in file_name:
                modalities["adc"] = os.path.join(basedir, patient_id, file_name)
            
        patient["modalities"] = modalities

        patients[patient_id] = patient

    return patients

def data_read(path) -> Tuple[np.array, np.array]:
    data = nib.load(path)
    data = nib.as_closest_canonical(data)   # Converts to RAS
    img = data.get_fdata().astype(np.float32)
    affine = data.affine
    # print(f"{path}", nib.aff2axcodes(affine))
    return img, affine

def save_patch(
    save_root: str,
    subject_id: str,
    patch_index: int,
    modalities: Dict[str, np.array],
    label: np.array,
    affines: Dict[str, np.array],
):
    subject_id = int(subject_id)
    patch_dir = Path(save_root) / f"subject{subject_id:03d}_patch{patch_index:03d}"
    patch_dir.mkdir(parents=True, exist_ok=True)

    for name, volume in modalities.items():
        nib.save(
            nib.Nifti1Image(volume.astype(np.float32), affines[name]),
            patch_dir / f"subject{subject_id:03d}_patch{patch_index:03d}_{name.upper()}.nii.gz",
        )

    nib.save(
        nib.Nifti1Image(label.astype(np.uint8), affines["label"]),
        patch_dir / f"subject{subject_id:03d}_patch{patch_index:03d}_label.nii.gz",
    )

def keep_patch(label, brain_keep_treshold=0.15, brain_keep_prob=0.25):
    total = label.size

    brain = np.count_nonzero(label == 1.)
    edema = np.count_nonzero(label == 2.)
    abscess = np.count_nonzero(label == 3.)
    ring = np.count_nonzero(label == 4.)

    brain_ratio = brain / total
    edema_ratio = edema / total
    abscess_ratio = abscess / total
    ring_ratio = ring / total

    lesion_ratio = edema_ratio + abscess_ratio + ring_ratio

    # Always keep lesion patches
    if lesion_ratio > 0.:
        return True

    # Keep patches containing meaningful brain
    if brain_ratio > brain_keep_treshold:
        return random.random() < brain_keep_prob

    # Discard nearly empty patches
    return False

def compute_patch_starts(length, roi, stride):
    if length <= roi:
        return [0]

    starts = []
    pos = 0

    while pos + roi < length:
        starts.append(pos)
        pos += stride

    starts.append(length - roi)

    return sorted(set(starts))

def extract_patches(
        data_dir,
        output_dir,
        roi_x,
        roi_y,
        roi_z,
    ):
    patients = data_files(basedir=data_dir)
    spatial_pad = SpatialPad(spatial_size=[roi_x, roi_y, roi_z])

    stride = (
        roi_x // 2,
        roi_y // 2,
        roi_z // 2,
    )

    for patient_id, patient in patients.items():
        print(f"Processing subject {patient_id}")
        modalities = dict()
        affines = dict()

        label_data, label_affine = data_read(patient["label"])
        label, label_affine = spatial_pad(label_data.astype(np.uint8), label_affine)
        affines["label"] = label_affine

        for modality_name, modality_path in patient["modalities"].items():
            m, a = data_read(modality_path)
            # if modality_name == "flair":
            #     k = 8
            #     plt.imshow(m[:, :, k], cmap="gray")
            #     plt.contour(label_data[:, :, k], colors="r")
            #     plt.show()
            assert m.shape == label_data.shape, f"raw image and label shape mismatch => {m.shape} != {label_data.shape}"
            modalities[modality_name], affines[modality_name] = spatial_pad(m, a)
            # assert np.testing.assert_allclose(affines[modality_name], label_affine)


        # k = 96 // 2
        # plt.imshow(modalities["flair"][:, :, k], cmap="gray")
        # plt.contour(label[:, :, k], colors="r")
        # plt.show()
        

        X, Y, Z = label.shape

        xs = compute_patch_starts(
            X,
            roi_x,
            stride[0],
        )

        ys = compute_patch_starts(
            Y,
            roi_y,
            stride[1],
        )

        zs = compute_patch_starts(
            Z,
            roi_z,
            stride[2],
        )

        patch_id = 1

        for x in xs:
            for y in ys:
                for z in zs:
                    label_patch = label[
                        x:x + roi_x,
                        y:y + roi_y,
                        z:z + roi_z,
                    ]

                    label_patch = label_patch.copy()
                    if not keep_patch(label_patch, brain_keep_treshold=0.0, brain_keep_prob=1.0):
                        continue
                    
                    assert label_patch.shape == (roi_x, roi_y, roi_z)
                    
                    patches = dict()
                    for name, modality in modalities.items():
                        patches[name] = modality[
                            x:x + roi_x,
                            y:y + roi_y,
                            z:z + roi_z,
                        ]
                        assert patches[name].shape == (roi_x, roi_y, roi_z)

                    # k = 96 // 2
                    # plt.imshow(patches["flair"][:, :, k], cmap="gray")
                    # plt.contour(label_patch[:, :, k], colors="r")
                    # plt.show()
                    
                    save_patch(
                        output_dir,
                        subject_id=patient_id,
                        patch_index=patch_id,
                        modalities=patches,
                        label=label_patch,
                        affines=affines,
                    )

                    patch_id += 1
        print(f"    to {patch_id - 1} patches")


def main():
    data_dir = "./dataset/final_labeled_dataset"
    output_dir = "./dataset/train_patches"
    roi_x, roi_y, roi_z = 101, 101, 101
    extract_patches(data_dir, output_dir, roi_x, roi_y, roi_z)
    train_json.create_json(Path(output_dir))
    valid_json.create_json(Path(data_dir))


if __name__ == "__main__":
    main()
