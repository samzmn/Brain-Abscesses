from __future__ import annotations

import argparse
from pathlib import Path
from functools import partial
from typing import List, Dict

import ants
import nibabel as nib
import numpy as np
import torch
from torchvision.transforms.v2 import Compose
from torch.utils.data import DataLoader

from labeling.register import register_seq_to_seq
from data.data_utils import get_loader, BrainDataset
import data.transforms as transforms
from utils.inferers import separate_sliding_window_inference
from networks.swin import SwinUNETR
from networks.model import AbscessSepSwinUNETR


# ============================================================
# Argument parser
# python ./src/main.py --root "./dataset/test_set/90" --flair "subject_090_FLAIR.nii.gz" --t1 "subject_090_T1.nii.gz" --t2 "subject_090_T2.nii.gz" --t1c "subject_090_T1.nii.gz"

def parse_args():

    parser = argparse.ArgumentParser(
        description="Run brain/lesion segmentation inference for a single patient."
    )

    # --------------------------------------------------------
    # Patient data
    # --------------------------------------------------------

    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root directory containing the patient's MRI modalities."
    )

    parser.add_argument(
        "--flair",
        type=str,
        required=True,
        help="Filename of the FLAIR image inside root_dir."
    )

    parser.add_argument(
        "--t1",
        type=str,
        required=True,
        help="Filename of the T1 image inside root_dir."
    )

    parser.add_argument(
        "--t2",
        type=str,
        required=True,
        help="Filename of the T2 image inside root_dir."
    )

    parser.add_argument(
        "--t1c",
        type=str,
        required=True,
        help="Filename of the contrast-enhanced T1 image inside root_dir."
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default="./pretrained_models/final_sep_model.pt",
        help="Path to the trained model checkpoint."
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    parser.add_argument(
        "--roi_size",
        type=int,
        nargs=3,
        default=(128, 128, 128),
        metavar=("D", "H", "W"),
        help="Sliding-window ROI size. Default: 128 128 128."
    )

    parser.add_argument(
        "--overlap",
        type=float,
        default=0.6,
        help="Sliding-window overlap. Default: 0.6."
    )

    parser.add_argument(
        "--sw_batch_size",
        type=int,
        default=1,
        help="Number of windows processed simultaneously. Default: 1."
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory where prediction files will be saved."
    )

    parser.add_argument(
        "--output_name",
        type=str,
        default="prediction.nii.gz",
        help="Output segmentation filename."
    )

    # --------------------------------------------------------
    # Hardware
    # --------------------------------------------------------

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=("cuda", "cpu"),
        help="Inference device. Default: cuda."
    )

    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="CUDA GPU index. Default: 0."
    )

    return parser.parse_args()


# ============================================================
# Utilities

def resolve_path(root_dir: Path, filename: str) -> Path:
    path = root_dir / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find modality:\n{path}"
        )
    return path


def load_nifti(path: Path):
    image = nib.load(str(path))
    image = nib.as_closest_canonical(image)   # Converts to RAS
    data = image.get_fdata(dtype=np.float32)
    return data, image.affine, image.header


# ============================================================
# Preprocessing

def preprocess_patient(
    flair_path: Path,
    t1_path: Path,
    t2_path: Path,
    t1c_path: Path,
    output_dir: Path,
):
    output_dir = output_dir / "temp"
    output_dir.mkdir(parents=True, exist_ok=True)

    t1 = ants.image_read(str(t1_path))
    t2 = ants.image_read(str(t2_path))
    flair = ants.image_read(str(flair_path))
    t1c = ants.image_read(str(t1c_path))

    register_seq_to_seq(t1, flair, out_path=f"{output_dir}/T1_registered_to_FLAIR.nii.gz")
    register_seq_to_seq(t2, flair, out_path=f"{output_dir}/T2_registered_to_FLAIR.nii.gz")
    register_seq_to_seq(t1c, flair, out_path=f"{output_dir}/T1C_registered_to_FLAIR.nii.gz")

    flair, flair_affine, _ = load_nifti(flair_path)
    nib.save(nib.Nifti1Image(flair, flair_affine), f"{output_dir}/flair.nii.gz")

    t1, t1_affine, _ = load_nifti(f"{output_dir}/T1_registered_to_FLAIR.nii.gz")
    nib.save(nib.Nifti1Image(t1, t1_affine), f"{output_dir}/t1.nii.gz")

    t2, t2_affine, _ = load_nifti(f"{output_dir}/T2_registered_to_FLAIR.nii.gz")
    nib.save(nib.Nifti1Image(t2, t2_affine), f"{output_dir}/t2.nii.gz")

    t1c, t1c_affine, _ = load_nifti(f"{output_dir}/T1C_registered_to_FLAIR.nii.gz")
    nib.save(nib.Nifti1Image(t1c, t1c_affine), f"{output_dir}/t1c.nii.gz")

    files = [{
        "image": [
            f"{output_dir}/flair.nii.gz",
            f"{output_dir}/t1c.nii.gz",
            f"{output_dir}/t1.nii.gz",
            f"{output_dir}/t2.nii.gz",
        ]
    }]

    return files


def get_loader(files: Dict[str, List[str]]):
    transform = Compose(
        [
            transforms.LoadImaged(keys=["image"], n_classes=5),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.ToTensord(keys=["image"]),
        ]
    )
    dataset = BrainDataset(files, transform=transform)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, sampler=None, 
            num_workers=1, pin_memory=False, prefetch_factor=1, persistent_workers=True
    )
    return loader


# ============================================================
# Model

def create_model(
    device: torch.device,
):
    swin_unetr_model = SwinUNETR(
        in_channels=4,
        out_channels=3,
        feature_size=48,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=False,
    )

    model = AbscessSepSwinUNETR(
        brain_out_channels=2, 
        lesion_out_channels=4, 
        swin_unetr_model=swin_unetr_model, 
        feature_size=48, 
        freeze_all=True
    )
    
    model.to(device)

    return model


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
):
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])

    print(f"\nLoaded checkpoint: {checkpoint_path}")


# ============================================================
# Main inference

def main():

    args = parse_args()

    # Device
    if args.device == "cuda":

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available."
            )

        device = torch.device(
            f"cuda:{args.gpu}"
        )

    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Resolve modality paths
    root_dir = args.root

    # Output directory
    if args.output_dir is None:
        output_dir = root_dir / "prediction"
    else:
        output_dir = args.output_dir

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    flair_path = resolve_path(root_dir, args.flair)
    t1_path = resolve_path(root_dir, args.t1)
    t2_path = resolve_path(root_dir, args.t2)
    t1c_path = resolve_path(root_dir, args.t1c)

    # Preprocessing
    files = preprocess_patient(
        flair_path,
        t1_path,
        t2_path,
        t1c_path,
        output_dir,
    )

    # Load MRI volumes
    print("\nLoading MRI volumes...")
    data_loader = get_loader(files)

    # Model
    model = create_model(device=device)

    load_checkpoint(
        model=model,
        checkpoint_path=args.checkpoint,
    )

    model.eval()

    # Inference
    model_inferer = partial(
        separate_sliding_window_inference,
        roi_size=tuple(args.roi_size),
        sw_batch_size=args.sw_batch_size,
        predictor=model,
        overlap=args.overlap,
        device=device,
        valid_type=None,
    )

    print("\nRunning inference...")
    for batch in data_loader:
        image = batch["image"][0]
        affine = batch["affine"][0]

        logit = model_inferer(image)
        brain_logit = logit[0:2, :, :, :]
        lesion_logit = logit[2:, :, :, :]
        y_brain = np.argmax(brain_logit, axis=0, keepdims=True)
        y_lesion = np.argmax(lesion_logit, axis=0, keepdims=True) + 1
        y_pred = np.where(y_lesion != 1, y_lesion, y_brain)
    
    # Save segmentation
    output_path = output_dir / args.output_name

    prediction_img = nib.Nifti1Image(
        y_pred.squeeze().astype(np.uint8),
        affine,
    )

    nib.save(
        prediction_img,
        str(output_path),
    )

    print(
        f"\nSegmentation saved to:\n"
        f"  {output_path}"
    )

    # --------------------------------------------------------
    # Print class statistics
    # --------------------------------------------------------

    print("\nPredicted classes:")

    classes = {
        "background": 0,
        "brain": 1,
        "edema": 2,
        "abscess": 3,
        "ring": 4,
    }

    for class_name, class_id in classes.items():

        voxel_count = np.sum(
            y_pred == class_id
        )

        print(
            f"  Class {class_name}: "
            f"{voxel_count:,} voxels"
        )

    print("\nInference completed.")


if __name__ == "__main__":
    main()
