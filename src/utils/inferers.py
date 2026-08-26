from collections.abc import Callable
from typing import Tuple
from tqdm import tqdm
import torch
from torch.amp import autocast
import numpy as np
from scipy.ndimage import gaussian_filter
from data.transforms import SpatialPad


def sliding_window_inference(
    input_volume: torch.Tensor,                 # (C, D, H, W)
    roi_size: Tuple[int, int, int],             # (D, H, W)
    sw_batch_size: int,                         # batch size
    predictor: Callable[..., torch.Tensor],     # model
    device: torch.device | str | None,          # 'cpu' or 'cuda'
    return_device: torch.device |str = "cpu",   # device to return the output tensor - 'cpu' or 'cuda'
    overlap: float = 0.25,                      # overlap fraction
    mode: str = "constant",                     # or 'gaussian'
    include_edge_patches: bool = True
) -> torch.Tensor:
    """
    PyTorch-based sliding window inference with full device support.

    Returns:
        output_volume: torch.Tensor of shape (C, D, H, W)
    """
    spatial_padding = SpatialPad(spatial_size=roi_size)
    input_volume , pad_width = spatial_padding.pad(input_volume.cpu())

    C, D, H, W = input_volume.shape
    roi_d, roi_h, roi_w = roi_size
    stride = [int(s * (1 - overlap)) for s in roi_size]

    # Infer output shape
    dummy = torch.zeros((1, C, roi_d, roi_h, roi_w), dtype=torch.float32, device=device)

    with torch.no_grad():
        dummy_out = predictor(dummy)

    num_classes = dummy_out.shape[1]

    output_volume = torch.zeros((num_classes, D, H, W), dtype=torch.float32, device=device)
    count_map = torch.zeros((D, H, W), dtype=torch.float32, device=device)

    def _get_weight_patch(shape):
        if mode == "gaussian":
            center = [s // 2 for s in shape]
            sigma = [s * 0.125 for s in shape]
            grid = np.zeros(shape, dtype=np.float32)
            grid[tuple(center)] = 1
            return gaussian_filter(grid, sigma=sigma)
        else:
            return np.ones(shape, dtype=np.float32)

    weight_patch = torch.tensor(
        _get_weight_patch((roi_d, roi_h, roi_w))[np.newaxis, ...],
        dtype=torch.float32,
        device=device
    )

    # Sliding window locations
    def compute_starts(dim, roi, stride):
        starts = list(range(0, max(dim - roi + 1, 1), stride))
        if include_edge_patches and (starts[-1] + roi < dim):
            starts.append(dim - roi)
        return starts

    z_starts = compute_starts(D, roi_d, stride[0])
    y_starts = compute_starts(H, roi_h, stride[1])
    x_starts = compute_starts(W, roi_w, stride[2])
    patch_coords = [(z, y, x) for z in z_starts for y in y_starts for x in x_starts]

    batch_patches = []
    batch_coords = []

    for i, (z, y, x) in enumerate(tqdm(patch_coords)):
        patch = input_volume[:, z:z+roi_d, y:y+roi_h, x:x+roi_w]  # shape: (C, D, H, W)
        batch_patches.append(patch)
        batch_coords.append((z, y, x))

        if len(batch_patches) == sw_batch_size or i == len(patch_coords) - 1:
            batch_np = np.stack(batch_patches, axis=0)  # (B, C, D, H, W)
            batch_tensor = torch.tensor(batch_np, dtype=torch.float32, device=device)

            with torch.no_grad():
                preds = predictor(batch_tensor)  # (B, C, D, H, W)

            for j, (pz, py, px) in enumerate(batch_coords):
                output_volume[:, pz:pz+roi_d, py:py+roi_h, px:px+roi_w] += preds[j] * weight_patch
                count_map[pz:pz+roi_d, py:py+roi_h, px:px+roi_w] += weight_patch[0, ...]

            batch_patches.clear()
            batch_coords.clear()

    count_map = count_map.clamp(min=1e-5)
    output_volume = output_volume / count_map[None, ...]

    # Zero out voxels that were never predicted
    output_volume[:, count_map == 0] = 0
    output_volume = output_volume.to(return_device)
    output_volume = spatial_padding.unpad(output_volume, pad_width)
    return output_volume

def separate_sliding_window_inference(
    input_volume: torch.Tensor,                 # (C, D, H, W)
    roi_size: Tuple[int, int, int],             # (D, H, W)
    sw_batch_size: int,                         # batch size
    predictor: Callable[..., torch.Tensor],     # model
    device: torch.device | str | None,          # 'cpu' or 'cuda'
    return_device: torch.device |str = "cpu",   # device to return the output tensor - 'cpu' or 'cuda'
    overlap: float = 0.25,                      # overlap fraction
    mode: str = "constant",                     # or 'gaussian'
    include_edge_patches: bool = True,
    valid_type: str | None = None,              # either "brain" or "lesion" or None
    use_amp:bool = False,
) -> torch.Tensor:
    """
    PyTorch-based sliding window inference with full device support.

    Returns:
        output_volume: torch.Tensor of shape (C, D, H, W)
    """
    spatial_padding = SpatialPad(spatial_size=roi_size)
    input_volume , pad_width = spatial_padding.pad(input_volume.cpu())

    C, D, H, W = input_volume.shape
    roi_d, roi_h, roi_w = roi_size
    stride = [int(s * (1 - overlap)) for s in roi_size]

    # Infer output shape
    dummy = torch.zeros((1, C, roi_d, roi_h, roi_w), dtype=torch.float32, device=device)

    with torch.no_grad():
        dummy_brain_out, dummy_lesion_out = predictor(dummy)
        if valid_type is None:
            dummy_out = torch.concatenate([dummy_brain_out, dummy_lesion_out], dim=1)
        elif valid_type == "brain":
            dummy_out = dummy_brain_out
        elif valid_type == "lesion":
            dummy_out = dummy_lesion_out
        else:
            raise ValueError(f"no such type as {valid_type}. it can be either 'brain' or 'lesion'")

    num_classes = dummy_out.shape[1]

    output_volume = torch.zeros((num_classes, D, H, W), dtype=torch.float32, device=return_device)
    count_map = torch.zeros((D, H, W), dtype=torch.float32, device=return_device)

    def _get_weight_patch(shape):
        if mode == "gaussian":
            center = [s // 2 for s in shape]
            sigma = [s * 0.125 for s in shape]
            grid = np.zeros(shape, dtype=np.float32)
            grid[tuple(center)] = 1
            return gaussian_filter(grid, sigma=sigma)
        else:
            return np.ones(shape, dtype=np.float32)

    weight_patch = torch.tensor(
        _get_weight_patch((roi_d, roi_h, roi_w))[np.newaxis, ...],
        dtype=torch.float32,
        device=return_device
    )

    # Sliding window locations
    def compute_starts(dim, roi, stride):
        starts = list(range(0, max(dim - roi + 1, 1), stride))
        if include_edge_patches and (starts[-1] + roi < dim):
            starts.append(dim - roi)
        return starts

    z_starts = compute_starts(D, roi_d, stride[0])
    y_starts = compute_starts(H, roi_h, stride[1])
    x_starts = compute_starts(W, roi_w, stride[2])
    patch_coords = [(z, y, x) for z in z_starts for y in y_starts for x in x_starts]

    batch_patches = []
    batch_coords = []

    for i, (z, y, x) in enumerate(tqdm(patch_coords)):
        patch = input_volume[:, z:z+roi_d, y:y+roi_h, x:x+roi_w]  # shape: (C, D, H, W)
        batch_patches.append(patch)
        batch_coords.append((z, y, x))

        if len(batch_patches) == sw_batch_size or i == len(patch_coords) - 1:
            batch_np = np.stack(batch_patches, axis=0)  # (B, C, D, H, W)
            # print(batch_np.shape)
            batch_tensor = torch.tensor(batch_np, dtype=torch.float32).to(device)
            # batch_tensor = torch.stack(batch_patches).to(device, non_blocking=True)
            # print(batch_tensor.shape)

            with torch.no_grad():
                with autocast(device_type=str(device), enabled=use_amp):
                    brain_preds, lesion_preds = predictor(batch_tensor)  # (B, C, D, H, W)
                if valid_type is None:
                    preds = torch.concatenate([brain_preds, lesion_preds], dim=1)
                elif valid_type == "brain":
                    preds = brain_preds
                elif valid_type == "lesion":
                    preds = lesion_preds

            preds = preds.to(return_device)
            for j, (pz, py, px) in enumerate(batch_coords):
                output_volume[:, pz:pz+roi_d, py:py+roi_h, px:px+roi_w] += preds[j] * weight_patch
                count_map[pz:pz+roi_d, py:py+roi_h, px:px+roi_w] += weight_patch[0, ...]

            batch_patches.clear()
            batch_coords.clear()

    count_map = count_map.clamp(min=1e-5)
    output_volume = output_volume / count_map[None, ...]

    # Zero out voxels that were never predicted
    output_volume[:, count_map == 0] = 0
    output_volume = output_volume.to(return_device)
    output_volume = spatial_padding.unpad(output_volume, pad_width)
    return output_volume


