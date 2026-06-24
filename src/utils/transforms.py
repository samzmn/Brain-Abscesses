from typing import Dict, Union, Tuple
import nibabel as nib
import numpy as np
import torch
np.random.seed(47)
np.random.RandomState(47)

class LoadImaged:
    def __init__(self, keys):
        self.keys = keys

    def _load_nifti(self, path: str) -> Tuple[np.array, np.array]:
        img = nib.load(path)
        return img.get_fdata().astype(np.float32), img.affine

    def __call__(self, data: Dict[str, Union[list|str]]):
        for key in self.keys:
            value = data[key]

            # Multiple modalities
            if isinstance(value, list):
                # channels = [self._load_nifti(f)[0] for f in value]
                channels = []
                for i, f in enumerate(value):
                    img, aff = self._load_nifti(f)
                    if i == 0: # only save the first image's affine
                        data["affine"] = aff
                    channels.append(img)
                # (C, H, W, D)
                data[key] = np.stack(channels, axis=0)
                # data["affine"] = self._load_nifti(value[0])[1]

            # Single volume
            else: # instance of str
                volume = self._load_nifti(value)[0]

                # Label remains (H, W, D)
                data[key] = volume

        return data


class NormalizeIntensityd:
    def __init__(
        self,
        keys,
        nonzero=True,
        channel_wise=True,
        eps=1e-8,
    ):
        self.keys = keys if isinstance(keys, (list, tuple)) else [keys]
        self.nonzero = nonzero
        self.channel_wise = channel_wise
        self.eps = eps

    def _normalize(self, img):
        img = img.astype(np.float32)

        # img shape: (C, H, W, D)
        if self.channel_wise:
            out = np.empty_like(img)

            for c in range(img.shape[0]):
                channel = img[c]

                if self.nonzero:
                    mask = channel != 0

                    if np.any(mask):
                        mean = channel[mask].mean()
                        std = channel[mask].std()

                        out[c] = channel.copy()
                        out[c][mask] = (
                            channel[mask] - mean
                        ) / (std + self.eps)
                    else:
                        out[c] = channel

                else:
                    mean = channel.mean()
                    std = channel.std()

                    out[c] = (channel - mean) / (std + self.eps)

            return out

        else:
            if self.nonzero:
                mask = img != 0

                if np.any(mask):
                    mean = img[mask].mean()
                    std = img[mask].std()

                    out = img.copy()
                    out[mask] = (img[mask] - mean) / (std + self.eps)
                    return out

                return img

            mean = img.mean()
            std = img.std()

            return (img - mean) / (std + self.eps)

    def __call__(self, data: Dict):
        for key in self.keys:
            data[key] = self._normalize(data[key])

        return data
    

class ToTensord:
    def __init__(self, keys):
        self.keys = keys if isinstance(keys, (list, tuple)) else [keys]

    def __call__(self, data: Dict):
        for key in self.keys:
            data[key] = np.ascontiguousarray(data[key])
            data[key] = torch.tensor(data[key], dtype=torch.float32)

        return data


class CropForegroundd:
    def __init__(
        self,
        keys,
        source_key,
        margin=0,
        allow_smaller=True,
        k_divisible=1,
        start_coord_key="foreground_start_coord",
        end_coord_key="foreground_end_coord",
    ):
        self.keys = keys if isinstance(keys, (list, tuple)) else [keys]
        self.source_key = source_key
        self.margin = margin
        self.allow_smaller = allow_smaller
        self.k_divisible = k_divisible
        self.start_coord_key = start_coord_key
        self.end_coord_key = end_coord_key

    def _ensure_tuple(self, value, ndim):
        if isinstance(value, int):
            return (value,) * ndim
        return tuple(value)

    def compute_bounding_box(self, img):
        """
        img shape:
            (C,H,W,D) for image
        """

        # foreground mask
        mask = np.any(img != 0, axis=0)

        if not np.any(mask):
            spatial_shape = img.shape[1:]
            return np.zeros(3, dtype=int), np.array(spatial_shape)

        coords = np.where(mask)

        start = np.array(
            [
                coords[0].min(),
                coords[1].min(),
                coords[2].min(),
            ]
        )

        end = np.array(
            [
                coords[0].max() + 1,
                coords[1].max() + 1,
                coords[2].max() + 1,
            ]
        )

        spatial_shape = np.array(img.shape[1:])

        margin = self._ensure_tuple(self.margin, 3)

        start = start - np.array(margin)
        end = end + np.array(margin)

        if self.allow_smaller:
            start = np.maximum(start, 0)
            end = np.minimum(end, spatial_shape)

        k_divisible = self._ensure_tuple(self.k_divisible, 3)

        # expand bbox so every dimension is divisible by k
        for dim in range(3):
            k = k_divisible[dim]

            if k <= 1:
                continue

            size = end[dim] - start[dim]

            target_size = int(np.ceil(size / k) * k)

            extra = target_size - size

            before = extra // 2
            after = extra - before

            start[dim] -= before
            end[dim] += after

            if self.allow_smaller:
                start[dim] = max(start[dim], 0)
                end[dim] = min(end[dim], spatial_shape[dim])

        return start.astype(int), end.astype(int)

    def crop(self, img, start, end):
        if img.ndim == 4:
            return img[
                :,
                start[0]:end[0],
                start[1]:end[1],
                start[2]:end[2],
            ]

        elif img.ndim == 3:
            return img[
                start[0]:end[0],
                start[1]:end[1],
                start[2]:end[2],
            ]

        raise ValueError(f"Unsupported ndim={img.ndim}")

    def __call__(self, data):
        data = dict(data)

        box_start, box_end = self.compute_bounding_box(
            data[self.source_key]
        )

        data[self.start_coord_key] = box_start
        data[self.end_coord_key] = box_end

        for key in self.keys:
            data[key] = self.crop(
                data[key],
                box_start,
                box_end,
            )

        return data
    

class RandSpatialCropd:
    def __init__(
        self,
        keys,
        roi_size,
        random_center=True,
        random_size=False,
    ):
        self.keys = keys if isinstance(keys, (list, tuple)) else [keys]
        self.roi_size = tuple(roi_size)
        self.random_center = random_center
        self.random_size = random_size

        if random_size:
            raise NotImplementedError(
                "random_size=True not implemented."
            )

    def _get_spatial_shape(self, img):
        if img.ndim == 4:
            return img.shape[1:]  # C,H,W,D
        elif img.ndim == 3:
            return img.shape
        else:
            raise ValueError(f"Unsupported shape {img.shape}")

    def _compute_crop_slices(self, spatial_shape):
        starts = []
        ends = []

        for dim_size, roi_size in zip(
            spatial_shape,
            self.roi_size,
        ):
            # MONAI behavior:
            # if roi > image dimension -> keep full dimension
            roi = min(roi_size, dim_size)

            if self.random_center:
                max_start = dim_size - roi

                if max_start > 0:
                    start = np.random.randint(
                        0,
                        max_start + 1,
                    )
                else:
                    start = 0
            else:
                start = max(
                    (dim_size - roi) // 2,
                    0,
                )

            end = start + roi

            starts.append(start)
            ends.append(end)

        return starts, ends

    def _crop(self, img, starts, ends):
        if img.ndim == 4:
            return img[
                :,
                starts[0]:ends[0],
                starts[1]:ends[1],
                starts[2]:ends[2],
            ]

        elif img.ndim == 3:
            return img[
                starts[0]:ends[0],
                starts[1]:ends[1],
                starts[2]:ends[2],
            ]

        raise ValueError(f"Unsupported ndim={img.ndim}")

    def __call__(self, data: Dict):
        reference = data[self.keys[0]]

        spatial_shape = self._get_spatial_shape(reference)

        starts, ends = self._compute_crop_slices(
            spatial_shape
        )

        for key in self.keys:
            data[key] = self._crop(
                data[key],
                starts,
                ends,
            )

        return data


class SpatialPadd:
    def __init__(
        self,
        keys,
        spatial_size,
        mode="constant",
        constant_values=0,
    ):
        self.keys = keys if isinstance(keys, (list, tuple)) else [keys]
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
        if img.ndim == 4:
            # (C,H,W,D)
            spatial_shape = img.shape[1:]

            spatial_pad = self._compute_pad_width(
                spatial_shape
            )

            pad_width = [(0, 0)] + spatial_pad

        elif img.ndim == 3:
            # (H,W,D)
            spatial_shape = img.shape

            pad_width = self._compute_pad_width(
                spatial_shape
            )

        else:
            raise ValueError(
                f"Unsupported shape {img.shape}"
            )

        return np.pad(
            img,
            pad_width=pad_width,
            mode=self.mode,
            constant_values=self.constant_values,
        )

    def __call__(self, data: Dict):
        for key in self.keys:
            data[key] = self._pad(data[key])

        return data


class RandFlipd:
    def __init__(
        self,
        keys,
        prob=0.1,
        spatial_axis=None,
    ):
        self.keys = keys if isinstance(keys, (list, tuple)) else [keys]
        self.prob = prob

        if spatial_axis is None:
            self.spatial_axis = None
        elif isinstance(spatial_axis, int):
            self.spatial_axis = [spatial_axis]
        else:
            self.spatial_axis = list(spatial_axis)

    @staticmethod
    def _flip_axis(arr, axis):
        idx = [slice(None)] * arr.ndim
        idx[axis] = slice(None, None, -1)
        return arr[tuple(idx)]
    
    def _flip(self, img):
        if self.spatial_axis is None:
            return img

        result = img

        for axis in self.spatial_axis:

            if result.ndim == 4:
                # channel-first image
                flip_axis = axis + 1

            elif result.ndim == 3:
                # label
                flip_axis = axis

            else:
                raise ValueError(
                    f"Unsupported ndim={result.ndim}"
                )

            # result = np.flip(
            #     result,
            #     axis=flip_axis,
            # )
            result = self._flip_axis(result, axis)

        return result

    def __call__(self, data: Dict):
        do_flip = np.random.rand() < self.prob

        if not do_flip:
            return data

        for key in self.keys:
            data[key] = self._flip(data[key])

        return data


class RandScaleIntensityd:
    def __init__(
        self,
        keys,
        factors,
        prob=0.1,
        channel_wise=False,
        dtype=np.float32,
    ):
        self.keys = keys if isinstance(keys, (list, tuple)) else [keys]
        self.prob = prob
        self.channel_wise = channel_wise
        self.dtype = dtype

        if isinstance(factors, (int, float)):
            self.factor_min = -float(factors)
            self.factor_max = float(factors)
        else:
            self.factor_min = float(factors[0])
            self.factor_max = float(factors[1])

    def _sample_factor(self):
        return np.random.uniform(
            self.factor_min,
            self.factor_max,
        )

    def _scale(self, img):
        img = img.astype(self.dtype, copy=False)

        # channel-first image (C,H,W,D)
        if self.channel_wise:

            result = img.copy()

            for c in range(img.shape[0]):
                factor = self._sample_factor()

                result[c] *= (1.0 + factor)

            return result

        # one factor for all channels
        factor = self._sample_factor()

        return img * (1.0 + factor)

    def __call__(self, data: Dict):
        if np.random.random() >= self.prob:
            return data

        # MONAI uses the SAME factor for all keys
        factor = self._sample_factor()

        for key in self.keys:
            data[key] = self._scale(data[key])

        return data


class RandShiftIntensityd:
    def __init__(
        self,
        keys,
        offsets,
        prob=0.1,
        channel_wise=False,
    ):
        self.keys = keys if isinstance(keys, (list, tuple)) else [keys]
        self.prob = prob
        self.channel_wise = channel_wise

        if isinstance(offsets, (int, float)):
            self.offset_min = -float(offsets)
            self.offset_max = float(offsets)
        else:
            self.offset_min = float(offsets[0])
            self.offset_max = float(offsets[1])

    def _sample_offset(self):
        return np.random.uniform(
            self.offset_min,
            self.offset_max,
        )

    def _shift(self, img):
        # img = img.astype(np.float32, copy=False)

        # (C,H,W,D)
        if self.channel_wise:
            out = img.copy()

            for c in range(img.shape[0]):
                offset = self._sample_offset()
                out[c] += offset

            return out

        offset = self._sample_offset()
        return img + offset

    def __call__(self, data: Dict):
        if np.random.random() >= self.prob:
            return data

        for key in self.keys:
            data[key] = self._shift(data[key])

        return data


class RandGaussianNoised:
    def __init__(
        self,
        keys,
        prob=0.1,
        mean=0.0,
        std=0.01,
        channel_wise=False,
    ):
        self.keys = keys if isinstance(keys, (list, tuple)) else [keys]
        self.prob = prob
        self.mean = float(mean)
        self.std = float(std)
        self.channel_wise = channel_wise

    def _add_noise(self, img):
        # (C,H,W,D)
        if self.channel_wise:
            out = img.copy()

            for c in range(img.shape[0]):
                noise = np.random.normal(
                    loc=self.mean,
                    scale=self.std,
                    size=img[c].shape,
                )
                out[c] += noise

            return out

        noise = np.random.normal(
            loc=self.mean,
            scale=self.std,
            size=img.shape,
        )

        return img + noise

    def __call__(self, data: Dict):
        if np.random.random() >= self.prob:
            return data

        for key in self.keys:
            data[key] = self._add_noise(data[key])

        return data


class RandRotated:
    """
    Random 90-degree rotation on 3D volumes.
    Equivalent to MONAI RandRotate90d with prob and spatial_axis.
    """

    def __init__(
        self,
        keys,
        prob=0.1,
        spatial_axes=((0, 1), (1, 2), (0, 2)),
        max_k=3,
    ):
        """
        spatial_axes: pairs of axes to rotate in 3D.
        max_k: number of 90-degree steps (1–3).
        """
        self.keys = keys if isinstance(keys, (list, tuple)) else [keys]
        self.prob = prob
        self.spatial_axes = spatial_axes
        self.max_k = max_k

    def _rotate90(self, img, k, axes):
        """
        90-degree rotation using numpy rot90.
        """

        # channel-first handling
        if img.ndim == 4:
            # (C, H, W, D)
            rotated = np.empty_like(img)

            for c in range(img.shape[0]):
                rotated[c] = np.rot90(
                    img[c],
                    k=k,
                    axes=axes,
                )

            return rotated

        elif img.ndim == 3:
            return np.rot90(img, k=k, axes=axes)

        else:
            raise ValueError(f"Unsupported shape {img.shape}")

    def __call__(self, data: Dict):
        if np.random.random() >= self.prob:
            return data

        # pick random axis pair and rotation step
        axes = self.spatial_axes[
            np.random.randint(len(self.spatial_axes))
        ]

        k = np.random.randint(1, self.max_k + 1)

        for key in self.keys:
            data[key] = self._rotate90(
                data[key],
                k=k,
                axes=axes,
            )

        return data
    