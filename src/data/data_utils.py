import json
import math
import os
from typing import Any, Union, List, Tuple, Dict
from torchvision.transforms.v2 import Compose
from torch.utils.data import DataLoader, Dataset
import nibabel as nib
import numpy as np

import data.transforms as transforms

class BrainDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = dict(self.samples[idx])

        if self.transform is not None:
            item = self.transform(item)

        return item
    

def data_read(datalist, basedir) -> List[Dict[str, str]]:
    with open(datalist) as f:
        json_data = json.load(f)

    data = []
    for key, value in json_data.items():
        for d in value:
            for k, v in d.items():
                if isinstance(d[k], list):
                    d[k] = [os.path.join(basedir, iv) for iv in d[k]]
                elif isinstance(d[k], str):
                    d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]
            data.append(d)

    return data


def get_loader(
        data_dir, datalist_json, test_mode: bool, roi_x, roi_y, roi_z, batch_size, num_workers
    ) -> DataLoader:
    files = data_read(datalist=datalist_json, basedir=data_dir)
    if test_mode:
        transform = Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                # transforms.CropForegroundd(
                #     keys=["image", "label"], source_key="image", k_divisible=[roi_x, roi_y, roi_z], allow_smaller=True
                # ),
                # transforms.SpatialPadd(keys=["image", "label"], spatial_size=[roi_x, roi_y, roi_z]),
                transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                transforms.ToTensord(keys=["image", "label"]),
            ]
        )
    else:
        transform = Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                # transforms.CropForegroundd(
                #     keys=["image", "label"], source_key="image", k_divisible=[roi_x, roi_y, roi_z], allow_smaller=True
                # ),
                # transforms.SpatialPadd(keys=["image", "label"], spatial_size=[roi_x, roi_y, roi_z]),
                transforms.RandSpatialCropd(
                    keys=["image", "label"], roi_size=[roi_x, roi_y, roi_z], random_center=True, random_size=False
                ),
                transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
                transforms.ToTensord(keys=["image", "label"]),
            ]
        )

    if test_mode:
        dataset = BrainDataset(files, transform=transform)
        loader = DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=num_workers, sampler=None, pin_memory=False, prefetch_factor=1, persistent_workers=True
        )
    else:
        dataset = BrainDataset(files, transform=transform)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            sampler=None,
            pin_memory=False,
            prefetch_factor=1,
            persistent_workers=True
        )

    return loader


def test():
    train_loader = get_loader(data_dir="dataset/train_patches", datalist_json="jsons/train.json", test_mode=True, 
                                            roi_x=96, roi_y=96, roi_z=96, batch_size=1, num_workers=8)
    print(len(train_loader))

    for i, batch in enumerate(train_loader):
        print(batch.keys()) # dict_keys(['image', 'label', 'affine', 'foreground_start_coord', 'foreground_end_coord'])
        print(batch["affine"].shape)
        print(batch['foreground_start_coord'].shape) # torch.Size([1, 3])
        print(batch['foreground_end_coord'].shape) # torch.Size([1, 3])
        image = batch["image"]
        print(image.shape) 
        # After foreground crop : torch.Size([1, 4, 256, 256, 24])
        # After Spatial padding : torch.Size([1, 4, 256, 256, 64])
        # After rand spatial crop : torch.Size([1, 4, 64, 64, 64])
        print(image.max(), image.min(), image.mean(), image.std())
        nib.save(nib.Nifti1Image(np.array(image[0][3], copy=True, dtype=np.float64), np.array(batch["affine"][0], copy=True, dtype=np.float64)), 
                 "outputs/augmented.nii.gz")
        label = batch["label"]
        print(label.shape) # torch.Size([1, 4, 256, 256, 24])
        print()

if __name__ == "__main__":
    test()
