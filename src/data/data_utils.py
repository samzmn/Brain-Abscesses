import json
import math
import os
from typing import Any, Union, List, Tuple, Dict
from torchvision.transforms.v2 import Compose
from torch.utils.data import DataLoader, Dataset
import nibabel as nib
import numpy as np

import data.transforms as transforms
# import transforms as transforms

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
        data_dir, datalist_json, n_classes, to_one_hot_y: bool, test_mode: bool, roi_x, roi_y, roi_z, batch_size, num_workers
    ) -> DataLoader:
    files = data_read(datalist=datalist_json, basedir=data_dir)
    if test_mode:
        transform = Compose(
            [
                transforms.LoadImaged(keys=["image", "label"], n_classes=n_classes, to_one_hot_y=to_one_hot_y),
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
                transforms.LoadImaged(keys=["image", "label"], n_classes=n_classes, to_one_hot_y=to_one_hot_y),
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
            dataset, batch_size=batch_size, shuffle=False, sampler=None, 
             num_workers=num_workers, pin_memory=False, prefetch_factor=1, persistent_workers=True
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
    train_lesion_data_dir = "./dataset/train_lesion_patches/" # dataset directory
    train_lesion_json_list = "./jsons/train_lesion.json" # dataset json file
    train_loader = get_loader(train_lesion_data_dir, train_lesion_json_list, 4, True, False, 96, 96, 96, 1, 1)
    print(len(train_loader))

    for i, batch in enumerate(train_loader):
        print(batch.keys()) # dict_keys(['image', 'label', 'affine', 'foreground_start_coord', 'foreground_end_coord'])
        # print(batch["affine"].shape)
        affine = np.array(batch['affine'][0])
        image = np.array(batch["image"][0][3])
        print(image.shape)
        label = np.array(batch['label'][0])
        label = np.argmax(np.array(label).astype(np.uint8), axis=0)
        label = np.array(label).astype(float)
        print(np.unique(label), label.shape)
        if np.unique(label).size < 3:
            continue
        nib.save(nib.Nifti1Image(image, affine), 
                "outputs/test_img.nii.gz")
        nib.save(nib.Nifti1Image(label, affine), 
                "outputs/test_label.nii.gz")
        break

if __name__ == "__main__":
    test()
