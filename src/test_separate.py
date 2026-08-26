import os
from functools import partial
from typing import Literal

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import nibabel as nib
import numpy as np
import torch
from torch.amp import autocast
from openpyxl import Workbook

from torchmetrics.classification import BinaryJaccardIndex
from torchmetrics.classification import (
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
)

from data.data_utils import get_loader
from utils.inferers import separate_sliding_window_inference
from utils.metrics import DiceMetric
from networks.swin import SwinUNETR
from networks.model import AbscessSepSwinUNETR


def main(
    roi = (128, 128, 128),
    infer_overlap = 0.6, # sliding window inference overlap
    output_directory = "./outputs/" + "test_sep",
    data_dir = "./dataset/brain_labeled_dataset/",
    json_list = "./jsons/valid.json",
    batch_size = 1,
    n_workers = 4,
    checkpoint_dir: str = "./runs/train/brain_out_dec1_trained_2/model_epoch_13_0.7920.pt", # checkpoint dir to continue training from saved checkpoint
    load_pretrained = True,
    use_amp = False,
    has_label = True,
    test_type: Literal["brain", "lesion", None] = "brain" 
):
    roi_x, roi_y, roi_z = roi
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    if test_type is None:
        out_excel = os.path.join(output_directory, "test_results.xlsx")
        # Excle Configuration
        labels = {
            0: "Background",
            1: "Brain Voxels",
            2: "Edema Voxels",
            3: "Abscess Voxels",
            4: "Ring Voxels",
        }
        wb = Workbook() # Create Excel workbook
        ws = wb.active
        ws.title = "model results"
        headers = [
            "Patient ID",
            "Brain Voxels",
            "Edema Voxels",
            "Abscess Voxels",
            "Ring Voxels",
            "Total Labeled Voxels",
            "Physical Voxel Volume",
            "Brain Volume",
            "Edema Volume",
            "Abscess Volume",
            "Ring Volume",
            "Total Labeled Volume",
            "Edema/Brain Ratio",
            "Edema/Lesion Ratio",
            "Lesion/Brain Ratio",
            "Background Dice Score",
            "Brain Dice Score",
            "Edema Dice Score",
            "Abscess Dice Score",
            "Ring Dice Score",
            "MEAN Dice Score",
            "Background Precision",
            "Brain Precision",
            "Edema Precision",
            "Abscess Precision",
            "Ring Precision",
            "MEAN Precision",
            "Background Recall",
            "Brain Recall",
            "Edema Recall",
            "Abscess Recall",
            "Ring Recall",
            "MEAN Recall",
            "Background F1 Score",
            "Brain F1 Score",
            "Edema F1 Score",
            "Abscess F1 Score",
            "Ring F1 Score",
            "MEAN F1 Score",
        ]
        ws.append(headers)

    if test_type == "brain":
        n_classes = 2
    elif test_type == "lesion":
        n_classes = 4
    else:
        n_classes = 5
        
    test_loader = get_loader(data_dir, json_list, n_classes=n_classes, to_one_hot_y=True, test_mode=True, roi_x=roi_x, roi_y=roi_y, roi_z=roi_z, 
                             batch_size=1, num_workers=n_workers, with_label=has_label)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth = os.path.join("./pretrained_models/fold1_f48_ep300_4gpu_dice0_9059/", "model.pt")
    model = SwinUNETR(
        in_channels=4,
        out_channels=3,
        feature_size=48,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=False,
    )
    if load_pretrained:
        model_dict = torch.load(pretrained_pth, weights_only=False)["state_dict"]
        model.load_state_dict(model_dict)
        
    model = AbscessSepSwinUNETR(
        brain_out_channels=2, 
        lesion_out_channels=4, 
        swin_unetr_model=model, 
        feature_size=48, 
        freeze_all=True
    )
    checkpoint = torch.load(checkpoint_dir, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    print("=> loaded checkpoint '{}'".format(checkpoint_dir))

    model.eval()
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,}")

    model_inferer_test = partial(
        separate_sliding_window_inference,
        roi_size=[roi_x, roi_y, roi_z],
        sw_batch_size=1,
        predictor=model,
        overlap=infer_overlap,
        device=device,
        valid_type=test_type,
    )

    for i, batch in enumerate(test_loader):
        patient_id = batch["id"][0]
        image = batch["image"][0]
        affine = batch["affine"][0]
        if has_label:
            label = batch["label"][0]
            
        print(f"Inference on case {patient_id}")

        logit = model_inferer_test(image)
        if test_type is None:
            brain_logit = logit[0:2, :, :, :]
            lesion_logit = logit[2:, :, :, :]
            # y_brain = torch.argmax(brain_logit, dim=0, keepdim=True)
            # y_lesion = torch.argmax(lesion_logit, dim=0, keepdim=True) + 1
            # y_pred = torch.where(y_lesion != 1, y_lesion, y_brain)
            y_brain = np.argmax(brain_logit, axis=0, keepdims=True)
            y_lesion = np.argmax(lesion_logit, axis=0, keepdims=True) + 1
            y_pred = np.where(y_lesion != 1, y_lesion, y_brain)
            # y_lesion = torch.sigmoid(lesion_logit)
            # y_lesion = y_lesion > 0.5
            # y_pred = torch.zeros((y_lesion.shape[1], y_lesion.shape[2], y_lesion.shape[3]))
            # y_pred[y_brain[0] == 1] = 1
            # y_pred[y_lesion[1] == 1] = 2
            # y_pred[y_lesion[2] == 1] = 3
            # y_pred[y_lesion[3] == 1] = 4
            # y_pred = y_pred.unsqueeze(0)
        else:
            y_pred = torch.argmax(logit, dim=0, keepdim=True)

        if test_type is None:
            excl_row = []
            excl_row.append(patient_id)
            counts = {
                label: int(np.count_nonzero(y_pred == label))
                for label in labels if label != 0
            }
            total = sum(counts.values())
            voxel_volume_mm3 = abs(np.linalg.det(affine[:3, :3]))
            for i in range(1, 5):
                excl_row.append(counts[i])
            excl_row.append(total)
            excl_row.append(voxel_volume_mm3)
            for i in range(1, 5):
                excl_row.append(counts[i] * voxel_volume_mm3 / 1000)
            excl_row.append(total * voxel_volume_mm3 / 1000)
            excl_row.append(counts[2] / total)
            excl_row.append(counts[2] / (counts[3] + counts[4]))
            excl_row.append((counts[3] + counts[4]) / total)

        if has_label:
            target = label.unsqueeze(0)
            y_pred = torch.tensor(y_pred).unsqueeze(0)
            print(y_pred.shape, target.shape)
            
            dice_acc = DiceMetric(include_background=True, reduction="mean_batch", num_classes=n_classes)
            dice_acc.reset()
            dice_acc.update(y_pred=y_pred, y=target)
            acc = dice_acc.aggregate()
            acc_mean = acc.nanmean()
            print(
                "Dice:", acc,
                ", MEAN", acc_mean,
            )
            target = torch.argmax(target, dim=1, keepdim=True)
            precision_metric = MulticlassPrecision(num_classes=n_classes, average=None)
            recall_metric = MulticlassRecall(num_classes=n_classes, average=None)
            f1_metric = MulticlassF1Score(num_classes=n_classes, average=None)
            precision_metric.reset()
            precision_metric.update(y_pred, target)
            precision = precision_metric.compute()
            print("precision: ", precision)
            recall_metric.reset()
            recall_metric.update(y_pred, target)
            recall = recall_metric.compute()
            print("recall: ", recall)
            f1_metric.reset()
            f1_metric.update(y_pred, target)
            f1 = f1_metric.compute()
            print("f1 score: ", f1)
            if test_type is None:
                for d_score in acc.numpy():
                    excl_row.append(d_score * 100)
                excl_row.append(acc_mean.numpy() * 100)
                for p in precision.numpy():
                    excl_row.append(p * 100)
                excl_row.append(precision.nanmean().numpy() * 100)
                for r in recall.numpy():
                    excl_row.append(r * 100)
                excl_row.append(recall.nanmean().numpy() * 100)
                for f in f1.numpy():
                    excl_row.append(f * 100)
                excl_row.append(f1.nanmean().numpy() * 100)

        y_pred = np.array(y_pred)
        img_name = f"subject_{patient_id}_pred_out.nii.gz"
        nib.save(nib.Nifti1Image(y_pred.squeeze().astype(np.uint8), affine), os.path.join(output_directory, img_name))
        print(f"Output file saved as '{img_name}'")
        if test_type is None:
            ws.append(excl_row)

    if test_type is None:
        wb.save(out_excel)
    print("Finished inference!")


if __name__ == "__main__":
    # main(checkpoint_dir="./runs/train/lesion_out_dec1_trained_7/model_epoch_4_0.8110.pt", 
    #      data_dir="./dataset/lesion_labeled_dataset", test_type="lesion")
    # main(checkpoint_dir="./runs/train/final_sep_model.pt", data_dir="./dataset/final_labeled_dataset",
    #      json_list="./jsons/test.json", test_type=None, output_directory="./outputs/test_results_3/")
    # main(checkpoint_dir="./runs/train/final_sep_model.pt", data_dir="./dataset/final_labeled_dataset",
    #          json_list="./jsons/test.json", test_type=None, output_directory="./outputs/test_results_5/", roi=(128, 128, 128))

    main(checkpoint_dir="./runs/train/final_sep_model.pt", data_dir="./dataset/test_set", has_label=False,
                 json_list="./jsons/test_set.json", test_type=None, output_directory="./outputs/test_set_results/", roi=(128, 128, 128))
