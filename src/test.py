import os
from functools import partial

import nibabel as nib
import numpy as np
import torch

from data.data_utils import get_loader
from utils.inferers import sliding_window_inference
from networks.swin import SwinUNETR
from networks.model import AbscessSwinUNETR
from utils.metrics import DiceMetric

def main(
    in_channels = 4,
    out_channels = 5,
    feature_size = 48,
    use_checkpoint = True, # use gradient checkpointing to save memory
    roi = (128, 128, 128),
    infer_overlap = 0.6, # sliding window inference overlap
    output_directory = "./outputs/" + "test1",
    data_dir = "./dataset/final_labeled_dataset/",
    json_list = "./jsons/test.json",
    batch_size = 1,
    n_workers = 4,
    checkpoint_dir: str | None = "./runs/train/out_dec1_trained/model_epoch_38_0.7663.pt", # checkpoint dir to continue training from saved checkpoint
    load_pretrained = False,
    use_amp = False,
    has_label = True,
):
    roi_x, roi_y, roi_z = roi
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    test_loader = get_loader(data_dir, json_list, test_mode=True, roi_x=roi_x, roi_y=roi_y, roi_z=roi_z, batch_size=batch_size,
                             num_workers=n_workers)
    pretrained_dir = "./pretrained_models/fold1_f48_ep300_4gpu_dice0_9059/"
    model_name = "model.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth = os.path.join(pretrained_dir, model_name)
    model = SwinUNETR(
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=feature_size,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=use_checkpoint,
    )
    if load_pretrained:
        model_dict = torch.load(pretrained_pth, weights_only=False)["state_dict"]
        model.load_state_dict(model_dict)
    if checkpoint_dir is not None:
        model = AbscessSwinUNETR(5, 5, model, swin_in_channels=in_channels, feature_size=feature_size, freeze_all=True)
        checkpoint = torch.load(checkpoint_dir, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        print("=> loaded checkpoint '{}'".format(checkpoint_dir))

    model.eval()
    model.to(device)

    # print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    model_inferer_test = partial(
        sliding_window_inference,
        roi_size=[roi_x, roi_y, roi_z],
        sw_batch_size=1,
        predictor=model,
        overlap=infer_overlap,
        device=device
    )

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            print(batch.keys())
            image = batch["image"][0]
            print(image.shape, image.dtype)
            affine = batch["affine"][0]
            if has_label:
                label = batch["label"][0]
                
            print(f"Inference on case {batch["id"][0]}")

            if checkpoint_dir is None:
                prob = torch.sigmoid(model_inferer_test(image))
                seg = prob.detach().cpu().numpy()
                seg = (seg > 0.5).astype(np.int8)
                seg_out = np.zeros((seg.shape[1], seg.shape[2], seg.shape[3]))
                # 1 for NCR, 2 for edema(ED), 4 for ET, and 0 for everything else.
                seg_out[seg[1] == 1] = 2
                seg_out[seg[0] == 1] = 1
                seg_out[seg[2] == 1] = 4
                img_name = "patient_237_out_f1.nii.gz"
                nib.save(nib.Nifti1Image(seg_out.astype(np.uint8), affine), os.path.join(output_directory, img_name))
            else:
                logit = model_inferer_test(image)
                y_pred = torch.argmax(logit, dim=0, keepdim=True)

                acc_mean = 1.0
                if has_label:
                    target = label.unsqueeze(0).unsqueeze(0)
                    y_pred = y_pred.unsqueeze(0)
                    print(y_pred.shape, target.shape)
                    dice_acc = DiceMetric(include_background=False, reduction="mean_batch", num_classes=5)
                    dice_acc.reset()
                    dice_acc.update(y_pred=y_pred, y=target)
                    acc = dice_acc.aggregate().numpy()
                    acc_mean = acc.mean()
                    print(
                        "Dice_Br:", acc[0].item(),
                        ", Dice_Ed:", acc[1].item(),
                        ", Dice_Ab:", acc[2].item(),
                        ", Dice_Rg:", acc[3].item(),
                        ", MEAN", acc_mean
                    )
                img_name = f"subject_{batch['id'][0]}_out_{acc_mean:.3f}.nii.gz"
                nib.save(nib.Nifti1Image(y_pred.squeeze().numpy().astype(np.uint8), affine), os.path.join(output_directory, img_name))

            # pred = torch.sigmoid(model_inferer_test(image[0]))
            # pred = pred.detach().cpu().numpy()
            # pred = (pred > 0.5).astype(np.int8)
            # print(f"pred shape : {pred.shape}")
            # pred = np.argmax(pred, axis=0)
            # print(f"pred argmax: {pred.shape} , dtype: {pred.dtype}")
            # nib.save(nib.Nifti1Image(pred.astype(np.uint8), affine), os.path.join(output_directory, "patient_237_argmax_f1.nii.gz"))
        print("Finished inference!")


if __name__ == "__main__":
    main()
