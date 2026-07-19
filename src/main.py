import os
from functools import partial

import numpy as np
import torch
from utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from utils.trainer import run_training
from data.data_utils import get_loader

from utils.inferers import sliding_window_inference
from utils.losses import DiceLoss
from utils.metrics import DiceMetric
from networks.swin import SwinUNETR
from networks.model import AbscessSwinUNETR


logdir = "./runs/train" # directory to save the tensorboard logs
train_data_dir = "./dataset/train_patches/" # dataset directory
valid_data_dir = "./dataset/final_labeled_dataset/"
train_json_list = "./jsons/train.json" # dataset json file
valid_json_list = "./jsons/valid.json"

max_epochs = 50 # max number of training epochs
batch_size = 2 # number of batch size
sw_batch_size = 1 #4 # number of sliding window batch size
optim_lr = 1e-4 # 1e-4 # optimization learning rate
optim_name = "adamw" # optimization algorithm
reg_weight = 1e-5 # regularization weight
momentum = 0.99 # momentum
val_every = 1 #100 # validation frequency
n_workers = 4 # number of workers
feature_size = 48 # feature size
in_channels = 4 # number of input channels
out_channels = 3 # number of output channels

roi_x = 96 # roi size in x direction
roi_y = 96 # roi size in y direction
roi_z = 96 # roi size in z direction
# dropout_rate = 0.0 # dropout rate
# dropout_path_rate = 0.0 # drop path rate
# RandFlipd_prob = 0.2 # RandFlipd aug probability
# RandRotate90d_prob = 0.2 # RandRotate90d aug probability
# RandScaleIntensityd_prob = 0.1 # RandScaleIntensityd aug probability
# RandShiftIntensityd_prob = 0.1 # RandShiftIntensityd aug probability
infer_overlap = 0.5 # sliding window inference overlap
lrschedule = "warmup_cosine" # type of learning rate scheduler
warmup_epochs = 5 #50 # number of warmup epochs
smooth_dr = 1e-6 # constant added to dice denominator to avoid nan
smooth_nr = 0.0 # constant added to dice numerator to avoid zero
squared_dice = True # use squared Dice
include_background = False

checkpoint_dir = "./runs/train/out_dec1_trained/model_epoch_38_0.7663.pt" # checkpoint dir to continue training from saved checkpoint
use_saved_epoch = False
save_checkpoint = True # save checkpoint during training
load_pretrained = False # Load original pretrained model from pretrained_dir, if False, load pretrained model from checkpoint
pretrained_dir = "./pretrained_models/fold1_f48_ep300_4gpu_dice0_9059/" # pretrained checkpoint directory
pretrained_model_name = "model.pt" # pretrained model name

use_grad_checkpoint = False # use gradient checkpointing to save memory
use_amp = True # use auto mixed precision for training

def main():
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"running on device: {device}")

    train_loader = get_loader(train_data_dir, train_json_list, False, roi_x, roi_y, roi_z, batch_size, n_workers)
    valid_loader = get_loader(valid_data_dir, valid_json_list, True, roi_x, roi_y, roi_z, 1, n_workers)
    print("Batch size is:", batch_size, "epochs", max_epochs)

    model = SwinUNETR(
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=feature_size,
        use_checkpoint=use_grad_checkpoint,
    )

    if load_pretrained:
        model_dict = torch.load(
            os.path.join(pretrained_dir, pretrained_model_name), 
            weights_only=False
        )["state_dict"]
        model.load_state_dict(model_dict)
        print("Using pretrained weights")
    else:
        print("Using Abscess-SwinUNETR")

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total SwinUNETR parameters count", pytorch_total_params)
    print()

    model = AbscessSwinUNETR(5, 5, model, swin_in_channels=in_channels, feature_size=feature_size, freeze_all=True)
    model.unfreeze_decoders(decoder1=True, decoder2=False, decoder3=False, decoder4=False, decoder5=False)
    model.unfreeze_encoders(encoder1=True, encoder2=False, encoder3=False, encoder4=False, encoder10=False)
    # model.unfreeze_transformers()

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total Abscess-SwinUNETR parameters count", pytorch_total_params)
    print()

    best_acc = 0
    start_epoch = 0
    if checkpoint_dir is not None:
        checkpoint = torch.load(checkpoint_dir, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        if "epoch" in checkpoint and use_saved_epoch:
            start_epoch = checkpoint["epoch"]
        if "best_acc" in checkpoint and use_saved_epoch:
            best_acc = checkpoint["best_acc"]
        print("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(checkpoint_dir, start_epoch, best_acc))

    model.to(device)

    if squared_dice:
        dice_loss = DiceLoss(
            include_background=include_background, to_onehot_y=True, from_logits=True, squared_pred=True, smooth_nr=smooth_nr, smooth_dr=smooth_dr
        )
    else:
        dice_loss = DiceLoss(include_background=include_background, to_onehot_y=True, from_logits=True)

    dice_acc = DiceMetric(include_background=include_background, reduction="mean_batch", num_classes=5)

    if optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=optim_lr, weight_decay=reg_weight)
    elif optim_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=optim_lr, weight_decay=reg_weight)
    elif optim_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=optim_lr, momentum=momentum, nesterov=True, weight_decay=reg_weight
        )
    else:
        raise ValueError("Unsupported Optimization Procedure: " + str(optim_name))

    if lrschedule == "warmup_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=warmup_epochs, max_epochs=max_epochs
        )
    elif lrschedule == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        if checkpoint_dir is not None:
            scheduler.step(epoch=start_epoch)
    else:
        scheduler = None

    model_test_inferer = partial(
        sliding_window_inference,
        roi_size=(roi_x, roi_y, roi_z),
        sw_batch_size=sw_batch_size,
        predictor=model,
        overlap=infer_overlap,
        device=device,
        return_device=device,
    )

    semantic_classes = ["Dice_Val_Br", "Dice_Val_Ed", "Dice_Val_Ab", "Dice_Val_rg"]

    accuracy = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=valid_loader,
        optimizer=optimizer,
        loss_func=dice_loss,
        acc_func=dice_acc,
        batch_size=batch_size,
        logdir=logdir,
        model_inferer=model_test_inferer,
        val_every=val_every,
        save_best_checkpoint=save_checkpoint,
        scheduler=scheduler,
        start_epoch=start_epoch,
        max_epochs=max_epochs,
        semantic_classes=semantic_classes,
        device=device,
    )
    return accuracy


if __name__ == "__main__":
    main()
