import os
import copy
from functools import partial

import numpy as np
import torch

from data.data_utils import get_loader
from utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from utils.sep_trainer import run_training
from utils.inferers import separate_sliding_window_inference, sliding_window_inference
from utils.losses import DiceLoss
from utils.metrics import DiceMetric
from networks.swin import SwinUNETR
from networks.model import AbscessSepSwinUNETR, BrainStripNet, AbscessSwinUNETR


def main(
    logdir = "./runs/train/lesion_out_dec1_trained_10", # directory to save the tensorboard logs
    train_brain_data_dir = "./dataset/train_brain_patches/", # dataset directory
    valid_brain_data_dir = "./dataset/brain_labeled_dataset/",
    train_brain_json_list = "./jsons/train_brain.json", # dataset json file
    valid_brain_json_list = "./jsons/valid_brain.json",
    train_lesion_data_dir = "./dataset/train_lesion_patches/", # dataset directory
    valid_lesion_data_dir = "./dataset/lesion_labeled_dataset/",
    train_lesion_json_list = "./jsons/train_lesion.json", # dataset json file
    valid_lesion_json_list = "./jsons/valid_lesion.json",

    max_epochs = 15, # max number of training epochs
    batch_size = 2, # number of batch size
    sw_batch_size = 1, #4 # number of sliding window batch size
    optim_lr = 1e-5, # 1e-4 # optimization learning rate
    optim_name = "adamw", # optimization algorithm
    reg_weight = 1e-5, # regularization weight
    momentum = 0.99, # momentum
    val_every = 1, #100 # validation frequency
    n_workers = 8, # number of workers
    feature_size = 48, # feature size

    roi_x = 96, # roi size in x direction
    roi_y = 96, # roi size in y direction
    roi_z = 96, # roi size in z direction
    infer_overlap = 0.5, # sliding window inference overlap
    lrschedule = "warmup_cosine", # type of learning rate scheduler
    warmup_epochs = 5, #50 # number of warmup epochs
    warmup_start_lr = 1e-4,
    smooth_dr = 1e-6, # constant added to dice denominator to avoid nan
    smooth_nr = 0.0, # constant added to dice numerator to avoid zero
    squared_dice = True, # use squared Dice
    include_background = True,
    genralized_dice = True,

    checkpoint_dir = "./runs/train/lesion_out_dec1_trained_8/model_epoch_9_loss_0.9127.pt", # checkpoint dir to continue training from saved checkpoint
    use_saved_epoch = False,
    save_checkpoint = True, # save checkpoint during training
    load_pretrained = True, # Load original pretrained model from pretrained_dir, if False, load pretrained model from checkpoint
    pretrained_dir = "./pretrained_models/fold1_f48_ep300_4gpu_dice0_9059/", # pretrained checkpoint directory
    pretrained_model_name = "model.pt", # pretrained model name

    use_grad_checkpoint = False, # use gradient checkpointing to save memory
    use_amp = False, # use auto mixed precision for training

    train_brain = False, # true for brain or false for lesion
):
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"running on device: {device}")

    if train_brain:
        train_loader = get_loader(train_brain_data_dir, train_brain_json_list, 2, True, False, roi_x, roi_y, roi_z, batch_size, n_workers)
        valid_loader = get_loader(valid_brain_data_dir, valid_brain_json_list, 2, True, True, roi_x, roi_y, roi_z, 1, n_workers)
    else:
        train_loader = get_loader(train_lesion_data_dir, train_lesion_json_list, 4, True, False, roi_x, roi_y, roi_z, batch_size, n_workers)
        valid_loader = get_loader(valid_lesion_data_dir, valid_lesion_json_list, 4, True, True, roi_x, roi_y, roi_z, 1, n_workers)
        
    print("Batch size is:", batch_size, "epochs", max_epochs)

    swin_model = SwinUNETR(
        in_channels=4,
        out_channels=3,
        feature_size=feature_size,
        use_checkpoint=use_grad_checkpoint,
    )

    if load_pretrained:
        model_dict = torch.load(
            os.path.join(pretrained_dir, pretrained_model_name), 
            weights_only=False
        )["state_dict"]
        swin_model.load_state_dict(model_dict)
        print("Using pretrained weights")
    else:
        print("Using Abscess-SwinUNETR")

    pytorch_total_params = sum(p.numel() for p in swin_model.parameters() if p.requires_grad)
    print("Total SwinUNETR parameters count", pytorch_total_params)
    print()
    ############################################
    # br_model = AbscessSwinUNETR(4, 5, swin_model, swin_in_channels=4, feature_size=feature_size, freeze_all=True)
    # checkpoint = torch.load("./runs/train/out_dec1_wbg_trained/model_epoch_1_0.7663.pt", weights_only=False)
    # br_model.load_state_dict(checkpoint["state_dict"])
    # br_decoder_state = copy.deepcopy(br_model.swin_unetr.decoder1.state_dict())
    ############################################

    model = AbscessSepSwinUNETR(
        brain_out_channels=2, 
        lesion_out_channels=4, 
        swin_unetr_model=swin_model,
        feature_size=feature_size, 
        freeze_all=True
    )

    model.unfreeze_outs(brain_out=False, lesion_out=True)
    model.unfreeze_decoders(brain_decoder=False, lesion_decoder=True)

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

    ###################################################################3
    # model.brain_decoder1.load_state_dict(br_decoder_state)
    #####################################################################3

    model.to(device)

    if train_brain:
        dice_loss = DiceLoss(
            include_background=include_background, from_logits=True, softmax=True, 
            squared_pred=squared_dice, smooth_nr=smooth_nr, smooth_dr=smooth_dr, generalized=genralized_dice
        )
        dice_acc = DiceMetric(include_background=True, reduction="mean_batch", num_classes=2)
    else:
        # dice_loss = DiceLoss(
        #     include_background=include_background,
        #     squared_pred=squared_dice, smooth_nr=smooth_nr, smooth_dr=smooth_dr, generalized=genralized_dice
        # )
        dice_loss = DiceLoss(
            batch_dice=True,
            ignore_empty=True,
            squared_pred=squared_dice,
            generalized=True,
            log_dice=False,
            focal_tversky=False,
            alpha=0.4,
            beta=0.6,
            gamma=1.5,
            # class_weights=[0.001, 0.1, 0.60, 0.50],
        )
        dice_acc = DiceMetric(include_background=True, reduction="mean_batch", num_classes=4)

    if optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=optim_lr)
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
            optimizer, warmup_epochs=warmup_epochs, max_epochs=max_epochs, warmup_start_lr=warmup_start_lr,
        )
    elif lrschedule == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        if checkpoint_dir is not None:
            scheduler.step(epoch=start_epoch)
    else:
        scheduler = None

    valid_type = "brain" if train_brain else "lesion"
    model_test_inferer = partial(
        separate_sliding_window_inference,
        roi_size=(roi_x, roi_y, roi_z),
        sw_batch_size=sw_batch_size,
        predictor=model,
        overlap=infer_overlap,
        device=device,
        return_device=device,
        valid_type=valid_type
    )

    accuracy = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=valid_loader,
        optimizer=optimizer,
        loss_func=dice_loss,
        acc_func=dice_acc,
        batch_size=batch_size,
        use_amp=use_amp,
        logdir=logdir,
        model_inferer=model_test_inferer,
        val_every=val_every,
        save_best_checkpoint=save_checkpoint,
        scheduler=scheduler,
        start_epoch=start_epoch,
        max_epochs=max_epochs,
        device=device,
        brain=train_brain,
    )
    return accuracy


if __name__ == "__main__":
    main()
