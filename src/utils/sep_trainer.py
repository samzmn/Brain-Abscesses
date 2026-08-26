import os
import time
from typing import Any, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast

from utils.utils import AverageMeter
from utils.metrics import DiceMetric


def train_epoch(model: torch.nn.Module,
                loader: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer,
                epoch: int,
                loss_func: torch.nn.Module,
                device: str,
                max_epochs: int, 
                batch_size: int,
                use_amp: bool, 
                scaler: GradScaler | None,
                brain: bool,
) -> int | np.typing.NDArray[Any]:
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    # run_dice_loss = AverageMeter()
    # run_bce_loss = AverageMeter()
    # bce_loss_func = nn.BCEWithLogitsLoss()
    # ce_loss = torch.nn.CrossEntropyLoss()#weight=torch.tensor([0.01, 0.25, 0.49, 0.25], dtype=torch.float32, device=device))
    for idx, batch_data in enumerate(loader):
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["image"], batch_data["label"]
            
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        with autocast(device_type=str(device), enabled=use_amp):
            brain_logits, lesion_logits = model(data)
            if brain:
                loss = loss_func(brain_logits, target)
            else:
                # loss = loss_func(lesion_logits, target) + 0.1 * ce_loss(lesion_logits, target)
                loss = loss_func(lesion_logits, target)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        run_loss.update(loss.item(), n=batch_size)
        # run_dice_loss.update(dice_loss.item(), n=batch_size)
        # run_bce_loss.update(bce_loss.item(), n=batch_size)

        print(
            "\rEpoch {}/{} {}/{}".format(epoch+1, max_epochs, idx+1, len(loader)),
            # "dice_loss: {:.4f}".format(run_dice_loss.avg),
            # "bce_loss: {:.4f}".format(run_bce_loss.avg),
            "loss: {:.4f}".format(run_loss.avg),
            "time {:.2f}s".format(time.time() - start_time),
            end=''
        )
        start_time = time.time()

    print()
    return run_loss.avg


def val_epoch(model: torch.nn.Module,
              loader: torch.utils.data.DataLoader,
              epoch: int,
              acc_func: DiceMetric,
              device: str,
              model_inferer,
              max_epochs:int,
              use_amp: bool,
              brain: bool,
) -> torch.Tensor:
    model.eval()
    # print(
    #     model.brain_out.conv.weight.abs().mean().item()
    # )
    start_time = time.time()
    # run_acc = AverageMeter()
    # dice_metric = DiceScore(num_classes=1, average='micro')
    # iou_metric = BinaryJaccardIndex().to(device)
    # precision = BinaryPrecision().to(device)
    # recall = BinaryRecall().to(device)
    # f1 = BinaryF1Score().to(device)

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data, target = batch_data["image"], batch_data["label"]
            data, target = data.to(device), target.to(device)
            y_preds = []
            for image in data:
                with autocast(device_type=str(device), enabled=use_amp):
                    logit = model_inferer(image)
                    y_pred = torch.argmax(logit, dim=0, keepdim=True)
                    y_preds.append(y_pred[None, ...])
            y_preds = torch.concat(y_preds, dim=0)
            # print(
            #     y_preds.min().item(),
            #     y_preds.max().item(),
            #     y_preds.mean().item()
            # )
            acc_func.reset()
            acc_func.update(y_pred=y_preds, y=target)
            acc = acc_func.aggregate()

            # dice_metric.reset()
            # precision.reset()
            # recall.reset()
            # f1.reset()
            # dice_metric.update(y_preds, target)
            # precision.update(y_preds, target)
            # recall.update(y_preds, target)
            # f1.update(y_preds, target)
            # dice: torch.Tensor = dice_metric.compute()
            # iou = iou_metric(y_preds, target)
            # pre: torch.Tensor = precision.compute()
            # rec: torch.Tensor = recall.compute()
            # f1_score: torch.Tensor = f1.compute()
            if brain:
                print(
                    "Val {}/{} {}/{}".format(epoch+1, max_epochs, idx+1, len(loader)),
                    ", Dice: ",
                    acc,
                    ", MEAN: ",
                    acc.mean(),
                    # ", Dice:",
                    # dice.detach().cpu().numpy(),
                    # ", IOU:",
                    # iou,
                    # ", precision:",
                    # pre.detach().cpu().numpy(),
                    # ", recall:",
                    # rec.detach().cpu().numpy(),
                    # ", f1-score:",
                    # f1_score.detach().cpu().numpy(),
                    ", time {:.2f}s".format(time.time() - start_time),
                )
            else:
                print(
                    "Val {}/{} {}/{}".format(epoch+1, max_epochs, idx+1, len(loader)),
                    ", Dice_Bg:",
                    acc[0],
                    ", Dice_Ed:",
                    acc[1],
                    ", Dice_Ab:",
                    acc[2],
                    ", Dice_Rg:",
                    acc[3],
                    ", MEAN:",
                    acc.mean(),
                    ", time {:.2f}s".format(time.time() - start_time),
                )
            start_time = time.time()

    return acc


def save_checkpoint(model: torch.nn.Module,
                    epoch: int,
                    logdir: str,
                    filename: str="model.pt",
                    best_acc: int=0,
                    optimizer: torch.optim.Optimizer | None=None, 
                    scheduler: torch.optim.lr_scheduler._LRScheduler | None=None
) -> None:
    state_dict = model.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def run_training(
    model,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_func: torch.nn.Module,
    acc_func,
    batch_size: int,
    use_amp: bool=False,
    logdir: str | None=None,
    model_inferer=None,
    val_every: int=10,
    save_best_checkpoint: bool=True,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    start_epoch: int=0,
    max_epochs: int=300,
    semantic_classes: List[str]=None,
    device: torch.types.Device = "cuda",
    brain: bool = True,
) -> float:
    writer = None
    if logdir is not None:
        writer = SummaryWriter(log_dir=logdir)
        print("Writing Tensorboard logs to ", logdir)

    scaler = None
    if use_amp:
        scaler = GradScaler(device=device)
        
    val_acc_max = 0.0
    train_loss_min = np.inf
    for epoch in range(start_epoch, max_epochs):
        print(time.ctime(), "Epoch:", epoch+1)
        epoch_time = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, epoch, loss_func, device, max_epochs, batch_size, use_amp, scaler, brain
        )
        print(
            "Final training  {}/{}".format(epoch+1, max_epochs),
            "loss: {:.4f}".format(train_loss),
            "time {:.2f}s".format(time.time() - epoch_time),
            # f"LR scheduler {scheduler.get_last_lr() if scheduler is not None else 0.}, ",
            f"LR {optimizer.param_groups[0]['lr']}"
        )
        if writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)
            writer.add_scalar("Learning_Rate", optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar("Epoch_Time", time.time() - epoch_time, epoch)

        if train_loss < train_loss_min:
            train_loss_min = train_loss
            save_checkpoint(
                model, epoch, logdir, filename=f"model_epoch_{epoch+1}_loss_{train_loss_min:.4f}.pt", 
                best_acc=val_acc_max, optimizer=optimizer, scheduler=scheduler
            )

        if (epoch + 1) % val_every == 0:
            epoch_time = time.time()
            val_acc = val_epoch(
                model,
                val_loader,
                epoch,
                acc_func,
                device,
                model_inferer,
                max_epochs,
                use_amp,
                brain
            ).numpy()

            val_avg_acc = val_acc.mean()

            if writer is not None:
                writer.add_scalar("Mean_Val_Dice", val_avg_acc, epoch)
                if brain:
                    semantic_classes = ["Dice_Val_Bg", "Dice_Val_Br"]
                else:
                    semantic_classes = ["Dice_Val_Bg", "Dice_Val_Ed", "Dice_Val_Ab", "Dice_Val_Rg"]
                for val_channel_ind in range(len(semantic_classes)):
                    if val_channel_ind < len(val_acc):
                        writer.add_scalar(semantic_classes[val_channel_ind], val_acc[val_channel_ind], epoch)
        
            if val_avg_acc > val_acc_max:
                print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                val_acc_max = val_avg_acc
                if logdir is not None and save_best_checkpoint:
                    save_checkpoint(
                        model, epoch, logdir, filename=f"model_epoch_{epoch+1}_{val_acc_max:.4f}.pt", 
                        best_acc=val_acc_max, optimizer=optimizer, scheduler=scheduler
                    )

        if scheduler is not None:
            scheduler.step()

    print("Training Finished !, Best Accuracy: ", val_acc_max)

    return val_acc_max
