import os
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from utils.utils import AverageMeter


def train_epoch(model, loader, optimizer, epoch, loss_func, device, max_epochs, batch_size):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["image"], batch_data["label"]
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        logits = model(data)
        loss = loss_func(logits, target)

        loss.backward()
        optimizer.step()
        run_loss.update(loss.item(), n=batch_size)

        print(
            "\rEpoch {}/{} {}/{}".format(epoch+1, max_epochs, idx+1, len(loader)),
            "loss: {:.4f}".format(run_loss.avg),
            "time {:.2f}s".format(time.time() - start_time),
            end=''
        )
        start_time = time.time()
    print()
    return run_loss.avg


def val_epoch(model, loader, epoch, acc_func, device, model_inferer, max_epochs):
    model.eval()
    start_time = time.time()
    # run_acc = AverageMeter()

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data, target = batch_data["image"], batch_data["label"]
            data, target = data.to(device), target.to(device)
            for image in data:
                logit = model_inferer(image)
            y_pred = torch.argmax(logit, dim=0, keepdim=True)
            y_pred, target = y_pred.unsqueeze(0), target.unsqueeze(0)
            # print("y_pred shape: ", y_pred.shape)
            # print("target shape: ", target.shape)
            acc_func.reset()
            acc_func.update(y_pred=y_pred, y=target)
            acc = acc_func.aggregate()
            # acc = acc.to(device)
            # print("acc lengh: ", len(acc))
            # run_acc.update(acc.cpu().numpy(), n=1)
            # print("length of run_acc", run_acc.avg)
            print(
                "Val {}/{} {}/{}".format(epoch+1, max_epochs, idx+1, len(loader)),
                ", Dice_Br:",
                acc[0],
                # run_acc.avg[0],
                ", Dice_Ed:",
                acc[1],
                # run_acc.avg[1],
                ", Dice_Ab:",
                acc[2],
                # run_acc.avg[2],
                ", Dice_Rg:",
                acc[3],
                ", time {:.2f}s".format(time.time() - start_time),
            )
            start_time = time.time()

    # return run_acc.avg
    return acc


def save_checkpoint(model, epoch, logdir, filename="model.pt", best_acc=0, optimizer=None, scheduler=None):
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
    loss_func,
    acc_func,
    batch_size,
    logdir=None,
    model_inferer=None,
    val_every=10,
    save_best_checkpoint=True,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    start_epoch=0,
    max_epochs=300,
    semantic_classes=None,
    device = "cuda",
):
    writer = None
    if logdir is not None:
        writer = SummaryWriter(log_dir=logdir)
        print("Writing Tensorboard logs to ", logdir)
        
    val_acc_max = 0.0
    for epoch in range(start_epoch, max_epochs):
        print(time.ctime(), "Epoch:", epoch+1)
        epoch_time = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, epoch=epoch, loss_func=loss_func, device=device, max_epochs=max_epochs, batch_size=batch_size
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
            
        if (epoch + 1) % val_every == 0:
            epoch_time = time.time()
            val_acc = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                model_inferer=model_inferer,
                device=device,
                max_epochs=max_epochs
            )

            Dice_Br = val_acc[0]
            Dice_Ed = val_acc[1]
            Dice_Ab = val_acc[2]
            Dice_Rg = val_acc[3]
            print(
                "Final validation stats {}/{}".format(epoch+1, max_epochs),
                ", Dice_Br:",
                Dice_Br,
                ", Dice_Ed:",
                Dice_Ed,
                ", Dice_Ab:",
                Dice_Ab,
                ", Dice_Rg:",
                Dice_Rg,
                ", time {:.2f}s".format(time.time() - epoch_time),
            )

            if writer is not None:
                writer.add_scalar("Mean_Val_Dice", np.array(val_acc.mean()), epoch)
                if semantic_classes is not None:
                    for val_channel_ind in range(len(semantic_classes)):
                        if val_channel_ind < len(val_acc):
                            writer.add_scalar(semantic_classes[val_channel_ind], val_acc[val_channel_ind], epoch)
            val_avg_acc = np.array(val_acc.mean())
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
