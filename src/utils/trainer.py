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
            "Epoch {}/{} {}/{}".format(epoch, max_epochs, idx, len(loader)),
            "loss: {:.4f}".format(run_loss.avg),
            "time {:.2f}s".format(time.time() - start_time),
        )
        start_time = time.time()
    return run_loss.avg


def val_epoch(model, loader, epoch, acc_func, device, model_inferer, max_epochs):
    model.eval()
    start_time = time.time()
    run_acc = AverageMeter()

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data, target = batch_data["image"], batch_data["label"]
            data, target = data.to(device), target.to(device)
            val_labels_list, val_outputs_list = list(), list()
            for (image, label) in zip(data, target):
                logit = model_inferer(image)
                val_labels_list.append(label)
                val_outputs_list.append(logit)

            val_output_convert = [torch.argmax(val_pred_tensor, dim=0) for val_pred_tensor in val_outputs_list]
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_list)
            acc = acc_func.aggregate()
            # acc = acc.to(device)
            
            run_acc.update(acc.cpu().numpy(), n=1)
            
            print(
                "Val {}/{} {}/{}".format(epoch, max_epochs, idx, len(loader)),
                ", Dice_TC:",
                run_acc.avg[0],
                ", Dice_WT:",
                run_acc.avg[1],
                ", Dice_ET:",
                run_acc.avg[2],
                ", time {:.2f}s".format(time.time() - start_time),
            )
            start_time = time.time()

    return run_acc.avg


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
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    batch_size,
    logdir=None,
    model_inferer=None,
    val_every=10,
    save_best_checkpoint=True,
    scheduler=None,
    start_epoch=0,
    max_epochs=300,
    semantic_classes=None,
    device="cuda",
):
    writer = None
    if logdir is not None:
        writer = SummaryWriter(log_dir=logdir)
        print("Writing Tensorboard logs to ", logdir)
        
    val_acc_max = 0.0
    for epoch in range(start_epoch, max_epochs):
        print(time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, epoch=epoch, loss_func=loss_func, device=device, max_epochs=max_epochs, batch_size=batch_size
        )
        print(
            "Final training  {}/{}".format(epoch, max_epochs - 1),
            "loss: {:.4f}".format(train_loss),
            "time {:.2f}s".format(time.time() - epoch_time),
        )
        if writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)
            
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

            Dice_TC = val_acc[0]
            Dice_WT = val_acc[1]
            Dice_ET = val_acc[2]
            print(
                "Final validation stats {}/{}".format(epoch, max_epochs - 1),
                ", Dice_TC:",
                Dice_TC,
                ", Dice_WT:",
                Dice_WT,
                ", Dice_ET:",
                Dice_ET,
                ", time {:.2f}s".format(time.time() - epoch_time),
            )

            if writer is not None:
                writer.add_scalar("Mean_Val_Dice", np.mean(val_acc), epoch)
                if semantic_classes is not None:
                    for val_channel_ind in range(len(semantic_classes)):
                        if val_channel_ind < val_acc.size:
                            writer.add_scalar(semantic_classes[val_channel_ind], val_acc[val_channel_ind], epoch)
            val_avg_acc = np.mean(val_acc)
            if val_avg_acc > val_acc_max:
                print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                val_acc_max = val_avg_acc
                if logdir is not None and save_best_checkpoint:
                    save_checkpoint(
                        model, epoch, logdir, best_acc=val_acc_max, optimizer=optimizer, scheduler=scheduler
                    )


        if scheduler is not None:
            scheduler.step()

    print("Training Finished !, Best Accuracy: ", val_acc_max)

    return val_acc_max
