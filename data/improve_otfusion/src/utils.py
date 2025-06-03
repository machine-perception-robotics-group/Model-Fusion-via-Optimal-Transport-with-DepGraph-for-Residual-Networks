# ref: https://github.com/weiaicunzai/pytorch-cifar100 (2025/01/07)

""" helper function

author baiyu
"""

import datetime
import os
import re

import numpy
import torch
import torch.nn as nn
import torchvision
from torch.optim.lr_scheduler import _LRScheduler


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def most_recent_folder(net_weights, fmt):
    """
    return most recent created folder under net_weights
    if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ""

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]


def most_recent_weights(weights_folder):
    """
    return most recent created weights file
    if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ""

    regex_str = r"([A-Za-z0-9]+)-([0-9]+)-(regular|best)"

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]


def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
        raise Exception("no recent weights were found")
    resume_epoch = int(weight_file.split("-")[1])

    return resume_epoch


def best_acc_weights(weights_folder):
    """
    return the best acc .pth file in given folder, if no
    best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ""

    regex_str = r"([A-Za-z0-9]+)-([0-9]+)-(regular|best)"
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == "best"]
    if len(best_files) == 0:
        return ""

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]


def run_for_batchnorm_statistics(model, train_loader, device=None, verbose=False):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.train()

    loss_function = nn.CrossEntropyLoss(reduction="sum")

    correct_1 = 0.0
    correct_5 = 0.0
    loss_total = 0.0

    with torch.no_grad():
        for image, label in train_loader:

            image = image.to(device)
            label = label.to(device)

            output = model(image)
            _, pred = output.topk(5, 1, largest=True, sorted=True)

            loss = loss_function(output, label)
            loss_total += loss.item()

            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()

            # compute top 5
            correct_5 += correct[:, :5].sum()
            # compute top1
            correct_1 += correct[:, :1].sum()

    if verbose:
        if device == "cuda":
            print("GPU INFO.....")
            print(torch.cuda.memory_summary(), end="")

        print()
        print("Top 1 err: ", 1 - correct_1 / len(train_loader.dataset))
        print("Top 5 err: ", 1 - correct_5 / len(train_loader.dataset))
        print("Parameter numbers: {}".format(sum(p.numel() for p in model.parameters())))

    top1_acc = correct_1 / len(train_loader.dataset)
    top5_acc = correct_5 / len(train_loader.dataset)
    test_loss = loss_total / len(train_loader.dataset)

    model.eval()
