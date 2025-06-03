# ref: https://github.com/weiaicunzai/pytorch-cifar100 (2025/01/07)

""" train network using pytorch

author baiyu
"""

import argparse
import os
import random
import sys
import time
from datetime import datetime
from os.path import join

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models.get_network import get_network
from torch.utils.tensorboard import SummaryWriter
from utils import WarmUpLR, best_acc_weights, last_epoch, most_recent_folder, most_recent_weights

IMAGENET_DIR = "/data/imagenet/ILSVRC2012"


def get_class_num(dataset_name):
    if dataset_name == "cifar100":
        class_num = 100
    elif dataset_name == "cifar10":
        class_num = 10
    elif dataset_name == "imagenet":
        class_num = 1000
    else:
        print("the dataset name you have entered is not supported yet")
        sys.exit()

    return class_num


def train(epoch, device):

    start = time.time()
    net.train()
    for batch_index, (images, labels) in enumerate(train_loader):

        labels = labels.to(device)
        images = images.to(device)

        optimizer.zero_grad()
        outputs = net(images)

        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(train_loader) + batch_index + 1

        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if "weight" in name:
                writer.add_scalar("LastLayerGradients/grad_norm2_weights", para.grad.norm(), n_iter)
            if "bias" in name:
                writer.add_scalar("LastLayerGradients/grad_norm2_bias", para.grad.norm(), n_iter)

        print(
            "Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}".format(
                loss.item(),
                optimizer.param_groups[0]["lr"],
                epoch=epoch,
                trained_samples=batch_index * args.b + len(images),
                total_samples=len(train_loader.dataset),
            )
        )

        # update training loss for each iteration
        writer.add_scalar("Train/loss", loss.item(), n_iter)

        if epoch <= args.warm:
            warmup_scheduler.step()

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()

    print("epoch {} training time consumed: {:.2f}s".format(epoch, finish - start))


@torch.no_grad()
def eval_training(epoch, tb, device):

    start = time.time()
    net.eval()

    test_loss = 0.0  # cost function error
    correct = 0.0

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = net(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    print("GPU INFO.....")
    print(torch.cuda.memory_summary(), end="")
    print("Evaluating Network.....")
    print(
        "Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s".format(
            epoch, test_loss / len(test_loader.dataset), correct.float() / len(test_loader.dataset), finish - start
        )
    )
    print()

    # add informations to tensorboard
    if tb:
        writer.add_scalar("Test/Average loss", test_loss / len(test_loader.dataset), epoch)
        writer.add_scalar("Test/Accuracy", correct.float() / len(test_loader.dataset), epoch)

    return correct.float() / len(test_loader.dataset)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-net",
        type=str,
        required=True,
        help="net type (vgg11|resnet18|resnet50|vgg11_nobn|resnet18_nobn|resnet50_nobn)",
    )
    parser.add_argument("-dataset", type=str, required=True, help="dataset type (cifar10|cifar100|imagenet)")
    parser.add_argument("-num_workers", type=int, default=4, help="number of workers in training")
    parser.add_argument("-b", type=int, default=128, help="batch size for dataloader")
    parser.add_argument("-lr", type=float, default=0.1, help="initial learning rate")
    parser.add_argument("-warm", type=int, default=1, help="warm up training phase")

    parser.add_argument("-checkpoint_dir", type=str, default="checkpoint", help="directory to save weights file")
    parser.add_argument("-log_dir", type=str, default="runs", help="tensorboard log dir")
    parser.add_argument("-total_epoch", type=int, default=200, help="total training epoches")
    parser.add_argument("-save_per_epoch", type=int, default=20, help="save weights file per save_per_epoch epoch")
    parser.add_argument(
        "-start_save_best_epoch",
        type=int,
        default=180,
        help="start to save best performance model after start_save_best_epoch epoch",
    )

    parser.add_argument("-resume", action="store_true", default=False, help="resume training")
    parser.add_argument("-seed", type=int, default=42, help="random seed")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_format = "%A_%d_%B_%Y_%Hh_%Mm_%Ss"
    time_now = datetime.now().strftime(data_format)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    class_num = get_class_num(args.dataset)
    net = get_network(args.net, class_num, device)

    if args.dataset == "cifar100":
        class_num = 100
        from datasets.cifar100 import get_test_dataloader, get_train_dataloader

        train_loader = get_train_dataloader(num_workers=args.num_workers, batch_size=args.b)

        test_loader = get_test_dataloader(num_workers=1, batch_size=args.b)
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    elif args.dataset == "cifar10":
        class_num = 10
        from datasets.cifar10 import get_test_dataloader, get_train_dataloader

        train_loader = get_train_dataloader(num_workers=args.num_workers, batch_size=args.b)

        test_loader = get_test_dataloader(num_workers=1, batch_size=args.b)
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    elif args.dataset == "imagenet":
        class_num = 1000
        # ref: https://github.com/pytorch/examples/blob/main/imagenet/main.py (2025/01/07)
        from datasets.imagenet import get_test_dataloader, get_train_dataloader

        train_loader = get_train_dataloader(
            data_dir=join(IMAGENET_DIR, "train"), num_workers=args.num_workers, batch_size=args.b
        )

        test_loader = get_test_dataloader(data_dir=join(IMAGENET_DIR, "val"), num_workers=1, batch_size=args.b)
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        train_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        print("the dataset name you have entered is not supported yet")
        sys.exit()

    loss_function = nn.CrossEntropyLoss()
    iter_per_epoch = len(train_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    if args.resume:
        recent_folder = most_recent_folder(os.path.join(args.checkpoint_dir, args.dataset, args.net), fmt=data_format)
        if not recent_folder:
            raise Exception("no recent folder were found")

        checkpoint_path = os.path.join(args.checkpoint_dir, args.dataset, args.net, recent_folder)

    else:
        checkpoint_path = os.path.join(args.checkpoint_dir, args.dataset, args.net, time_now)

    # use tensorboard
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    # since tensorboard can't overwrite old values
    # so the only way is to create a new tensorboard log
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.dataset, args.net, time_now))
    input_tensor = torch.Tensor(1, 3, 32, 32)
    input_tensor = input_tensor.to(device)
    writer.add_graph(net, input_tensor)

    # create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, "{net}-{epoch}-{type}.pth")

    best_acc = 0.0
    if args.resume:
        best_weights = best_acc_weights(os.path.join(args.checkpoint_dir, args.dataset, args.net, recent_folder))
        if best_weights:
            weights_path = os.path.join(args.checkpoint_dir, args.dataset, args.net, recent_folder, best_weights)
            print("found best acc weights file:{}".format(weights_path))
            print("load best training file to test acc...")
            net.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(epoch=0, tb=False, device=device)
            print("best acc is {:0.2f}".format(best_acc))

        recent_weights_file = most_recent_weights(
            os.path.join(args.checkpoint_dir, args.dataset, args.net, recent_folder)
        )
        if not recent_weights_file:
            raise Exception("no recent weights file were found")
        weights_path = os.path.join(args.checkpoint_dir, args.dataset, args.net, recent_folder, recent_weights_file)
        print("loading weights file {} to resume training.....".format(weights_path))
        net.load_state_dict(torch.load(weights_path))

        resume_epoch = last_epoch(os.path.join(args.checkpoint_dir, args.dataset, args.net, recent_folder))

    for epoch in range(1, args.total_epoch + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        if args.resume:
            if epoch <= resume_epoch:
                continue

        train(epoch=epoch, device=device)
        acc = eval_training(epoch=epoch, tb=True, device=device)

        if epoch > args.start_save_best_epoch and best_acc < acc:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type="best")
            print("saving weights file to {}".format(weights_path))
            torch.save(net.state_dict(), weights_path)
            best_acc = acc

        if not epoch % args.save_per_epoch:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type="regular")
            print("saving weights file to {}".format(weights_path))
            torch.save(net.state_dict(), weights_path)

    weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type="last")
    print("saving weights file to {}".format(weights_path))
    torch.save(net.state_dict(), weights_path)

    writer.close()
