# ref: https://github.com/weiaicunzai/pytorch-cifar100 (2025/01/07)
#      https://github.com/sidak/otfusion (2025/01/07)

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

CIFAR10_TRAIN_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_TRAIN_STD = (0.2023, 0.1994, 0.2010)


def get_train_dataloader(
    batch_size=16, num_workers=2, no_randomness=False, data_dir="./data", mean=CIFAR10_TRAIN_MEAN, std=CIFAR10_TRAIN_STD
):
    if no_randomness:
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        shuffle = False
        print("disabling shuffle train as well in no_randomness!")
    else:
        shuffle = True
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return data_loader


def get_test_dataloader(
    batch_size=16, num_workers=2, data_dir="./data", mean=CIFAR10_TRAIN_MEAN, std=CIFAR10_TRAIN_STD
):
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

    data_loader = DataLoader(dataset, shuffle=False, num_workers=num_workers, batch_size=batch_size)

    return data_loader
