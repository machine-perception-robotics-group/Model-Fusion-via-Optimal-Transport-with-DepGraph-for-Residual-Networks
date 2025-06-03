# ref: https://github.com/weiaicunzai/pytorch-cifar100 (2025/01/07)
#      https://github.com/sidak/otfusion (2025/01/07)

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)


def get_train_dataloader(
    batch_size=16,
    num_workers=2,
    no_randomness=False,
    data_dir="./data",
    mean=CIFAR100_TRAIN_MEAN,
    std=CIFAR100_TRAIN_STD,
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

    cifar100_training = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size
    )

    return cifar100_training_loader


def get_test_dataloader(
    batch_size=16, num_workers=2, data_dir="./data", mean=CIFAR100_TRAIN_MEAN, std=CIFAR100_TRAIN_STD
):

    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    cifar100_test = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(cifar100_test, shuffle=False, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader
