# ref: https://github.com/pytorch/examples/blob/main/imagenet/main.py (2025/01/07)

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

IMAGENET_TRAIN_MEAN = (0.485, 0.456, 0.406)
IMAGENET_TRAIN_STD = (0.229, 0.224, 0.225)


def get_train_dataloader(
    data_dir, batch_size=16, num_workers=2, no_randomness=False, mean=IMAGENET_TRAIN_MEAN, std=IMAGENET_TRAIN_STD
):
    if no_randomness:
        transform_train = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
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
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=transform_train)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return data_loader


def get_test_dataloader(data_dir, batch_size=16, num_workers=2, mean=IMAGENET_TRAIN_MEAN, std=IMAGENET_TRAIN_STD):
    transform_test = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean, std)]
    )

    dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=transform_test)

    data_loader = DataLoader(dataset, shuffle=False, num_workers=num_workers, batch_size=batch_size)

    return data_loader
