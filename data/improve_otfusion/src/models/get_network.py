# ref: https://github.com/weiaicunzai/pytorch-cifar100 (2025/01/07)

import sys


def get_network(model_name, class_num, device=None):
    """return given network"""

    if model_name == "vgg16":
        from .vgg import vgg16_bn

        net = vgg16_bn(class_num)
    elif model_name == "vgg13":
        from .vgg import vgg13_bn

        net = vgg13_bn(class_num)
    elif model_name == "vgg11":
        from .vgg import vgg11_bn

        net = vgg11_bn(class_num)
    elif model_name == "vgg19":
        from .vgg import vgg19_bn

        net = vgg19_bn(class_num)
    elif model_name == "resnet18":
        from .resnet import resnet18

        net = resnet18(class_num)
    elif model_name == "resnet34":
        from .resnet import resnet34

        net = resnet34(class_num)
    elif model_name == "resnet50":
        from .resnet import resnet50

        net = resnet50(class_num)
    elif model_name == "resnet101":
        from .resnet import resnet101

        net = resnet101(class_num)
    elif model_name == "resnet152":
        from .resnet import resnet152

        net = resnet152(class_num)
    elif model_name == "resnet18_nobn":
        from .resnet_otfusion import ResNet18

        net = ResNet18(num_classes=class_num, use_batchnorm=False, linear_bias=True)
    elif model_name == "vgg11_nobn":
        from .vgg_otfusion import VGG

        net = VGG("VGG11", num_classes=class_num, batch_norm=False, bias=True)
    elif model_name == "resnet18_nobn_nobias":
        from .resnet_otfusion import ResNet18

        net = ResNet18(num_classes=class_num, use_batchnorm=False, linear_bias=False)
    elif model_name == "vgg11_nobn_nobias":
        from .vgg_otfusion import VGG

        net = VGG("VGG11", num_classes=class_num, batch_norm=False, bias=False)
    else:
        print("the network name you have entered is not supported yet")
        sys.exit()

    if device is not None:
        net = net.to(device)

    return net
