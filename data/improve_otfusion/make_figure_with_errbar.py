import os
from glob import glob
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

result_root_dir = "./results"


###################################################################################################
# それぞれの手法について図を作成
###################################################################################################
def make_figure(result_dir):
    print(f"\n画像を作成: {result_dir}")
    result_filepath_list = glob(join(result_dir, "*", "results.csv"))

    align_acc_list = []
    align_loss_list = []
    vanilla_acc_list = []
    vanilla_loss_list = []
    wb = None

    for result_filepath in result_filepath_list:
        df = pd.read_csv(result_filepath)
        wb = df.wb
        align_acc_list.append(df.align_acc_top1.to_numpy())
        align_loss_list.append(df.align_loss.to_numpy())
        vanilla_acc_list.append(df.vanilla_acc_top1.to_numpy())
        vanilla_loss_list.append(df.vanilla_loss.to_numpy())

    # Top-1 accuracyの図を作成
    align_acc_mean = np.mean(align_acc_list, axis=0)
    align_acc_std = np.std(align_acc_list, axis=0)
    vanilla_acc_mean = np.mean(vanilla_acc_list, axis=0)
    vanilla_acc_std = np.std(vanilla_acc_list, axis=0)

    plt.errorbar(
        wb,
        vanilla_acc_mean,
        yerr=vanilla_acc_std,
        capsize=5,
        fmt="or:",
        markersize=5,
        ecolor="r",
        markeredgecolor="r",
        label="Vanilla",
    )
    plt.errorbar(
        wb,
        align_acc_mean,
        yerr=align_acc_std,
        capsize=5,
        fmt="ob:",
        markersize=5,
        ecolor="b",
        markeredgecolor="b",
        label="Align",
    )
    plt.ylabel("Test accuracy")
    plt.xlabel(r"Weight towards model_B ($w_b$)")
    plt.ylim(0, 1)
    plt.legend()
    save_filepath = join(result_dir, "accuracy.png")
    plt.savefig(save_filepath)
    plt.clf()
    plt.close()

    # Lossの図を作成
    align_loss_mean = np.mean(align_loss_list, axis=0)
    align_loss_std = np.std(align_loss_list, axis=0)
    vanilla_loss_mean = np.mean(vanilla_loss_list, axis=0)
    vanilla_loss_std = np.std(vanilla_loss_list, axis=0)

    plt.errorbar(
        wb,
        vanilla_loss_mean,
        yerr=vanilla_loss_std,
        capsize=5,
        fmt="or:",
        markersize=5,
        ecolor="r",
        markeredgecolor="r",
        label="Vanilla",
    )
    plt.errorbar(
        wb,
        align_loss_mean,
        yerr=align_loss_std,
        capsize=5,
        fmt="ob:",
        markersize=5,
        ecolor="b",
        markeredgecolor="b",
        label="Align",
    )
    plt.ylabel("Loss")
    plt.xlabel(r"Weight towards model_B ($w_b$)")
    plt.legend()
    save_filepath = join(result_dir, "loss.png")
    plt.savefig(save_filepath)
    plt.clf()
    plt.close()

    print(f"完了")


method_list = ["activation_based", "weight_based"]
dataset_list = ["cifar10", "cifar100"]
model_list = [
    "resnet18",
    "resnet18_wo_recalc_batchnorm_stats",
    "resnet18_nobn_wo_recalc_batchnorm_stats",
    "vgg11",
    "vgg11_wo_recalc_batchnorm_stats",
    "vgg11_nobn_wo_recalc_batchnorm_stats",
    "resnet50",
    "resnet50_wo_recalc_batchnorm_stats",
]


for method in method_list:
    for dataset in dataset_list:
        for model in model_list:
            result_dir = f"{result_root_dir}/{method}/{dataset}/{model}/"
            make_figure(result_dir)


###################################################################################################
# それぞれの手法をまとめた図を作成
###################################################################################################

method_list = ["activation_based", "weight_based"]
dataset_list = ["cifar10", "cifar100"]
model_list = [
    "resnet18",
    "vgg11",
    "resnet50",
]

label_dict = {
    "resnet18": "ResNet-18",
    "vgg11": "VGG-11",
    "resnet50": "ResNet-50",
    "activation_based": "act",
    "weight_based": "wts",
}

color_dict = {
    "ResNet-18 (act)": "g",
    "ResNet-18 (wts)": "c",
    "VGG-11 (act)": "m",
    "VGG-11 (wts)": "y",
    "ResNet-50 (act)": "b",
    "ResNet-50 (wts)": "r",
}


def make_summarized_figure(result_root_dir, dataset, method_list, model_list):
    print(f"\n{dataset}の画像を作成")

    # Top-1 accuracyの図を作成
    print(f"\naccuracyの画像")
    for method in method_list:
        for model in model_list:
            result_dir = f"{result_root_dir}/{method}/{dataset}/{model}"
            print(result_dir)
            result_filepath_list = glob(join(result_dir, "*", "results.csv"))

            align_acc_list = []
            wb = None

            for result_filepath in result_filepath_list:
                df = pd.read_csv(result_filepath)
                wb = df.wb
                align_acc_list.append(df.align_acc_top1.to_numpy())

            align_acc_mean = np.mean(align_acc_list, axis=0)
            align_acc_std = np.std(align_acc_list, axis=0)

            label = f"{label_dict[model]} ({label_dict[method]})"
            color = color_dict[label]

            plt.errorbar(
                wb,
                align_acc_mean,
                yerr=align_acc_std,
                capsize=5,
                fmt=f"o{color}:",
                markersize=5,
                ecolor=color,
                markeredgecolor=color,
                label=label,
            )
    plt.ylabel("Test accuracy")
    plt.xlabel(r"Weight towards model_B ($w_b$)")
    # plt.ylim(0, 1)
    plt.legend()
    save_filepath = join(result_root_dir, f"{dataset}_accuracy.png")
    print(f"{save_filepath}に保存")
    plt.savefig(save_filepath)
    plt.clf()
    plt.close()
    print("完了")

    # Lossの図を作成
    print(f"\nlossの画像")
    for method in method_list:
        for model in model_list:
            result_dir = f"{result_root_dir}/{method}/{dataset}/{model}"
            print(result_dir)
            result_filepath_list = glob(join(result_dir, "*", "results.csv"))

            align_loss_list = []
            wb = None

            for result_filepath in result_filepath_list:
                df = pd.read_csv(result_filepath)
                wb = df.wb
                align_loss_list.append(df.align_loss.to_numpy())

            align_loss_mean = np.mean(align_loss_list, axis=0)
            align_loss_std = np.std(align_loss_list, axis=0)

            label = f"{label_dict[model]} ({label_dict[method]})"
            color = color_dict[label]

            plt.errorbar(
                wb,
                align_loss_mean,
                yerr=align_loss_std,
                capsize=5,
                fmt=f"o{color}:",
                markersize=5,
                ecolor=color,
                markeredgecolor=color,
                label=label,
            )
    plt.ylabel("Loss")
    plt.xlabel(r"Weight towards model_B ($w_b$)")
    plt.legend()
    save_filepath = join(result_root_dir, f"{dataset}_loss.png")
    print(f"{save_filepath}に保存")
    plt.savefig(save_filepath)
    plt.clf()
    plt.close()

    print("完了")


for dataset in dataset_list:
    make_summarized_figure(result_root_dir, dataset, method_list, model_list)
