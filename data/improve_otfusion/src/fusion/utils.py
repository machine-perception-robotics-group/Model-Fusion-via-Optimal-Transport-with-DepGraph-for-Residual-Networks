import copy
import sys
from dataclasses import dataclass
from os.path import dirname, join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..models.get_network import get_network
from ..test_model import test
from ..utils import run_for_batchnorm_statistics

# DepGraphのパッケージ読み込み
ext_pkg_dir = dirname(dirname(__file__))
sys.path.append(join(ext_pkg_dir, "ext_pkg", "Torch-Pruning"))
import torch_pruning as tp


def get_datasets(dataset_name):
    # データセット読み込み
    if dataset_name == "cifar10":
        from ..datasets.cifar10 import get_test_dataloader, get_train_dataloader

        num_class = 10
        example_inputs = torch.randn(1, 3, 32, 32)

    elif dataset_name == "cifar100":
        from ..datasets.cifar100 import get_test_dataloader, get_train_dataloader

        num_class = 100
        example_inputs = torch.randn(1, 3, 32, 32)

    elif dataset_name == "imagenet":
        from ..datasets.imagenet import get_test_dataloader, get_train_dataloader

        num_class = 1000
        example_inputs = torch.randn(1, 3, 224, 224)
    else:
        print("the dataset name you have entered is not supported yet")
        sys.exit()

    return get_test_dataloader, get_train_dataloader, num_class, example_inputs


def get_model(model_name, num_class, device, model_filepath, is_otfusion_model, test_loader):
    print(f"\nモデル読み込み: {model_filepath}")
    model = get_network(model_name, num_class, device)
    if is_otfusion_model:
        state_a = torch.load(
            model_filepath,
            map_location=(lambda s, _: torch.serialization.default_restore_location(s, "cpu")),
        )
        model.load_state_dict(state_a["model_state_dict"])
        model.to(device)
        model.eval()
    else:
        model.load_state_dict(torch.load(model_filepath))
        model.to(device)
        model.eval()
    acc_top1, acc_top5, loss = test(model, test_loader, device=device)
    print(f"acc_top1: {acc_top1}")

    return model, acc_top1


def get_group_info(model_a, example_inputs, skip_last=True):
    """
    モデルをDepGraphを使って枝刈りした結果から、パラメータ形状の変化からネットワークのつながりを取得する
    """

    model = copy.deepcopy(model_a)

    DG = tp.DependencyGraph().build_dependency(model.to("cpu"), example_inputs=example_inputs)

    group_id__layer_names_dict = {}
    group_id__param_names_dict = {}
    param_name__group_id_dict = {}
    param_name__prev_group_id_dict = {}
    # 作成した Dependency Graph の各グループについて DepGraph を使った枝刈りを行う
    for group_id, group in enumerate(DG.get_all_groups(root_module_types=[nn.Conv2d, nn.Linear])):
        if skip_last and (group_id == 0):
            # group_id = 0 のグループの出力は各クラスの尤度となるので固定
            continue

        param_dict_orig = {k: v for k, v in model.state_dict().items() if len(v.shape) != 0}

        idxs = [0]  # pruning indices
        group.prune(idxs=idxs)

        param_dict_pruned = {k: v for k, v in model.state_dict().items() if len(v.shape) != 0}

        param_names = []
        for param_name in param_dict_orig.keys():

            is_param_1d = len(param_dict_orig[param_name].shape) == 1

            # パラメータが1次元の場合、右側に変換行列が掛からないので扱いが異なる
            if is_param_1d:
                out_feature_num_orig = param_dict_orig[param_name].shape[0]
                out_feature_num_pruned = param_dict_pruned[param_name].shape[0]
            else:
                in_feature_num_orig = param_dict_orig[param_name].shape[1]
                in_feature_num_pruned = param_dict_pruned[param_name].shape[1]

                out_feature_num_orig = param_dict_orig[param_name].shape[0]
                out_feature_num_pruned = param_dict_pruned[param_name].shape[0]

            if (in_feature_num_orig != in_feature_num_pruned) & (not is_param_1d):
                # 枝刈りで入力チャンネルの形状が変化したパラメータをリストに追加
                param_name__prev_group_id_dict[param_name] = group_id

            if out_feature_num_orig != out_feature_num_pruned:
                # 枝刈りで出力チャンネルの形状が変化したパラメータをリストに追加
                param_name__group_id_dict[param_name] = group_id
                param_names.append(param_name)

        # パラメータ名末尾の .weight や .bias を削除
        layer_names = [".".join(name.split(".")[:-1]) for name in param_names]
        # 重複を削除
        layer_names = list(set(layer_names))

        group_id__layer_names_dict[group_id] = layer_names
        group_id__param_names_dict[group_id] = param_names

    return (
        group_id__layer_names_dict,
        group_id__param_names_dict,
        param_name__group_id_dict,
        param_name__prev_group_id_dict,
    )


class GroundMetric:
    def __init__(self, squared=False, normalize=False):
        self.squared = squared
        self.normalize = normalize

    def _cost_matrix_xy(self, x, y, p=2, squared=True, device="cpu"):

        x_col = x.unsqueeze(1)
        y_lin = y.unsqueeze(0)
        # c = torch.sum((torch.abs(x_col - y_lin)) ** p, 2)
        # メモリ不足対策
        c = torch.zeros([x_col.shape[0], y_lin.shape[1]])
        for y_id in range(y_lin.shape[1]):
            # とても時間がかかる場合があるので、GPU を使えるようにした
            c[:, y_id] = (
                torch.sum(torch.abs(x_col.to(device) - y_lin[:, y_id, :].to(device)) ** p, 2).squeeze().to("cpu")
            )

        if not squared:
            c = c ** (1 / 2)
        return c

    def _get_euclidean(self, coordinates, other_coordinates, device):

        matrix = self._cost_matrix_xy(coordinates, other_coordinates, squared=self.squared, device=device)

        return matrix

    def _normed_vecs(self, vecs, eps=1e-9):
        norms = torch.norm(vecs, dim=-1, keepdim=True)
        return vecs / (norms + eps)

    def get_metric(self, coordinates, other_coordinates, device):
        return self._get_euclidean(coordinates, other_coordinates, device)

    def process(self, coordinates, other_coordinates=None, device="cpu"):
        if self.normalize:
            coordinates = self._normed_vecs(coordinates)
            other_coordinates = self._normed_vecs(other_coordinates)
        ground_metric_matrix = self.get_metric(coordinates, other_coordinates, device)

        return ground_metric_matrix


@dataclass
class FusionGroupInfo:
    group_id: int
    layer_names: list = None
    param_names: list = None
    optimal_transport_mat: torch.Tensor = None


@dataclass
class AlignParamInfo:
    param_name: str
    param_shape: torch.Tensor
    param_data: torch.Tensor
    optimal_transport_mat: torch.Tensor = None
    prev_optimal_transport_mat: torch.Tensor = None


def model_average(model_a, model_b, model_avg, b_weight=0.5):
    state_dict_a = model_a.state_dict()
    state_dict_b = model_b.state_dict()

    state_dict_avg = model_avg.state_dict()
    for k in state_dict_avg.keys():
        if "running_var" in k:
            # 定数 a, b に対し、確率変数 X, Y は独立であり、X ∼ N(mu1, sigma1^2), Y ∼ N(mu2, sigma2^2) ならば、
            # aX + bY -> N(a*mu1 + b*mu2, a^2*sigma1^2 + b^2*sigma2^2)
            # (実際は独立ではないので、上の関係は成り立たないと思われる)
            state_dict_avg[k] = ((1 - b_weight) ** 2) * state_dict_a[k] + (b_weight**2) * state_dict_b[k]
        else:
            state_dict_avg[k] = (1 - b_weight) * state_dict_a[k] + b_weight * state_dict_b[k]
    model_avg.load_state_dict(state_dict_avg)


def reset_bn(model):
    # https://stackoverflow.com/questions/68351686/how-to-drop-running-stats-to-default-value-for-norm-layer-in-pytorch
    # https://github.com/pytorch/pytorch/blob/15be5483c0222ca9fbb596e011eec41ae4061bcc/torch/nn/modules/batchnorm.py#L54
    for m in model.modules():
        if type(m) == nn.BatchNorm2d:
            m.reset_running_stats()


def fusion_interpolation(
    model_a,
    model_b,
    model_name,
    num_class,
    device,
    divide_num,
    test_loader,
    recalc_batchnorm_stats,
    train_loader_bn,
):
    model_b_weight_array = np.linspace(0, 1, divide_num)
    model_avg = get_network(model_name, num_class)
    model_avg.to(device)
    acc_top1_list = []
    acc_top5_list = []
    loss_list = []
    model_avg.eval()
    for w in model_b_weight_array:
        model_average(model_a, model_b, model_avg, b_weight=w)

        if recalc_batchnorm_stats:
            reset_bn(model_avg)
            run_for_batchnorm_statistics(model_avg, train_loader_bn, device=device)

        acc_top1_vanilla, acc_top5_vanilla, loss_vanilla = test(model_avg, test_loader, device=device)
        acc_top1_list.append(acc_top1_vanilla)
        acc_top5_list.append(acc_top5_vanilla)
        loss_list.append(loss_vanilla)
        print(f"wb {w:.04f}, acc_top1 {acc_top1_vanilla:.04f}")
    return acc_top1_list, acc_top5_list, loss_list, model_b_weight_array


def organize_info(model_a, param_name__group_id_dict, group_info_dict, param_name__prev_group_id_dict):
    param_info_dict = {}
    for param_name_a, param_data_a in model_a.state_dict().items():

        if len(param_data_a.shape) == 0:
            continue

        param_name = param_name_a
        param_shape = param_data_a.shape
        param_data = param_data_a

        if len(param_data_a.shape) > 1:
            param_data = param_data.data.view(param_data_a.shape[0], param_data_a.shape[1], -1)
            # [0] out channels, [1] in channels

        param_info = AlignParamInfo(param_name=param_name, param_shape=param_shape, param_data=param_data)
        param_info_dict[param_name] = param_info

    for k, v in param_info_dict.items():
        if k in param_name__group_id_dict.keys():
            v.optimal_transport_mat = group_info_dict[param_name__group_id_dict[k]].optimal_transport_mat

        if k in param_name__prev_group_id_dict.keys():
            v.prev_optimal_transport_mat = group_info_dict[param_name__prev_group_id_dict[k]].optimal_transport_mat
    return param_info_dict


def get_aligned_model(model_name, num_class, param_info_dict, device):
    aligned_dict = {}
    for param_name, param_info in param_info_dict.items():
        param_shape = param_info.param_shape

        if param_info.prev_optimal_transport_mat is None:
            aligned_wt = param_info.param_data
        else:
            param_data = param_info.param_data
            T_var = param_info.prev_optimal_transport_mat
            if len(param_info.param_shape) > 2:
                T_var = T_var.unsqueeze(0).repeat(param_data.shape[2], 1, 1)
                aligned_wt = torch.bmm(param_data.permute(2, 0, 1), T_var).permute(1, 2, 0)
            else:
                param_data = param_data.reshape(param_shape)
                aligned_wt = torch.matmul(param_data, T_var)

        if param_info.optimal_transport_mat is None:
            t_fc0_model = aligned_wt.contiguous().view(aligned_wt.shape[0], -1)
        else:
            T_var = param_info.optimal_transport_mat
            t_fc0_model = torch.matmul(T_var.t(), aligned_wt.contiguous().view(aligned_wt.shape[0], -1))

        geometric_fc = t_fc0_model.view(param_shape)

        aligned_dict[param_name] = geometric_fc

    model_a_aligned = get_network(model_name, num_class)
    model_a_aligned.to(device)
    model_a_aligned.eval()

    aligned_state_dict = model_a_aligned.state_dict()
    for k in aligned_state_dict.keys():
        if len(aligned_state_dict[k].shape) == 0:
            continue
        aligned_state_dict[k] = aligned_dict[k]
    model_a_aligned.load_state_dict(aligned_state_dict)
    model_a_aligned.eval()

    return model_a_aligned


def output_table(result_dir, model_b_weight_array, align_result_dict, vanilla_result_dict):
    print(f"\n結果を出力 {result_dir}")
    results_df = pd.DataFrame()
    results_df["wb"] = model_b_weight_array
    for k, v in align_result_dict.items():
        results_df[k] = v
    for k, v in vanilla_result_dict.items():
        results_df[k] = v
    save_filepath = join(result_dir, "results.csv")
    results_df.to_csv(save_filepath, index=None)


def output_figure(
    result_dir,
    model_b_weight_array,
    loss_list,
    loss_vanilla_list,
    acc_top1_list,
    acc_top1_vanilla_list,
):
    plt.plot(model_b_weight_array, loss_vanilla_list, label="Vanilla", marker="o", color="r")
    plt.plot(model_b_weight_array, loss_list, label="Align", marker="o", color="b")
    plt.ylabel("Loss")
    plt.xlabel(r"Weight towards model_B ($w_b$)")
    plt.legend()
    save_filepath = join(result_dir, "loss.png")
    plt.savefig(save_filepath)
    plt.clf()
    plt.close()

    plt.plot(
        model_b_weight_array,
        acc_top1_vanilla_list,
        label="Vanilla",
        marker="o",
        color="r",
    )
    plt.plot(model_b_weight_array, acc_top1_list, label="Align", marker="o", color="b")
    plt.ylabel("Test accuracy")
    plt.xlabel(r"Weight towards model_B ($w_b$)")
    plt.ylim(0, 1)
    plt.legend()
    save_filepath = join(result_dir, "accuracy.png")
    plt.savefig(save_filepath)
    plt.clf()
    plt.close()
