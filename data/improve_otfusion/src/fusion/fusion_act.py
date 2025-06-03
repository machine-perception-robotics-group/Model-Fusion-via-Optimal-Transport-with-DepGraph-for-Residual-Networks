import os
import sys
from os.path import dirname, join

import numpy as np
import ot
import torch

from ..test_model import test
from .utils import (
    FusionGroupInfo,
    GroundMetric,
    fusion_interpolation,
    get_aligned_model,
    get_datasets,
    get_group_info,
    get_model,
    organize_info,
    output_figure,
    output_table,
)

# DepGraphのパッケージ読み込み
ext_pkg_dir = dirname(dirname(__file__))
sys.path.append(join(ext_pkg_dir, "ext_pkg", "Torch-Pruning"))
import torch_pruning as tp


def compute_activation(model, train_loader, num_samples, device, activation_seed=21, to_cpu=True):

    torch.manual_seed(activation_seed)

    def get_activation(activation, name):
        def hook(model, input, output):
            if name not in activation:
                activation[name] = []

            activation[name].append(output.detach())

        return hook

    # パラメータ名末尾の .weight や .bias を削除
    layer_names = [".".join(k.split(".")[:-1]) for k, v in model.named_parameters()]
    # 重複を削除
    layer_names = list(set(layer_names))

    # Initialize the activation dictionary
    activation = {}
    forward_hooks = []
    # Set forward hooks for all layers inside a model
    for name, layer in model.named_modules():
        if name == "":
            continue
        elif name not in layer_names:
            continue
        forward_hooks.append(layer.register_forward_hook(get_activation(activation, name)))

    model.eval()

    # Run the same data samples ('num_samples' many) across all the models
    num_samples_processed = 0
    for data, target in train_loader:
        if num_samples_processed == num_samples:
            break
        data = data.to(device)
        model(data)

        num_samples_processed += 1

    for layer in activation:
        activation[layer] = torch.stack(activation[layer])

    # Remove the hooks (as this was intefering with prediction ensembling)
    for hook in forward_hooks:
        hook.remove()

    if to_cpu:
        activation = {group_name: activation[group_name].cpu() for group_name in activation.keys()}

    return activation


def reorder_activation(activation):
    reordered_activation = {}

    for k, v in activation.items():
        v_squeezed = v.squeeze(1)
        reorder_dim = [l for l in range(1, len(v_squeezed.shape))] + [0]
        v_reordered = v_squeezed.permute(*reorder_dim).contiguous()
        reordered_activation[k] = v_reordered

    return reordered_activation


def calc_ot_mat(activation_a, activation_b, group_id__layer_names_dict, device):
    ground_metric_object = GroundMetric()
    group_info_dict = {}
    for group_id, layer_names in group_id__layer_names_dict.items():
        print()
        print("*" * 20)
        print(f"group_id {group_id}, layer_names {layer_names}")
        layer_a_acts = []
        layer_b_acts = []
        for layer_name in layer_names:
            layer_a_acts.append(activation_a[layer_name])
            layer_b_acts.append(activation_b[layer_name])

        layer_a_acts = torch.concat(layer_a_acts, dim=-1)
        layer_b_acts = torch.concat(layer_b_acts, dim=-1)

        mu_cardinality = layer_a_acts.shape[0]
        nu_cardinality = layer_b_acts.shape[0]

        mu = np.ones(mu_cardinality) / mu_cardinality
        nu = np.ones(nu_cardinality) / nu_cardinality

        coordinates_a = layer_a_acts.view(mu_cardinality, -1).data.cpu()
        coordinates_b = layer_b_acts.view(nu_cardinality, -1).data.cpu()
        print(f"elem_num: {coordinates_a.shape}, {coordinates_b.shape}")

        # ResNet-50 だと時間がかかるので GPU を使えるようにした
        M0 = ground_metric_object.process(coordinates_a, coordinates_b, device)
        M0 = M0.numpy()
        print(f"cost matrix shape: {M0.shape}")

        T = ot.emd(mu, nu, M0)
        T_var = torch.from_numpy(T).to(device).float()

        eps = 1e-7
        marginals = torch.ones(T_var.shape).to(device)
        marginals = torch.matmul(T_var, marginals)
        marginals = 1 / (marginals + eps)
        T_var = T_var * marginals

        fusion_group_info = FusionGroupInfo(group_id=group_id, layer_names=layer_names, optimal_transport_mat=T_var)
        group_info_dict[group_id] = fusion_group_info
    return group_info_dict


def main_activation_based(
    model_name,  # vgg11|resnet18|resnet50|vgg11_nobn|resnet18_nobn|resnet50_nobn
    dataset_name,  # cifar10|cifar100|imagenet
    model_a_filepath,  # file path of checkpoint file for model A
    model_b_filepath,  # file path of checkpoint file for model B
    result_dir,  # output directory
    device=None,  # device that model runs
    recalc_batchnorm_stats=True,  # recalculate batchnorm layer statistics or not
    recalc_bs=256,  # batch size for recalculate batchnorm layer statistics
    act_num_samples=200,  # number of samples for calculate activation
    activation_seed=21,  # seed for calculate activation
    numpy_seed=100,  # seed of numpy
    divide_num=11,  # number of divisions model A and model B
):
    print("*" * 80)
    print()
    print("\n開始")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nmodel_name = {model_name}")
    print(f"dataset_name = {dataset_name}")
    print(f"model_a_filepath = {model_a_filepath}")
    print(f"model_b_filepath = {model_b_filepath}")
    print(f"result_dir = {result_dir}")
    print(f"device = {device}")
    print(f"recalc_batchnorm_stats = {recalc_batchnorm_stats}")
    print(f"recalc_bs = {recalc_bs}")
    print(f"act_num_samples = {act_num_samples}")
    print(f"activation_seed = {activation_seed}")
    print(f"numpy_seed = {numpy_seed}")
    print(f"divide_num = {divide_num}")
    print()

    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    # データセット取得
    print("\nデータセット取得")
    get_test_dataloader, get_train_dataloader, num_class, example_inputs = get_datasets(dataset_name)
    test_loader = get_test_dataloader()  # 評価用
    train_loader_act = get_train_dataloader(batch_size=1)  # アクティベーション計算用
    train_loader_bn = get_train_dataloader(batch_size=recalc_bs)  # バッチ正規化層の統計量再計算用
    print("完了")

    # Fusionするモデル取得
    is_otfusion_model = "otfusion" in result_dir
    model_a, acc_top1_a = get_model(model_name, num_class, device, model_a_filepath, is_otfusion_model, test_loader)
    model_b, acc_top1_b = get_model(model_name, num_class, device, model_b_filepath, is_otfusion_model, test_loader)

    # vanilla averaging
    print("\nvanilla averaging")
    (
        acc_top1_vanilla_list,
        acc_top5_vanilla_list,
        loss_vanilla_list,
        model_b_weight_array,
    ) = fusion_interpolation(
        model_a,
        model_b,
        model_name,
        num_class,
        device,
        divide_num,
        test_loader,
        recalc_batchnorm_stats,
        train_loader_bn,
    )

    # アクティベーション計算
    print("\nアクティベーション計算")
    np.random.seed(numpy_seed)
    activation_a = compute_activation(model_a, train_loader_act, act_num_samples, device, activation_seed)
    activation_b = compute_activation(model_b, train_loader_act, act_num_samples, device, activation_seed)
    activation_a = reorder_activation(activation_a)
    activation_b = reorder_activation(activation_b)

    # DNNの各層の情報を取得
    (
        group_id__layer_names_dict,
        group_id__param_names_dict,
        param_name__group_id_dict,
        param_name__prev_group_id_dict,
    ) = get_group_info(model_a, example_inputs)

    # グループ毎に最適輸送行列を計算
    print("\n最適輸送行列の計算")
    group_info_dict = calc_ot_mat(activation_a, activation_b, group_id__layer_names_dict, device)

    # 変換パラメータ整理
    param_info_dict = organize_info(
        model_a,
        param_name__group_id_dict,
        group_info_dict,
        param_name__prev_group_id_dict,
    )

    # model_a のアライメントを model_b に合わせる
    model_a_aligned = get_aligned_model(model_name, num_class, param_info_dict, device)
    acc_top1_aligned, acc_top5_aligned, loss_aligned = test(model_a_aligned, test_loader, device=device)
    print("\nacc_aligned", acc_top1_aligned)

    # フュージョン結果計算
    print("\nアライメント合わせ実施後")
    acc_top1_list, acc_top5_list, loss_list, model_b_weight_array = fusion_interpolation(
        model_a_aligned,
        model_b,
        model_name,
        num_class,
        device,
        divide_num,
        test_loader,
        recalc_batchnorm_stats,
        train_loader_bn,
    )

    # フュージョン結果出力
    align_result_dict = {}
    align_result_dict["align_acc_top1"] = acc_top1_list
    align_result_dict["align_acc_top5"] = acc_top5_list
    align_result_dict["align_loss"] = loss_list

    vanilla_result_dict = {}
    vanilla_result_dict["vanilla_acc_top1"] = acc_top1_vanilla_list
    vanilla_result_dict["vanilla_acc_top5"] = acc_top5_vanilla_list
    vanilla_result_dict["vanilla_loss"] = loss_vanilla_list

    output_table(result_dir, model_b_weight_array, align_result_dict, vanilla_result_dict)
    output_figure(
        result_dir,
        model_b_weight_array,
        loss_list,
        loss_vanilla_list,
        acc_top1_list,
        acc_top1_vanilla_list,
    )
