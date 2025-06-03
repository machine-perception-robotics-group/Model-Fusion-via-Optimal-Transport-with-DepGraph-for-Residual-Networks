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


def get_proc_order(
    model_a,
    param_name__group_id_dict,
    param_name__prev_group_id_dict,
    group_id__layer_names_dict,
    group_id__param_names_dict,
    elim_last=True,
):
    proc_params_list = []

    model_a_params_dict = dict(model_a.named_parameters())
    # 入力層を探索
    in_param_name_list = []
    for k in param_name__group_id_dict.keys():
        if (
            (not k in param_name__prev_group_id_dict.keys())
            and (k in model_a_params_dict.keys())
            and (len(model_a_params_dict[k].shape) > 1)
        ):
            in_param_name_list.append(k)
    # パラメータ名末尾の .weight や .bias を削除
    in_layer_name_list = [".".join(name.split(".")[:-1]) for name in in_param_name_list]
    # 入力層がみつからない、または複数みつかればストップ
    if len(in_layer_name_list) != 1:
        print("multiple input layer? or can't find input layer.")
        print("stop process!!")
        sys.exit()

    # 入力層のグループID
    target_grp_id = param_name__group_id_dict[in_param_name_list[0]]

    # 最適輸送行列計算対象となったかを記録する辞書
    is_param_checked_dict = {k: False for k in model_a_params_dict.keys()}
    is_grp_checked_dict = {k: False for k in group_id__layer_names_dict.keys()}

    old_target_grp_id = None
    while old_target_grp_id != target_grp_id:
        print()
        old_target_grp_id = target_grp_id
        target_param_names = []
        for param_name in [
            name for name in group_id__param_names_dict[target_grp_id] if name in model_a_params_dict.keys()
        ]:
            # 右側に変換行列がかからないもの
            if not param_name in param_name__prev_group_id_dict.keys():
                target_param_names.append(param_name)
                continue
            # 右側に自グループの変換行列がかかるもの
            if param_name__prev_group_id_dict[param_name] == target_grp_id:
                target_param_names.append(param_name)
                continue
            # 右側にかかる変換行列が最適輸送行列計算対象となったグループのもの
            prev_id = param_name__prev_group_id_dict[param_name]
            if is_grp_checked_dict[prev_id]:
                target_param_names.append(param_name)
                continue
        proc_params_list.append(target_param_names)

        for k in is_param_checked_dict.keys():
            if k in target_param_names:
                is_param_checked_dict[k] = True
        is_grp_checked_dict[target_grp_id] = True

        # 入力層から順々に最適輸送行列計算対象としたい
        # ここでは、.named_parameters() の順番で処理が進むようにしている
        for k in model_a_params_dict.keys():
            if k in param_name__prev_group_id_dict.keys():
                # まだ最適輸送行列計算対象となっておらず、右側にかかる変換行列が最適輸送行列計算対象となったグループのものを次のターゲットにする
                prev_grp_id = param_name__prev_group_id_dict[k]
                if (not is_param_checked_dict[k]) and (is_grp_checked_dict[prev_grp_id]):
                    target_grp_id = param_name__group_id_dict[k]
                    break
    if elim_last:
        # 最も出力側のグループは、出力が各クラスの尤度となるため処理しない
        print(f"skip: {proc_params_list[-1]}")
        proc_params_list = proc_params_list[:-1]
    return proc_params_list


def normlize_vecs(vecs, eps=1e-9):
    norms = torch.norm(vecs, dim=-1, keepdim=True)
    return vecs / (norms + eps)


def calc_ot_mat(
    model_a,
    model_b,
    proc_params_list,
    group_info_dict,
    param_name__group_id_dict,
    param_name__prev_group_id_dict,
    device,
    max_elm_num=None,
    use_1d_param=False,
):

    model_a_params_dict = dict(model_a.named_parameters())
    model_b_params_dict = dict(model_b.named_parameters())

    right_tmat_dict = {}
    for param_name in model_a_params_dict.keys():
        right_tmat_dict[param_name] = None

    ground_metric_object = GroundMetric()

    for param_names in proc_params_list:
        print()
        print("*" * 20)
        print(param_names)
        grp_id = param_name__group_id_dict[param_names[0]]
        print("grp_id", grp_id)

        grp_a_param_dict = {param_name: model_a_params_dict[param_name] for param_name in param_names}
        grp_b_param_dict = {param_name: model_b_params_dict[param_name] for param_name in param_names}

        grp_shape_dict = {k: v.shape for k, v in grp_a_param_dict.items()}

        # 1次元のパラメータは右から変換行列を掛けてアライメントを合わせる必要がないため、処理のやり方が異なる
        # ここでは2次元以上のパラメータを格納
        grp_a_param_dict = {
            k: v.data.view(v.shape[0], v.shape[1], -1) for k, v in grp_a_param_dict.items() if len(v.shape) >= 2
        }
        grp_b_param_dict = {
            k: v.data.view(v.shape[0], v.shape[1], -1) for k, v in grp_b_param_dict.items() if len(v.shape) >= 2
        }

        aligned_param_dict = {}
        param_b_dict = {}
        if use_1d_param:
            # 1次元のパラメータはアライメントを合わせる必要がないので、そのまま辞書に追加
            aligned_param_dict = {
                param_name: model_a_params_dict[param_name].data
                for param_name in param_names
                if len(model_a_params_dict[param_name].shape) == 1
            }
            param_b_dict = {
                param_name: model_b_params_dict[param_name].data
                for param_name in param_names
                if len(model_b_params_dict[param_name].shape) == 1
            }

        # 右から変換行列を掛けてアライメントを合わせる
        for param_name in grp_a_param_dict.keys():
            param_a = grp_a_param_dict[param_name]
            param_b = grp_b_param_dict[param_name]
            layer_shape = grp_shape_dict[param_name]

            if right_tmat_dict[param_name] == None:
                aligned_param = param_a
            else:
                right_T_var = right_tmat_dict[param_name]
                if len(layer_shape) > 2:
                    right_T_var = right_T_var.unsqueeze(0).repeat(param_a.shape[2], 1, 1)
                    aligned_param = torch.bmm(param_a.permute(2, 0, 1), right_T_var).permute(1, 2, 0)
                else:
                    param_a = param_a.reshape(layer_shape)
                    param_b = param_b.reshape(layer_shape)
                    aligned_param = torch.matmul(param_a, right_T_var)

            aligned_param_dict[param_name] = aligned_param
            param_b_dict[param_name] = param_b

        # 分布
        mu_cardinality = aligned_param_dict[param_names[0]].shape[0]
        nu_cardinality = param_b_dict[param_names[0]].shape[0]
        mu = np.ones(mu_cardinality) / mu_cardinality
        nu = np.ones(nu_cardinality) / nu_cardinality

        # コスト行列計算に使うパラメータ
        coordinates_a = []
        coordinates_b = []
        # 各パラメータを正規化し結合
        for param_name in grp_a_param_dict.keys():
            coordinates = aligned_param_dict[param_name].contiguous().view(mu_cardinality, -1)
            coordinates = normlize_vecs(coordinates)
            coordinates_a.append(coordinates)

            coordinates = param_b_dict[param_name].contiguous().view(nu_cardinality, -1)
            coordinates = normlize_vecs(coordinates)
            coordinates_b.append(coordinates)

        coordinates_a = torch.concat(coordinates_a, dim=-1)
        coordinates_b = torch.concat(coordinates_b, dim=-1)

        coordinates_a = coordinates_a.data.cpu()
        coordinates_b = coordinates_b.data.cpu()
        print(f"elem_num: {coordinates_a.shape}, {coordinates_b.shape}")
        """
        # コスト計算時の処理を見直すことで必要なくなった
        elm_num_a = coordinates_a.shape[0] * coordinates_a.shape[1]
        elm_num_b = coordinates_b.shape[0] * coordinates_b.shape[1]
        # メモリ不足回避のため、要素数が max_elm_num を超えないようにする
        if (elm_num_a > max_elm_num) or (elm_num_a > max_elm_num):
            elm_num = max(elm_num_a, elm_num_b)

            elm_a_1 = int(coordinates_a.shape[1] / (elm_num / max_elm_num))
            coordinates_a = coordinates_a[:, :elm_a_1]

            elm_b_1 = int(coordinates_b.shape[1] / (elm_num / max_elm_num))
            coordinates_b = coordinates_b[:, :elm_b_1]

            print(f"trimmed elem_num: {coordinates_a.shape}, {coordinates_b.shape}")
        """

        # コスト行列
        # ResNet-50 だと時間がかかるので GPU を使えるようにした
        M0 = ground_metric_object.process(coordinates_a, coordinates_b, device)
        M0 = M0.numpy()
        print(f"cost matrix shape: {M0.shape}")

        # 最適輸送行列
        T = ot.emd(mu, nu, M0, numItermax=10000000)  # numItermaxをデフォルトの100倍にした
        T_var = torch.from_numpy(T).to(device).float()

        eps = 1e-7
        marginals = torch.ones(T_var.shape).to(device)
        marginals = torch.matmul(T_var, marginals)
        marginals = 1 / (marginals + eps)
        T_var = T_var * marginals

        # 計算結果格納
        group_info_dict[grp_id].optimal_transport_mat = T_var

        for k in right_tmat_dict.keys():
            if (k in param_name__prev_group_id_dict.keys()) and (param_name__prev_group_id_dict[k] == grp_id):
                right_tmat_dict[k] = group_info_dict[param_name__prev_group_id_dict[k]].optimal_transport_mat


def main_weight_based(
    model_name,  # vgg11|resnet18|resnet50|vgg11_nobn|resnet18_nobn|resnet50_nobn
    dataset_name,  # cifar10|cifar100|imagenet
    model_a_filepath,  # file path of checkpoint file for model A
    model_b_filepath,  # file path of checkpoint file for model B
    result_dir,  # output directory
    device=None,  # device that model runs
    recalc_batchnorm_stats=True,  # recalculate batchnorm layer statistics or not
    recalc_bs=256,  # batch size for recalculate batchnorm layer statistics
    numpy_seed=100,  # seed of numpy
    divide_num=11,  # number of divisions model A and model B
    use_1d_param=True,  # use 1d parameters for optimal transport matrix calculation or not
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
    print(f"numpy_seed = {numpy_seed}")
    print(f"divide_num = {divide_num}")
    print()

    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    # データセット取得
    get_test_dataloader, get_train_dataloader, num_class, example_inputs = get_datasets(dataset_name)
    test_loader = get_test_dataloader()  # 評価用
    train_loader_bn = get_train_dataloader(batch_size=recalc_bs)  # バッチ正規化層の統計量再計算用

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

    # DNNの各層の情報を取得
    (
        group_id__layer_names_dict,
        group_id__param_names_dict,
        param_name__group_id_dict,
        param_name__prev_group_id_dict,
    ) = get_group_info(model_a, example_inputs, skip_last=False)

    group_info_dict = {}
    for k, v in group_id__param_names_dict.items():
        group_info_dict[k] = FusionGroupInfo(group_id=k, param_names=v, optimal_transport_mat=None)

    # パラメータの処理順
    proc_params_list = get_proc_order(
        model_a,
        param_name__group_id_dict,
        param_name__prev_group_id_dict,
        group_id__layer_names_dict,
        group_id__param_names_dict,
    )

    # グループ毎に最適輸送行列を計算
    calc_ot_mat(
        model_a,
        model_b,
        proc_params_list,
        group_info_dict,
        param_name__group_id_dict,
        param_name__prev_group_id_dict,
        device,
        use_1d_param,
    )

    # 変換パラメータ整理
    param_info_dict = organize_info(model_a, param_name__group_id_dict, group_info_dict, param_name__prev_group_id_dict)

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
