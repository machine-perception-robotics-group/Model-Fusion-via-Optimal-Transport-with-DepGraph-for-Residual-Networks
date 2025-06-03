from glob import glob
from os.path import join

from src.fusion.fusion_wts import main_weight_based as main_fusion


def run_fusion(result_root_dir, checkpoint_dir, dataset_name, model_name, epoch, use_bn=True):
    # 作成したモデルを2つに分け、フュージョンの実験を行う
    model_dir = join(checkpoint_dir, dataset_name, model_name)
    model_filepath_list = glob(join(model_dir, "*", f"{model_name}-{epoch}-last.pth"))
    if len(model_filepath_list) < 2:
        print("*" * 80)
        print(f"学習済みモデルが2個未満です\n{model_dir}")
        print("*" * 80)
    half_num = len(model_filepath_list) // 2
    model_filepath_list_a = model_filepath_list[:half_num]
    model_filepath_list_b = model_filepath_list[half_num:]
    cnt = 0
    for model_a_filepath in model_filepath_list_a:
        for model_b_filepath in model_filepath_list_b:
            # バッチ正規化層の統計量再計算あり
            if use_bn:
                result_dir = join(result_root_dir, dataset_name, model_name, str(cnt))
                main_fusion(
                    model_name,
                    dataset_name,
                    model_a_filepath,
                    model_b_filepath,
                    result_dir,
                    recalc_batchnorm_stats=True,
                )

            # バッチ正規化層の統計量再計算なし
            result_dir = join(
                result_root_dir,
                dataset_name,
                model_name + "_wo_recalc_batchnorm_stats",
                str(cnt),
            )
            main_fusion(
                model_name,
                dataset_name,
                model_a_filepath,
                model_b_filepath,
                result_dir,
                recalc_batchnorm_stats=False,
            )

            cnt += 1


result_root_dir = "results/weight_based"

checkpoint_dir = "/workspace/improve_otfusion/checkpoint"

# CIFAR10, ResNet-18 バッチ正規化あり
epoch = "200"
dataset_name = "cifar10"
model_name = "resnet18"
run_fusion(result_root_dir, checkpoint_dir, dataset_name, model_name, epoch)

# CIFAR10, ResNet-18 バッチ正規化なし
epoch = "200"
dataset_name = "cifar10"
model_name = "resnet18_nobn"
run_fusion(result_root_dir, checkpoint_dir, dataset_name, model_name, epoch, use_bn=False)

# CIFAR10, VGG-11 バッチ正規化あり
epoch = "200"
dataset_name = "cifar10"
model_name = "vgg11"
run_fusion(result_root_dir, checkpoint_dir, dataset_name, model_name, epoch)

# CIFAR10, VGG-11 バッチ正規化なし
epoch = "200"
dataset_name = "cifar10"
model_name = "vgg11_nobn"
run_fusion(result_root_dir, checkpoint_dir, dataset_name, model_name, epoch, use_bn=False)

# CIFAR10, ResNet-50 バッチ正規化あり
epoch = "200"
dataset_name = "cifar10"
model_name = "resnet50"
run_fusion(result_root_dir, checkpoint_dir, dataset_name, model_name, epoch)


# CIFAR100, ResNet-18 バッチ正規化あり
epoch = "200"
dataset_name = "cifar100"
model_name = "resnet18"
run_fusion(result_root_dir, checkpoint_dir, dataset_name, model_name, epoch)

# CIFAR100, ResNet-18 バッチ正規化なし
epoch = "200"
dataset_name = "cifar100"
model_name = "resnet18_nobn"
run_fusion(result_root_dir, checkpoint_dir, dataset_name, model_name, epoch, use_bn=False)

# CIFAR100, VGG-11 バッチ正規化あり
epoch = "200"
dataset_name = "cifar100"
model_name = "vgg11"
run_fusion(result_root_dir, checkpoint_dir, dataset_name, model_name, epoch)

# CIFAR100, VGG-11 バッチ正規化なし
epoch = "200"
dataset_name = "cifar100"
model_name = "vgg11_nobn"
run_fusion(result_root_dir, checkpoint_dir, dataset_name, model_name, epoch, use_bn=False)

# CIFAR100, ResNet-50 バッチ正規化あり
epoch = "200"
dataset_name = "cifar100"
model_name = "resnet50"
run_fusion(result_root_dir, checkpoint_dir, dataset_name, model_name, epoch)


# OTFusion公式リポジトリで公開されているモデル
otfusion_checkpoint_dir = "/workspace/improve_otfusion/otfusion_files/checkpoint"

# CIFAR10, ResNet-18 バッチ正規化なし
model_name = "resnet18_nobn_nobias"
dataset_name = "cifar10"
model_a_filepath = join(otfusion_checkpoint_dir, "resnet18_otfusion/model_0/best.checkpoint")
model_b_filepath = join(otfusion_checkpoint_dir, "resnet18_otfusion/model_1/best.checkpoint")
result_dir = join(result_root_dir, dataset_name, "otfusion_model", model_name)
main_fusion(
    model_name,
    dataset_name,
    model_a_filepath,
    model_b_filepath,
    result_dir,
    recalc_batchnorm_stats=False,
)

# CIFAR10, VGG-11 バッチ正規化なし
model_name = "vgg11_nobn_nobias"
dataset_name = "cifar10"
model_a_filepath = join(otfusion_checkpoint_dir, "vgg11_otfusion/model_0/best.checkpoint")
model_b_filepath = join(otfusion_checkpoint_dir, "vgg11_otfusion/model_1/best.checkpoint")
result_dir = join(result_root_dir, dataset_name, "otfusion_model", model_name)
main_fusion(
    model_name,
    dataset_name,
    model_a_filepath,
    model_b_filepath,
    result_dir,
    recalc_batchnorm_stats=False,
)
