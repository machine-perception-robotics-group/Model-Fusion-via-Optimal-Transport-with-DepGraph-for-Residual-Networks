# Model Fusion via Optimal Transport with DepGraph for Residual Networks 
"DepGraphを用いたグループ単位の整列とバッチ正規化層における統計量の再計算を実装したOT Fusion

モデルマージのチュートリアル資料：[Link](https://speakerdeck.com/rei0108/tiyutoriaru-moderumazi)

## 環境

| Library        | Version  |
|----------------|----------|
| Python         | 3.10.8   |
| PyTorch        | 1.13.1   |
| torchvision    | 0.14.1   |
| numpy          | 1.23.4   |
| pandas         | 2.2.3    |
| scikit-learn   | 1.6.1    |
| scipy          | 1.13.1   |
| matplotlib     | 3.10.0   |
| optuna         | 3.6.1    |



## データセット
* CIFAR-10
* CIFAR-100

---

## 1. セットアップ

### build_image.sh を実行して Docker イメージを作成
```bash
chmod +x build_image.sh
./build_image.sh
```
"improve_otfusion:v1" という名前の Docker イメージを作成．

### run_container.sh を実行して Docker コンテナの起動

```bash
chmod +x run_container.sh
./run_container.sh
```
ホストの /improve_otfusion は，コンテナ内の /workspace/improve_otfusion にマウント．

## 2.ベースモデルの作成

CIFAR-10, CIFAR-100 データセットで, seed を変えてモデルを学習.
コンテナ内の /workspace/improve_otfusion に移動し，下記コマンドを実行する．

```bash
chmod +x run_train_all.sh
./run_train_all.sh
```
モデルは /workspace/improve_otfusion/checkpoint に格納．
## 3.モデルフュージョンの実施

コンテナ内の /workspace/improve_otfusion/checkpoint に run_container.sh で作成したモデルを使用．

#### アクティベーションベースのモデルフュージョン

最適輸送の計算に各層からのアクティベーションを使用したモデルフュージョンを行うには，コンテナ内の /workspace/improve_otfusion で下記コマンドを実行．

```bash
python run_activation_based_fusion.py
```

実行結果は，コンテナ内の /workspace/improve_otfusion/results に格納．

#### 重みベースのモデルフュージョン

最適輸送の計算に各層の重み（パラメータ）を使用したモデルフュージョンを行うには，コンテナ内の /workspace/improve_otfusion で下記コマンドを実行．

```bash
python run_weight_based_fusion.py
```

実行結果は，コンテナ内の /workspace/improve_otfusion/results に格納．

---
## 実験結果
* モデル：ResNet18

| Dataset   | Model1  | Model2  | Activation-base Fusion  | Weights-base Fusion  |
|-----------|---------|---------|-------------------------|----------------------|
| CIFAR-10  | 94.99   | 94.76   | 92.16                   | 91.80                |
| CIFAR-100 | 75.88   | 75.72   | 63.11                   | 60.44                |

## 参考にしたリポジトリ

* Pytorch-cifar100 ([https://github.com/weiaicunzai/pytorch-cifar100](https://github.com/weiaicunzai/pytorch-cifar100))
  * CIFAR-10, CIFAR-100 でのモデルの学習で参考
* Model Fusion via Optimal Transport ([https://github.com/sidak/otfusion](https://github.com/sidak/otfusion))
  * 最適輸送行列の計算、及びそれを使用したモデルフュージョンで参考
  * バッチ正規化を使わないモデルの実装で参考
* Torch Pruning ([https://github.com/VainF/Torch-Pruning](https://github.com/VainF/Torch-Pruning))
  * 各層の依存関係を抽出するために使用
  * improve_otfusion/data/improve_otfusion/src/ext_pkg/Torch-Pruning に格納

コードを公開してくださった作者の方に深く感謝申しあげます．

---

