python src/train_model.py -net resnet18 -dataset cifar10 -seed 2
python src/train_model.py -net resnet18 -dataset cifar10 -seed 3
python src/train_model.py -net resnet18 -dataset cifar10 -seed 5
python src/train_model.py -net resnet18 -dataset cifar10 -seed 7
python src/train_model.py -net resnet18 -dataset cifar10 -seed 11
python src/train_model.py -net resnet18 -dataset cifar10 -seed 13

python src/train_model.py -net resnet18 -dataset cifar100 -seed 2
python src/train_model.py -net resnet18 -dataset cifar100 -seed 3
python src/train_model.py -net resnet18 -dataset cifar100 -seed 5
python src/train_model.py -net resnet18 -dataset cifar100 -seed 7
python src/train_model.py -net resnet18 -dataset cifar100 -seed 11
python src/train_model.py -net resnet18 -dataset cifar100 -seed 13

python src/train_model.py -net resnet50 -dataset cifar10 -seed 2
python src/train_model.py -net resnet50 -dataset cifar10 -seed 3
python src/train_model.py -net resnet50 -dataset cifar10 -seed 5
python src/train_model.py -net resnet50 -dataset cifar10 -seed 7
python src/train_model.py -net resnet50 -dataset cifar10 -seed 11
python src/train_model.py -net resnet50 -dataset cifar10 -seed 13

python src/train_model.py -net resnet50 -dataset cifar100 -seed 2
python src/train_model.py -net resnet50 -dataset cifar100 -seed 3
python src/train_model.py -net resnet50 -dataset cifar100 -seed 5
python src/train_model.py -net resnet50 -dataset cifar100 -seed 7
python src/train_model.py -net resnet50 -dataset cifar100 -seed 11
python src/train_model.py -net resnet50 -dataset cifar100 -seed 13

python src/train_model.py -net vgg11 -dataset cifar10 -seed 2
python src/train_model.py -net vgg11 -dataset cifar10 -seed 3
python src/train_model.py -net vgg11 -dataset cifar10 -seed 5
python src/train_model.py -net vgg11 -dataset cifar10 -seed 7
python src/train_model.py -net vgg11 -dataset cifar10 -seed 11
python src/train_model.py -net vgg11 -dataset cifar10 -seed 13

python src/train_model.py -net vgg11 -dataset cifar100 -seed 2
python src/train_model.py -net vgg11 -dataset cifar100 -seed 3
python src/train_model.py -net vgg11 -dataset cifar100 -seed 5
python src/train_model.py -net vgg11 -dataset cifar100 -seed 7
python src/train_model.py -net vgg11 -dataset cifar100 -seed 11
python src/train_model.py -net vgg11 -dataset cifar100 -seed 13


python src/train_model.py -net resnet18_nobn -dataset cifar10 -seed 2
python src/train_model.py -net resnet18_nobn -dataset cifar10 -seed 3
python src/train_model.py -net resnet18_nobn -dataset cifar10 -seed 5
python src/train_model.py -net resnet18_nobn -dataset cifar10 -seed 7
python src/train_model.py -net resnet18_nobn -dataset cifar10 -seed 11
python src/train_model.py -net resnet18_nobn -dataset cifar10 -seed 13

python src/train_model.py -net resnet18_nobn -dataset cifar100 -seed 2
python src/train_model.py -net resnet18_nobn -dataset cifar100 -seed 3
python src/train_model.py -net resnet18_nobn -dataset cifar100 -seed 5
python src/train_model.py -net resnet18_nobn -dataset cifar100 -seed 7
python src/train_model.py -net resnet18_nobn -dataset cifar100 -seed 11
python src/train_model.py -net resnet18_nobn -dataset cifar100 -seed 13

python src/train_model.py -net vgg11_nobn -dataset cifar10 -seed 2
python src/train_model.py -net vgg11_nobn -dataset cifar10 -seed 3
python src/train_model.py -net vgg11_nobn -dataset cifar10 -seed 5
# python src/train_model.py -net vgg11_nobn -dataset cifar10 -seed 7 学習がうまく進まなかった
python src/train_model.py -net vgg11_nobn -dataset cifar10 -seed 11
python src/train_model.py -net vgg11_nobn -dataset cifar10 -seed 13
python src/train_model.py -net vgg11_nobn -dataset cifar10 -seed 17

# python src/train_model.py -net vgg11_nobn -dataset cifar100 -seed 2 学習がうまく進まなかった
python src/train_model.py -net vgg11_nobn -dataset cifar100 -seed 3
# python src/train_model.py -net vgg11_nobn -dataset cifar100 -seed 5 学習がうまく進まなかった
python src/train_model.py -net vgg11_nobn -dataset cifar100 -seed 7
python src/train_model.py -net vgg11_nobn -dataset cifar100 -seed 11
python src/train_model.py -net vgg11_nobn -dataset cifar100 -seed 13
python src/train_model.py -net vgg11_nobn -dataset cifar100 -seed 17
# python src/train_model.py -net vgg11_nobn -dataset cifar100 -seed 19 学習がうまく進まなかった
python src/train_model.py -net vgg11_nobn -dataset cifar100 -seed 23



python src/train_model.py -net resnet50 -dataset imagenet -total_epoch 90 -b 16 -lr 0.00625 -seed 2
python src/train_model.py -net resnet50 -dataset imagenet -total_epoch 90 -b 16 -lr 0.00625 -seed 3

