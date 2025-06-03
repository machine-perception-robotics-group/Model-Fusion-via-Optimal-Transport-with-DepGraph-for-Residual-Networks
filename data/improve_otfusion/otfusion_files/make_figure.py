from os.path import join, dirname

import matplotlib.pyplot as plt
import pandas as pd


def make_figure(result_file):
    df = pd.read_csv(result_file)

    plt.plot(df.wb, df.geometric_acc, label='OTFusion', marker="o")
    plt.plot(df.wb, df.naive_acc, label='Vanilla', marker="o")
    plt.ylabel("Test accuracy")
    plt.xlabel(r"Weight towards model_B ($w_b$)")
    plt.ylim(0, 100)
    plt.legend()

    save_dir = dirname(result_file)
    save_fn = 'acc.png'
    save_filepath = join(save_dir, save_fn)
    print(f'{save_filepath}にファイルを保存します')

    plt.savefig(save_filepath)
    plt.clf()
    plt.close()

    print('完了')


result_file = 'otfusion_orig_results/cifar10_vgg11/results.csv'
make_figure(result_file)

result_file = 'otfusion_orig_results/cifar10_resnet18/results.csv'
make_figure(result_file)
