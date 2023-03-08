import os
from argparse import ArgumentParser
from glob import glob

from eval_model import eval_model, get_best_ckpt


def get_model_list(lr_name, prefix):
    model_list = []
    for dir_name in glob('model_dpr_{}*{}*'.format(prefix, lr_name)):
        if prefix == "" and len(dir_name.split('_')) == 7:
            continue
        if os.path.exists(os.path.join(dir_name, 'setting.txt')) and \
                not os.path.exists(os.path.join(dir_name, 'result.txt')):
            model_list.append(dir_name)

    return sorted(model_list)


def eval_main(args):
    lr_name = args.lr_name
    device = args.device
    prefix = args.prefix

    for model_name in get_model_list(lr_name, prefix):
        eval_model(model_name, device)
        get_best_ckpt(model_name)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--lr_name', type=str, choices=['2e5', '1e5', '5e6', '1e6'])
    parser.add_argument('--device', type=int, choices=[0, 1, 2, 3])
    parser.add_argument('--prefix', type=str, default='')
    main_args = parser.parse_args()

    eval_main(main_args)
