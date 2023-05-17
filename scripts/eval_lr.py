import os
from argparse import ArgumentParser
from glob import glob

from eval_model import eval_model, get_best_ckpt


def get_model_list(lr_name, prefix):
    model_list = []
    for dir_name in glob('model_dpr_{}*{}*'.format(prefix, lr_name)):
        if prefix == "" and len(dir_name.split('_')) == 7:
            continue
        # if os.path.exists(os.path.join(dir_name, 'setting.txt')) and \
        #         not os.path.exists(os.path.join(dir_name, 'result.txt')):
        #     model_list.append(dir_name)
        if not os.path.exists(os.path.join(dir_name, 'result.txt')):
            model_list.append(dir_name)

    return sorted(model_list)


def eval_main(args):
    if args.lr_name == '':
        lr_name = ['2e5', '1e5', '5e6', '1e6']
    else:
        lr_name = args.lr_name.split(' ')

    device = args.device
    prefix = args.prefix

    model_name_list = get_model_list(lr_name, prefix)
    for model_name in model_name_list:
        print(model_name)
    if input('Continue? (y/n)') != 'y':
        exit(0)

    for model_name in model_name_list:
        try:
            eval_model(model_name, device, args.dpr)
        except Exception as e:
            print(e)
            exit(-1)
        get_best_ckpt(model_name)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--lr_name', type=str, default='', choices=['', '2e5', '1e5', '5e6', '1e6'])
    parser.add_argument('--device', type=int)
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('--dpr', action='store_true')
    main_args = parser.parse_args()

    eval_main(main_args)
