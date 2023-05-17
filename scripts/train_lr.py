from argparse import ArgumentParser

from train_model import train_model, name_model
from eval_model import eval_model, get_best_ckpt
from print_result import print_result


DATASET_LIST = ['WTQ', 'WSQL', 'NQT', 'NQTt', 'TF', 'WQT', 'WQTt']


def train(args):
    dataset_list = args.dataset.split(' ')
    for dataset in dataset_list:
        assert dataset in DATASET_LIST

    if args.length != 0:
        length_list = [args.length]
    else:
        length_list = [128, 256]
    epoch_list = [[20, 40], [120, 160]]
    lr_name_list = args.lr_name.split(' ')
    for lr in lr_name_list:
        assert lr in ['2e5', '1e5', '5e6', '1e6']

    param_list = []
    model_name_list = []
    for lr_name in lr_name_list:
        for max_len in length_list:
            if args.prefix != '':
                prefix = '{}_'.format(args.prefix)
            else:
                prefix = ''
            for dataset in dataset_list:
                if dataset in ['WTQ', 'WSQL', 'TF', 'WQT', 'WQTt']:
                    epoch_idx = 0
                elif dataset in ['NQT', 'NQTt']:
                    epoch_idx = 1
                else:
                    raise ValueError('dataset {} not supported'.format(dataset))
                for epoch_num in epoch_list[epoch_idx]:
                    epoch_num = int(epoch_num * args.epoch_scale)
                    param = {'device': args.device,
                             'dataset': prefix + dataset,
                             'max_len': max_len,
                             'lr_name': lr_name,
                             'epoch_num': epoch_num,
                             'dpr': args.dpr,
                             'untie_encoder': args.untie_encoder,
                             'model': args.model,
                             'additional_model': args.additional_model,
                             'model_name': args.model_name,}
                    param_list.append(param)
                    model_name = name_model(dataset=param['dataset'],
                                            max_len=param['max_len'],
                                            lr_name=param['lr_name'],
                                            epoch_num=param['epoch_num'],
                                            model_name=param['model_name'])
                    print(model_name)
                    model_name_list.append(model_name)

    if input('Continue? (y/n)') != 'y':
        exit(0)

    for param in param_list:
        try:
            train_model(**param)
        except Exception as e:
            print(e)
            exit(-1)

    if args.do_eval:
        for model_name in model_name_list:
            try:
                eval_model(model_name, args.device, args.dpr)
            except Exception as e:
                print(e)
                exit(-1)
            get_best_ckpt(model_name)

        print_result(args)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--lr_name', type=str, default="1e5 5e6")
    parser.add_argument('--device', type=int)
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('--model_name', type=str, default='condenser')
    parser.add_argument('--model', type=str, default='Luyu/co-condenser-wiki')
    parser.add_argument('--additional_model', type=str, default='Luyu/co-condenser-wiki')
    parser.add_argument('--length', type=int, choices=[0, 128, 256], default=0)
    parser.add_argument('--dataset', type=str, default='NQT NQTt')
    parser.add_argument('--epoch_scale', type=float, default=1)
    parser.add_argument('--do_eval', type=bool, default=True)
    parser.add_argument('--dpr', action='store_true')
    parser.add_argument('--untie_encoder', action='store_true')
    main_args = parser.parse_args()

    train(main_args)
