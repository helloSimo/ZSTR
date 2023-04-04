from argparse import ArgumentParser

from train_model import train_model, name_model


DATASET_LIST = ['WTQ', 'WSQL', 'NQT', 'NQTt', 'TF']


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
    for lr_name in lr_name_list:
        for max_len in length_list:
            if args.prefix != '':
                prefix = '{}{}_'.format(args.prefix, max_len)
            else:
                prefix = ''
            for dataset in dataset_list:
                if dataset in ['WTQ', 'WSQL', 'TF']:
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
                             'epoch_num': epoch_num}
                    param_list.append(param)
                    print(name_model(**param))

    if input('Continue? (y/n)') != 'y':
        exit(0)

    for param in param_list:
        try:
            train_model(**param)
            # train_model(device=args.device,
            #             dataset=prefix + dataset,
            #             max_len=max_len,
            #             lr_name=lr_name,
            #             epoch_num=epoch_num)
        except Exception as e:
            print(e)
            exit(-1)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--lr_name', type=str)
    parser.add_argument('--device', type=int)
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('--length', type=int, choices=[0, 128, 256], default=0)
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--epoch_scale', type=float, default=1)
    main_args = parser.parse_args()

    train(main_args)
