from argparse import ArgumentParser

from train_model import train_model


def train(args):
    if args.dataset != '':
        dataset_list = args.dataset.split(' ')
        for dataset in dataset_list:
            assert dataset in ['WTQ', 'WSQL', 'NQT', 'NQTt']
    else:
        dataset_list = ['WTQ', 'WSQL', 'NQT', 'NQTt']
    if args.length != 0:
        length_list = [args.length]
    else:
        length_list = [128, 256]
    epoch_list = [[40, 80], [120, 160]]
    lr_name_list = args.lr_name.split(' ')
    for lr in lr_name_list:
        assert lr in ['2e5', '1e5', '5e6', '1e6']

    for lr_name in lr_name_list:
        for max_len in length_list:
            if args.prefix != '':
                prefix = args.prefix.format(max_len)
            else:
                prefix = ''
            for dataset in dataset_list:
                if dataset in ['WTQ', 'WSQL']:
                    epoch_idx = 0
                else:
                    epoch_idx = 1
                for epoch_num in epoch_list[epoch_idx]:
                    epoch_num = int(epoch_num*args.epoch_scale)
                    try:
                        train_model(device=args.device,
                                    dataset=prefix + dataset,
                                    max_len=max_len,
                                    lr_name=lr_name,
                                    epoch_num=epoch_num)
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
