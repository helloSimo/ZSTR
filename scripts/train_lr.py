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

    for max_len in length_list:
        for dataset in dataset_list:
            if dataset in ['WTQ', 'WSQL']:
                epoch_idx = 0
            else:
                epoch_idx = 1
            for epoch_num in epoch_list[epoch_idx]:
                try:
                    train_model(device=args.device,
                                dataset=args.prefix + dataset,
                                max_len=max_len,
                                lr_name=args.lr_name,
                                epoch_num=epoch_num)
                except Exception as e:
                    print(e)
                    exit(-1)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--lr_name', type=str, choices=['2e5', '1e5', '5e6', '1e6'])
    parser.add_argument('--device', type=int, choices=[0, 1, 2, 3])
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('--length', type=int, choices=[0, 128, 256], default=0)
    parser.add_argument('--dataset', type=str, default='')
    main_args = parser.parse_args()

    train(main_args)
