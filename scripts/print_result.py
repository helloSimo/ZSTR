import os
import re
from glob import glob
from argparse import ArgumentParser


def main(args):
    pattern = r'\d+'
    length_dict = {'128': 0, '256': 1}
    epoch_dict = {'40': 0, '80': 1, '120': 0, '160': 1}
    dataset_dict = {'WTQ': 0, 'WSQL': 1, 'NQT': 2, 'NQTt': 3}

    model_list = []
    for dir_name in glob('model_dpr_{}*{}*'.format(args.prefix, args.lr_name)):
        if args.prefix == "" and len(dir_name.split('_')) == 7:
            continue
        name_list = dir_name.split('_')
        dataset_index = dataset_dict[name_list[-4].strip()]
        length_index = length_dict[name_list[-3].strip()]
        epoch_index = epoch_dict[name_list[-1].strip()]

        with open(os.path.join(dir_name, 'result.txt')) as f:
            line = f.readline()
            while not line.startswith('best epoch'):
                line = f.readline()

            epoch = re.findall(pattern, line)[0]
            line = f.readline()

            model_list.append((epoch_index, length_index, dir_name, epoch.strip(), line.strip()))

    model_list = sorted(model_list)
    for i in range(0, len(model_list), args.num_per_row):
        for j in range(args.num_per_row):
            print(model_list[i + j][-3], end='  ')
        print()
        # print('/'.join([model_list[i][-2], model_list[i + 1][-2], model_list[i + 2][-2], model_list[i + 3][-2]]))
        for j in range(args.num_per_row):
            print(model_list[i + j][-1], end='\t')
        print('\n')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--lr_name', type=str)
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('--num_per_row', type=int, default=4)
    main_args = parser.parse_args()

    main(main_args)
