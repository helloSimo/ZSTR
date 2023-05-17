import os
import re
from glob import glob
from argparse import ArgumentParser


def print_result(args):
    pattern = r'\d+'
    length_dict = {'128': 0, '256': 1}
    # epoch_dict = {'40': 0, '80': 1, '120': 0, '160': 1}
    # dataset_dict = {'WTQ': 0, 'WSQL': 1, 'NQT': 2, 'NQTt': 3}

    model_list = []
    for lr_name in args.lr_name.split(' '):
        for dir_name in glob('model_{}_{}*{}*'.format(args.model_name, args.prefix, lr_name)):
            if not os.path.exists(os.path.join(dir_name, 'result.txt')):
                continue
            if args.prefix == "" and len(dir_name.split('_')) == 7:
                continue
            print(dir_name)
            name_list = dir_name.split('_')
            dataset_index = name_list[-4].strip()
            length_index = length_dict[name_list[-3].strip()]
            epoch_index = int(name_list[-1].strip())

            with open(os.path.join(dir_name, 'result.txt')) as f:
                line = f.readline()
                while not line.startswith('best epoch'):
                    line = f.readline()

                epoch = re.findall(pattern, line)[0]
                line = f.readline()

                model_list.append((lr_name, epoch_index, dir_name, line.strip()))

    model_list = sorted(model_list)
    for i in range(0, len(model_list)):
        print(model_list[i][-2])
    print()
    for i in range(0, len(model_list)):
        print(','.join(model_list[i][-1].split('\t')))
    print()

    results = []
    for i in range(0, len(model_list)):
        rs = []
        for result in model_list[i][-1].split('\t'):
            rs.append(float(result))
        results.append(rs)

    print('best result:')
    for col in zip(*results):
        print(max(col), end='\t')

    rs = []
    for i, row in enumerate(results):
        rs.append((sum(row[1:4]), sum(row), i))
    max_id = max(rs)[2]

    print('\nbest epoch:')
    print(*(results[max_id]), sep='\t')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, default='condenser')
    parser.add_argument('--lr_name', type=str, default='1e5 5e6')
    parser.add_argument('--prefix', type=str, default='')
    main_args = parser.parse_args()

    print_result(main_args)
