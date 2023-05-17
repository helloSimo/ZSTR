import os
from argparse import ArgumentParser
import jsonlines
import random


def get_data(split, src, part):
    qas = []
    for qa in jsonlines.open(os.path.join(src, '{}.jsonl'.format(split))):
        qas.append(qa)

    return random.sample(qas, round(len(qas)*part))


def process(src1, part1, src2, part2, tar):
    if not os.path.exists(tar):
        os.mkdir(tar)

    for split in ['train', 'dev']:
        tar_f = jsonlines.open(os.path.join(tar, '{}.jsonl'.format(split)), 'w')
        tar_f.write_all(get_data(split, src1, part1))
        tar_f.write_all(get_data(split, src2, part2))
        tar_f.close()

    tar_f = open(os.path.join(tar, 'setting.txt'), 'w')
    with open(os.path.join(src1, 'setting.txt')) as f:
        tar_f.write(f.read())
    with open(os.path.join(src2, 'setting.txt')) as f:
        tar_f.write(f.read())
    tar_f.close()


def main(args):
    random.seed(42)

    # src_formats = []
    # for length in [128, 256]:
    #     for dataset in ['NQT', 'NQTt']:
    #         src_formats.append('datasets/{}%d_%s_train' % (length, dataset))
    src_formats = ['datasets/{}128_NQT_train']

    for src_format in src_formats:
        process(src1=src_format.format(args.src1),
                part1=args.part1,
                src2=src_format.format(args.src2),
                part2=args.part2,
                tar=src_format.format(args.tar))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--src1', type=str)
    parser.add_argument('--src2', type=str)
    parser.add_argument('--part1', type=float, default=1)
    parser.add_argument('--part2', type=float, default=1)
    parser.add_argument('--tar', type=str)
    main_args = parser.parse_args()

    main(main_args)
