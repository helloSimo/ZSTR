import csv
import jsonlines
from argparse import ArgumentParser

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def get_qrel(label_path):
    qrel = {}
    for qa in jsonlines.open(label_path):
        qrel[qa['id']] = []
        for table_id in qa['table_id']:
            qrel[qa['id']].append(table_id)
    return qrel


def get_run(rank_path):
    run = {}
    with open(rank_path) as f:
        f = csv.reader(f, delimiter='\t')
        for line in f:
            run.setdefault(line[0], [])
            run[line[0]].append(line[1])
    return run


def get_top_k(qrel, run, top_k):
    acc = 0.0
    tot = 0.0
    for key in qrel.keys():
        tot += 1
        label_list = qrel[key]
        for table in run[key][:top_k]:
            if table in label_list:
                acc += 1
                break
    return acc/tot


def main():
    parser = ArgumentParser()
    parser.add_argument('--rank_path', required=True)
    parser.add_argument('--label_path', required=True)

    args = parser.parse_args()

    qrel = get_qrel(args.label_path)
    run = get_run(args.rank_path)

    top_k = [1, 10]
    results = []
    for k in top_k:
        results.append(get_top_k(qrel, run, k)*100)

    measure = ['R@{}'.format(k) for k in top_k]

    measure_str = ''
    for m in measure:
        measure_str += '{:12}'.format(m)
    logger.info(measure_str)
    results_str = ''
    for r in results:
        results_str += '{:<12.2f}'.format(r)
    logger.info(results_str)
    results_str = ''
    for r in results:
        results_str += '{:.2f}\t'.format(r)
    logger.info(results_str)


if __name__ == '__main__':
    main()
