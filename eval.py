import csv
import jsonlines
import pytrec_eval
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
        qrel[qa['id']] = {}
        for table_id in qa['table_id']:
            qrel[qa['id']][table_id] = 1
    return qrel


def get_run(rank_path):
    run = {}
    with open(rank_path) as f:
        f = csv.reader(f, delimiter='\t')
        for line in f:
            run.setdefault(line[0], {})
            run[line[0]][line[1]] = float(line[2])
    return run


def main():
    parser = ArgumentParser()
    parser.add_argument('--rank_path', required=True)
    parser.add_argument('--label_path', required=True)

    args = parser.parse_args()

    qrel = get_qrel(args.label_path)
    run = get_run(args.rank_path)

    top_k = [1, 10, 50]
    measure = ['map']
    measure.extend([f'recall_{k}' for k in top_k])
    measure.append('ndcg_cut_10')

    evaluator = pytrec_eval.RelevanceEvaluator(qrel, measure)

    tot = 0
    results = [0.0] * len(measure)
    for value in evaluator.evaluate(run).values():
        tot += 1
        for i, m in enumerate(measure):
            results[i] += value[m]

    measure_str = ''
    for m in measure:
        measure_str += '{:12}'.format(m)
    logger.info(measure_str)
    results_str = ''
    for r in results:
        results_str += '{:<12.2f}'.format(r / tot * 100)
    logger.info(results_str)
    results_str = ''
    for r in results:
        results_str += '{:.2f}\t'.format(r / tot * 100)
    logger.info(results_str)


if __name__ == '__main__':
    main()
