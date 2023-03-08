import os
import jsonlines
import argparse


def main(args):
    tot = 0
    corpus = set()
    for qa in jsonlines.open(os.path.join('../datasets/', args.dataset, 'train.jsonl')):
        tot += 1
        for table_id in qa['table_id']:
            corpus.add(table_id)
    print("{}/{} {:.2f}".format(len(corpus), tot, tot/len(corpus)))

    tot = 0
    corpus = set()
    for qa in jsonlines.open(os.path.join('../datasets/', args.dataset, 'dev.jsonl')):
        tot += 1
        for table_id in qa['table_id']:
            corpus.add(table_id)
    print("{}/{} {:.2f}".format(len(corpus), tot, tot/len(corpus)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='WTQ', choices=['WTQ', 'WikiSQL', 'NQTables'])

    args = parser.parse_args()

    main(args)
