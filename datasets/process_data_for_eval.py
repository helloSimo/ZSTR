import os
import jsonlines
import argparse


def process_table(table):
    text = ' '.join(table['header'])
    for row in table['rows']:
        text += ' ' + ' '.join(row)
    return text


def process_corpus(dataset,):
    dest_f = jsonlines.open('corpus.jsonl', 'w')
    for table in jsonlines.open(os.path.join(dataset, 'tables.jsonl')):
        dest_f.write({
            'docid': table['id'],
            'title': table['title'],
            'text': process_table(table)
        })


def process_query(dataset):
    for split in ['train', 'dev', 'test']:
        dest_f = jsonlines.open(f'{split}.jsonl', 'w')
        for qa in jsonlines.open(os.path.join(dataset, f'{split}.jsonl')):
            dest_f.write({
                'query_id': qa['id'],
                'query': qa['question']
            })


def main(args):
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    process_corpus(args.dataset)
    process_query(args.dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)

    args = parser.parse_args()

    main(args)
