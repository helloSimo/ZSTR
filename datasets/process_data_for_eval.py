import os
import jsonlines
import argparse
from transformers import AutoTokenizer
import sys
sys.path.append('..')
from table_utils.table_processor import get_processor


def process_corpus(dataset, max_cell_length, delimiter, tokenizer_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    processer = get_processor(max_cell_length=max_cell_length,
                              max_input_length=512,
                              tokenizer=tokenizer,
                              include_title=False,
                              index_row=delimiter,
                              row_choose=True)

    dest_f = jsonlines.open(os.path.join('eval', 'corpus.jsonl'), 'w')
    for table in jsonlines.open(os.path.join(dataset, 'tables.jsonl')):
        dest_f.write({
            'docid': table['id'],
            'title': table['title'],
            'text': processer.process_table(table)
        })


def process_query(dataset):
    for split in ['train', 'dev', 'test']:
        dest_f = jsonlines.open(os.path.join('eval', f'{split}.jsonl'), 'w')
        for qa in jsonlines.open(os.path.join(dataset, f'{split}.jsonl')):
            dest_f.write({
                'query_id': qa['id'],
                'query': qa['question']
            })


def main(args):
    if not os.path.exists('eval'):
        os.mkdir('eval')

    process_corpus(args.dataset, args.max_cell_length, args.delimiter, args.tokenizer_name_or_path)
    process_query(args.dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='WTQ', choices=['WTQ', 'WikiSQL', 'NQTables'])
    parser.add_argument('--max_cell_length', type=int, default=8)
    parser.add_argument('--delimiter', action='store_true')
    parser.add_argument('--tokenizer_name_or_path', type=str, default='bert-base-uncased')

    args = parser.parse_args()

    main(args)
