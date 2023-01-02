import os
import shutil
import jsonlines
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer
import sys
sys.path.append('..')
from table_utils.table_processor import get_processor


def process_corpus(dataset, max_cell_length, delimiter, include_title, tokenizer_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    processor = get_processor(max_cell_length=max_cell_length,
                              max_input_length=512,
                              tokenizer=tokenizer,
                              include_title=include_title,
                              index_row=delimiter,
                              row_choose=True)

    dest_f = jsonlines.open(os.path.join('eval', 'corpus.jsonl'), 'w')
    print("processing corpus...")
    for table in tqdm(jsonlines.open(os.path.join(dataset, 'tables.jsonl'))):
        dest_f.write({
            'id': table['id'],
            'text': processor.process_table(table)
        })


def process_query(dataset):
    dest_f = jsonlines.open(os.path.join('eval', 'test.jsonl'), 'w')
    print("processing query...")
    for qa in tqdm(jsonlines.open(os.path.join(dataset, 'test.jsonl'))):
        dest_f.write({
            'id': qa['id'],
            'text': qa['question']
        })


def main(args):
    if not os.path.exists('eval'):
        os.mkdir('eval')

    with open(os.path.join('eval', 'setting.txt'), 'w') as f:
        print(args, file=f)

    process_corpus(args.dataset, args.max_cell_length, args.delimiter, args.title, args.tokenizer_name_or_path)
    process_query(args.dataset)
    shutil.copyfile(os.path.join(args.dataset, 'test.jsonl'), os.path.join('eval', 'label.jsonl'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='WTQ', choices=['WTQ', 'WikiSQL', 'NQTables', 'NQTablesFull'])
    parser.add_argument('--max_cell_length', type=int, default=8)
    parser.add_argument('--delimiter', action='store_true')
    parser.add_argument('--title', action='store_true')
    parser.add_argument('--tokenizer_name_or_path', type=str, default='bert-base-uncased')

    args = parser.parse_args()

    main(args)
