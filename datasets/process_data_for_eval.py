import os
import shutil
import jsonlines
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer
import sys
sys.path.append('..')
from table_utils.table_processor import get_processor


def process_corpus(dataset, max_cell_length, delimiter, include_title, tokenizer_name_or_path, output_dir):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    processor = get_processor(max_cell_length=max_cell_length,
                              max_input_length=512,
                              tokenizer=tokenizer,
                              include_title=include_title,
                              index_row=delimiter,
                              row_choose=True)

    dest_f = jsonlines.open(os.path.join(output_dir, 'corpus.jsonl'), 'w')
    print("processing corpus...")
    for table in tqdm(jsonlines.open(os.path.join(dataset, 'tables.jsonl'))):
        dest_f.write({
            'id': table['id'],
            'text': processor.process_table(table)[0]
        })


def process_query(dataset, output_dir):
    dest_f = jsonlines.open(os.path.join(output_dir, 'test.jsonl'), 'w')
    print("processing query...")
    for qa in tqdm(jsonlines.open(os.path.join(dataset, 'test.jsonl'))):
        dest_f.write({
            'id': qa['id'],
            'text': qa['question']
        })


def main(args):
    if args.output_dir is None:
        args.output_dir = 'train'
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    with open(os.path.join(args.output_dir, 'setting.txt'), 'w') as f:
        print(args, file=f)

    process_corpus(args.dataset, args.max_cell_length, args.delimiter, args.title, args.tokenizer_name_or_path,
                   args.output_dir)
    process_query(args.dataset, args.output_dir)
    shutil.copyfile(os.path.join(args.dataset, 'test.jsonl'), os.path.join(args.output_dir, 'label.jsonl'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='WTQ')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--max_cell_length', type=int, default=8)
    parser.add_argument('--delimiter', action='store_true')
    parser.add_argument('--title', action='store_true')
    parser.add_argument('--tokenizer_name_or_path', type=str, default='bert-base-uncased')

    args = parser.parse_args()

    main(args)
