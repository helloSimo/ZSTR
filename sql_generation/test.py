import os
import math
import random
import jsonlines
import argparse
import collections
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, set_seed
import re
import sys
sys.path.append('..')
from table_utils.table_processor import get_processor
from model import Generator


def get_corpus(dataset, processor):
    tables = []
    print("processing corpus...")
    for i, table in enumerate(tqdm(jsonlines.open(os.path.join('../datasets/', dataset, 'tables.jsonl')))):
        if i == 100:
            break
        tables.append(processor.process_table(table)[1])
    return tables


def get_split_tables(tables):
    pattern1 = re.compile(r'\d+(?:[.,]\d+)*\s*[kmdcg]?[m²³lgbhsd℃jwa]?mile?bit?hour?min?cal?ah?$',
                          re.IGNORECASE)
    pattern2 = re.compile(r'^\d+(\s*[/\\\-_:.]+\s*\d+)+$', re.IGNORECASE)
    pattern3 = re.compile(r'^\(\s*\d+(?:[.,]\d+)*\s*\)$')

    count = 0
    split_tables = []
    for table in tables:
        count += 1
        if len(table['rows']) < 2:
            continue

        types = []
        for col in zip(*(table['rows'])):
            flag = True
            for cell in col:
                cell = cell.strip()
                if not (re.match(pattern1, cell) or re.match(pattern2, cell) or
                        re.match(pattern3, cell) or cell == ''):
                    flag = False
                    break
            if flag:
                types.append('number')
            else:
                types.append('text')
        table['types'] = types
        split_tables.append(table)
    return split_tables, count


def get_split_qas(generator, split_tables, split_count):
    split_qas = generator.generate(tgt_tables=split_tables,
                                   ques_per_passage=math.ceil(split_count / len(split_tables)))
    random.shuffle(split_qas)
    while len(split_qas) > split_count:
        split_qas.pop()
    return split_qas


def main(args):
    set_seed(args.seed)

    # process table corpus
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    processor = get_processor(max_cell_length=args.max_cell_length,
                              max_input_length=args.max_input_length,
                              tokenizer=tokenizer,
                              include_title=args.title,
                              index_row=args.delimiter,
                              row_choose=False)
    tables = get_corpus(args.dataset, processor)

    # collect split corpus from corpus
    # and count qas
    train_tables, train_count = get_split_tables(tables=tables)

    # get generator
    generator = Generator()
    split_qas = generator.generate(tgt_tables=train_tables,
                                   ques_per_passage=math.ceil(train_count / len(train_tables)))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='WTQ', choices=['WTQ', 'WikiSQL', 'NQTables'])
    parser.add_argument('--train_negative_num', type=int, default=8)
    parser.add_argument('--dev_negative_num', type=int, default=8)
    parser.add_argument('--max_cell_length', type=int, default=8)
    parser.add_argument('--max_input_length', type=int, default=128, choices=[128, 256])
    parser.add_argument('--delimiter', action='store_true')
    parser.add_argument('--title', action='store_true')
    parser.add_argument('--tokenizer_name_or_path', type=str, default='bert-base-uncased')

    parser.add_argument('--seed', type=int, default=42)

    main_args = parser.parse_args()

    main(main_args)
