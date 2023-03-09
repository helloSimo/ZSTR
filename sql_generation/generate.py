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


def get_corpus(dataset, processor):
    corpus = collections.OrderedDict()
    tables = {}
    print("processing corpus...")
    for table in tqdm(jsonlines.open(os.path.join('../datasets/', dataset, 'tables.jsonl'))):
        corpus[table['id']], tables[table['id']] = processor.process_table(table)[0]
    return corpus, tables


def get_split_tables(tables, dataset, split):
    pattern1 = re.compile(r'\d+(?:[.,]\d+)*\s*[kmdcg]?[m²³lgbhsd℃jwa]?mile?bit?hour?min?cal?ah?$',
                          re.IGNORECASE)
    pattern2 = re.compile(r'^\d+(\s*[/\\\-_:.]+\s*\d+)+$', re.IGNORECASE)
    pattern3 = re.compile(r'^\(\s*\d+(?:[.,]\d+)*\s*\)$')

    count = 0
    split_tables = {}
    for qa in jsonlines.open(os.path.join('../datasets/', dataset, '{}.jsonl'.format(split))):
        count += 1
        for table_id in qa['table_id']:
            if len(tables[table_id]['rows']) < 2 or table_id in tables:
                continue

            table = tables[table_id]
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
            split_tables[table_id] = table
    return split_tables, count


def get_split_qas(generator, split_corpus, split_count, max_query_length, batch_size):
    split_qas = generator.generate(corpus=split_corpus,
                                   max_length=max_query_length,
                                   ques_per_passage=math.ceil(split_count / len(split_corpus)),
                                   batch_size=batch_size)
    # while len(split_qas) > split_count:
    #     split_qas.pop(random.randrange(len(split_qas)))
    random.shuffle(split_qas)
    while len(split_qas) > split_count:
        split_qas.pop()
    return split_qas


def get_bm25(texts, tokenizer):
    tokenized_texts = []
    print("tokenizing corpus for bm25...")
    for text in tqdm(texts):
        tokens = tokenizer.tokenize(text)
        tokenized_text = tokenizer.convert_tokens_to_string(tokens)
        tokenized_texts.append(tokenized_text)
    return BM25Okapi(tokenized_texts)


def main(args):
    set_seed(args.seed)

    if args.output_dir is None:
        args.output_dir = 'train'
    if not os.path.exists(os.path.join('../datasets/', args.output_dir)):
        os.mkdir(os.path.join('../datasets/', args.output_dir))

    with open(os.path.join('../datasets/', args.output_dir, 'setting.txt'), 'w') as f:
        print(args, file=f)

    # process table corpus
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    processor = get_processor(max_cell_length=args.max_cell_length,
                              max_input_length=args.max_input_length,
                              tokenizer=tokenizer,
                              include_title=args.title,
                              index_row=args.delimiter,
                              row_choose=False)
    corpus, tables = get_corpus(args.dataset, processor)

    # collect split corpus from corpus
    # and count qas
    train_tables, train_count = get_split_tables(tables=tables,
                                                 dataset=args.dataset,
                                                 split='train')
    dev_tables, dev_count = get_split_tables(tables=corpus,
                                             dataset=args.dataset,
                                             split='dev')

    # get generator
    generator = QueryGenerator(model=QGenModel(args.model_name_or_path))

    # get split qas
    train_qas = get_split_qas(generator=generator,
                              split_corpus=train_corpus,
                              split_count=train_count,
                              max_query_length=args.max_query_length,
                              batch_size=args.batch_size)
    dev_qas = get_split_qas(generator=generator,
                            split_corpus=dev_corpus,
                            split_count=dev_count,
                            max_query_length=args.max_query_length,
                            batch_size=args.batch_size)

    bm25 = get_bm25(corpus.values(), tokenizer)
    ids = list(corpus.keys())

    print("building data...")
    for split in ['train', 'dev']:
        if split == 'train':
            negative_num = args.train_negative_num
            split_qas = train_qas
        else:
            negative_num = args.dev_negative_num
            split_qas = dev_qas
        dest_f = jsonlines.open(os.path.join('../datasets/', args.output_dir, '{}.jsonl'.format(split)), 'w')
        for qa in tqdm(split_qas, total=len(split_qas)):
            positive_passages = []
            for id_ in qa['table_id']:
                positive_passages.append(corpus[id_])
            negative_passages = []
            token = tokenizer.tokenize(qa['question'])
            tokenized_question = tokenizer.convert_tokens_to_string(token)
            cnt = 0
            start = 0
            done_flag = False
            while True:
                for id_ in bm25.get_top_n(tokenized_question, ids, n=(start + 1) * 10)[start * 10:]:
                    if id_ not in qa['table_id']:
                        negative_passages.append(corpus[id_])
                        cnt += 1
                        if cnt == negative_num:
                            done_flag = True
                            break
                if done_flag:
                    break
                else:
                    start += 1
            dest_f.write({
                'query': qa['question'],
                'positives': positive_passages,
                'negatives': negative_passages
            })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='WTQ', choices=['WTQ', 'WikiSQL', 'NQTables'])
    parser.add_argument('--output_dir', type=str, default=None)
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
