import os
import math
import random
import jsonlines
import argparse
import collections

import torch.cuda
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, set_seed
import sys

sys.path.append('..')
from table_utils.table_processor import get_processor

from model import QGenModel, QueryGenerator


def get_corpus(dataset, processor):
    corpus = collections.OrderedDict()
    print("processing corpus...")
    for table in tqdm(jsonlines.open(os.path.join('../datasets/', dataset, 'tables.jsonl'))):
        corpus[table['id']] = processor.process_table(table)[0]
    return corpus


def get_split_corpus(corpus, dataset, split):
    count = 0
    split_corpus = {}
    for qa in jsonlines.open(os.path.join('../datasets/', dataset, '{}.jsonl'.format(split))):
        count += 1
        for table_id in qa['table_id']:
            split_corpus[table_id] = corpus[table_id]
    return split_corpus, count


def get_split_qas(generator, split_corpus, target_count, max_query_length, batch_size):
    split_qas = generator.generate(corpus=split_corpus,
                                   max_length=max_query_length,
                                   ques_per_passage=math.ceil(target_count / len(split_corpus)),
                                   batch_size=batch_size)
    random.shuffle(split_qas)
    while len(split_qas) > target_count:
        split_qas.pop()
    return split_qas


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
    corpus = get_corpus(args.dataset, processor)

    # collect split corpus from corpus
    # and count qas
    train_corpus, train_count = get_split_corpus(corpus=corpus,
                                                 dataset=args.dataset,
                                                 split='train')
    dev_corpus, dev_count = get_split_corpus(corpus=corpus,
                                             dataset=args.dataset,
                                             split='dev')

    # get generator
    model = QGenModel(args.model_name_or_path)
    generator = QueryGenerator(model=model)

    # get split qas
    train_qas = get_split_qas(generator=generator,
                              split_corpus=train_corpus,
                              target_count=train_count*args.multiple,
                              max_query_length=args.max_query_length,
                              batch_size=args.batch_size)
    dev_qas = get_split_qas(generator=generator,
                            split_corpus=dev_corpus,
                            target_count=dev_count*args.multiple,
                            max_query_length=args.max_query_length,
                            batch_size=args.batch_size)
    del model
    del generator
    torch.cuda.empty_cache()

    bm25 = BM25Okapi(corpus.values())
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

    parser.add_argument('--model_name_or_path', type=str, default='BeIR/query-gen-msmarco-t5-base-v1')
    parser.add_argument('--max_query_length', type=int, default=32)
    parser.add_argument('--multiple', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)

    main_args = parser.parse_args()

    main(main_args)
