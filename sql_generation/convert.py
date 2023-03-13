import os
import jsonlines
import argparse
from typing import List, Dict

import torch
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from transformers import AutoModelWithLMHead, T5Tokenizer, AutoTokenizer
from transformers import set_seed
import sys
sys.path.append('..')
from table_utils.table_processor import get_processor
from model import Generator
from generate import get_corpus, get_split_tables, get_split_qas


def convert_qa(qas: List[Dict],
               convertor: AutoModelWithLMHead,
               tokenizer: T5Tokenizer,
               device: str):
    print("convert sql to natural query...")
    for qa in tqdm(qas, total=len(qas)):
        input_text = "translate Sql to English: %s </s>" % qa['question']
        features = tokenizer([input_text], return_tensors='pt')

        output = convertor.generate(input_ids=features['input_ids'].to(device),
                                    attention_mask=features['attention_mask'].to(device),
                                    max_new_tokens=32)
        qa['question'] = tokenizer.decode(output[0], skip_special_tokens=True)


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
    dev_tables, dev_count = get_split_tables(tables=tables,
                                             dataset=args.dataset,
                                             split='dev')

    # get generator
    generator = Generator()

    # get split qas
    train_qas = get_split_qas(generator=generator,
                              split_tables=train_tables,
                              split_count=train_count,)
    dev_qas = get_split_qas(generator=generator,
                            split_tables=dev_tables,
                            split_count=dev_count)

    # convert split qas
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    c_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    convertor = AutoModelWithLMHead.from_pretrained(args.model_name_or_path).to(device)

    convert_qa(qas=train_qas,
               convertor=convertor,
               tokenizer=c_tokenizer,
               device=device)
    convert_qa(qas=dev_qas,
               convertor=convertor,
               tokenizer=c_tokenizer,
               device=device)
    del convertor
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

    parser.add_argument('--model_name_or_path', type=str, default='mrm8488/t5-base-finetuned-wikiSQL-sql-to-en')
    parser.add_argument('--seed', type=int, default=42)

    main_args = parser.parse_args()

    main(main_args)
