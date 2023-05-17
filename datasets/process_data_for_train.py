import os
import jsonlines
import argparse
import collections
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer
import sys
sys.path.append('..')
from table_utils.table_processor import get_processor


def get_corpus(dataset, processor):
    corpus = collections.OrderedDict()
    print("processing corpus for bm25...")
    for table in tqdm(jsonlines.open(os.path.join(dataset, 'tables.jsonl'))):
        corpus[table['id']] = processor.process_table(table)[0]
    return corpus


def main(args):
    if args.output_dir is None:
        args.output_dir = 'train'
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    with open(os.path.join(args.output_dir, 'setting.txt'), 'w') as f:
        print(args, file=f)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    processor = get_processor(max_cell_length=args.max_cell_length,
                              max_input_length=256,
                              tokenizer=tokenizer,
                              include_title=args.title,
                              index_row=args.delimiter,
                              row_choose=True)
    corpus = get_corpus(args.dataset, processor)
    bm25 = BM25Okapi(corpus.values())
    ids = list(corpus.keys())

    print("building data...")
    for split in ['train', 'dev']:
        if split == 'train':
            negative_num = args.train_negative_num
        else:
            negative_num = args.dev_negative_num
        dest_f = jsonlines.open(os.path.join(args.output_dir, '{}.jsonl'.format(split)), 'w')
        for i, qa in tqdm(enumerate(jsonlines.open(os.path.join(args.dataset, '{}.jsonl'.format(split))))):
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
                for id_ in bm25.get_top_n(tokenized_question, ids, n=(start+1)*10)[start*10:]:
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
    parser.add_argument('--dataset', type=str, default='WTQ')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--train_negative_num', type=int, default=8)
    parser.add_argument('--dev_negative_num', type=int, default=8)
    parser.add_argument('--max_cell_length', type=int, default=8)
    parser.add_argument('--delimiter', action='store_true')
    parser.add_argument('--title', action='store_true')
    parser.add_argument('--tokenizer_name_or_path', type=str, default='bert-base-uncased')

    args = parser.parse_args()

    main(args)
