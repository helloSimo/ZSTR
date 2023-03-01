import os
import math
import random
import jsonlines
import argparse
import collections
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, set_seed
import sys
from model import QGenModel, QueryGenerator
import torch


def main(args):
    set_seed(42)

    # get generator
    generator = QueryGenerator(model=QGenModel(args.model_name_or_path))

    corpus = {
        '1': "col : Team | County | Wins | Years won  row 1 : Greystones | Wicklow | 1 | 2011  "
             "row 2 : Ballymore Eustace | Kildare | 1 | 2010  row 3 : Maynooth | Kildare | 1 | "
             "2009  row 4 : Ballyroan Abbey | Laois | 1 | 2008  row 5 : Fingal Ravens | Dublin "
             "| 1 | 2007  row 6 : Confey | Kildare | 1 | 2006  row 7 : Crettyard | Laois | 1 | "
             "2005  row 8 : Wolfe Tones | Meath | 1 | 2004  row 9 : Dundalk Gaels | Louth | 1 | "
             "2003",
        '2': 'col :  | Plain catgut | Chromic catgut | Polyglycolide\n(P.G.A.) | Polydioxanone '
             '(PDS)  row 1 : Tensile strength | strength retention for at least 7 days. | Maintains'
             'strength for 10–14 days | 84 % at 2 weeks, 23 % | 80 % at 2 weeks, 44 %  row 2 : '
             'Structure | Monofilament | Monofilament | Braided | Monofilament  row 3 : Precautions '
             '| special precautions should be taken in patients with | it is absorbed much faster when'
             ' used in | special precautions should be taken in elderly patients | the pds suture knots'
             ' must be  row 4 : Type of absorption | proteolytic enzymatic digest | proteolytic enzymatic '
             'digest | absorption by hydrolysis complete between 60 and | wound support can remain up to '
             '42 days  row 5 : Description | absorbable biological suture material. plain | absorbable '
             'biological suture material. ch | it is a synthetic absorbable suture | it is a synthetic '
             'absorbable suture  row 6 : Contraindications | not recommended for incisions that require '
             '| not recommended for an incision that requires | this suture being absorbable should not '
             '| this type of suture being absorbable  row 7 : Composition | ? | Natural purified collagen '
             '| Polyglycolic acid | polyester and poly ( p - di  row 8 : Indications | for all surgical '
             'procedures especially when tissues that | for all surgical procedures, especially for tissues '
             '| subcutaneous, intracutaneous closure | pds is particularly useful where the combination '
    }
    split_qas = generator.generate(corpus=corpus,
                                   max_length=128,
                                   ques_per_passage=5,
                                   batch_size=2)
    for qa in split_qas:
        print(qa['question'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='BeIR/query-gen-msmarco-t5-base-v1')

    main_args = parser.parse_args()

    main(main_args)
