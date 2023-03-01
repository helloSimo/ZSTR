from typing import List, Dict

import logging
import torch
from tqdm import trange
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logger = logging.getLogger(__name__)


class QGenModel:
    def __init__(self, model_path: str, use_fast: bool = True, device: str = None, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=use_fast)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info("Use pytorch device: {}".format(self.device))
        self.model = self.model.to(self.device)

    def generate(self, corpus: List[str], ques_per_passage: int, top_k: int, max_length: int,
                 top_p: float = None, temperature: float = None) -> List[str]:

        encodings = self.tokenizer(corpus, padding=True, truncation=True, return_tensors="pt")

        # Top-p nucleus sampling
        with torch.no_grad():
            if not temperature:
                outs = self.model.generate(
                    input_ids=encodings['input_ids'].to(self.device),
                    do_sample=True,
                    max_length=max_length,  # 64
                    top_k=top_k,  # 25
                    top_p=top_p,  # 0.95
                    num_return_sequences=ques_per_passage  # 1
                )
            else:
                outs = self.model.generate(
                    input_ids=encodings['input_ids'].to(self.device),
                    do_sample=True,
                    max_length=max_length,  # 64
                    top_k=top_k,  # 25
                    temperature=temperature,
                    num_return_sequences=ques_per_passage  # 1
                )

        return self.tokenizer.batch_decode(outs, skip_special_tokens=True)


class QueryGenerator:
    def __init__(self, model, **kwargs):
        self.model = model

    def generate(self,
                 corpus: Dict[str, str],
                 top_p: int = 0.95,
                 top_k: int = 25,
                 max_length: int = 64,
                 ques_per_passage: int = 1,
                 batch_size: int = 32,):

        logger.info(
            "Starting to Generate {} Questions Per Passage using top-p (nucleus) sampling...".format(ques_per_passage))
        logger.info("Params: top_p = {}".format(top_p))
        logger.info("Params: top_k = {}".format(top_k))
        logger.info("Params: max_length = {}".format(max_length))
        logger.info("Params: ques_per_passage = {}".format(ques_per_passage))
        logger.info("Params: batch size = {}".format(batch_size))

        count = 0
        corpus_ids = list(corpus.keys())
        corpus = [corpus[doc_id] for doc_id in corpus_ids]

        qas = []
        for start_idx in trange(0, len(corpus), batch_size):

            size = len(corpus[start_idx:start_idx + batch_size])
            queries = self.model.generate(
                corpus=corpus[start_idx:start_idx + batch_size],
                ques_per_passage=ques_per_passage,
                max_length=max_length,
                top_p=top_p,
                top_k=top_k,
            )

            assert len(queries) == size * ques_per_passage

            for idx in range(size):
                corpus_id = corpus_ids[start_idx + idx]
                start_id = idx * ques_per_passage
                end_id = start_id + ques_per_passage
                query_set = list(set([q.strip() for q in queries[start_id:end_id]]))

                for query in sorted(query_set):
                    count += 1
                    query_id = "genQ" + str(count)
                    qas.append({
                        'id': query_id,
                        'question': query,
                        'answer': [],
                        'table_id': [corpus_id]
                    })

        return qas
