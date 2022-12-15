import random
from dataclasses import dataclass
from typing import List, Tuple

import datasets
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, BatchEncoding, DataCollatorWithPadding, DefaultDataCollator


from .arguments import DataArguments
from .trainer import TevatronTrainer

import logging
logger = logging.getLogger(__name__)


@dataclass
class QPCollator:
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    tokenizer: PreTrainedTokenizer
    max_q_len: int
    max_p_len: int

    def __call__(self, features):
        qq = [f[0] for f in features]
        dd = [f[1] for f in features]

        if isinstance(qq[0], list):
            qq = sum(qq, [])
        if isinstance(dd[0], list):
            dd = sum(dd, [])

        q_collated = self.tokenizer(
            qq,
            padding='max_length',
            truncation='only_first',
            max_length=self.max_q_len,
            return_tensors="pt",
        )
        d_collated = self.tokenizer(
            dd,
            padding='max_length',
            truncation='only_first',
            max_length=self.max_p_len,
            return_tensors="pt",
        )

        return q_collated, d_collated


@dataclass
class EncodeCollator:
    max_len: int
    tokenizer: PreTrainedTokenizer

    def __call__(self, features):
        text_ids = [x[0] for x in features]
        text_features = [x[1] for x in features]
        collated_features = self.tokenizer(
            text_features,
            padding='max_length',
            truncation='only_first',
            max_length=self.max_len,
            return_tensors="pt",
        )
        return text_ids, collated_features
