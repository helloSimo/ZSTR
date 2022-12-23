import random
from typing import List, Tuple
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, BatchEncoding
from tevatron.arguments import DataArguments
from tevatron.trainer import TevatronTrainer


class TrainDevDataset(Dataset):
    def __init__(self, data_args: DataArguments, is_train: bool, cache_dir: str, trainer: TevatronTrainer = None):
        if is_train:
            dataset_split = 'train'
        else:
            dataset_split = 'dev'
        self.dataset = load_dataset(data_args.dataset_name,
                                    data_args.dataset_language,
                                    data_files=data_args.train_path, cache_dir=cache_dir,
                                    use_auth_token=True)[dataset_split]

        self.data_args = data_args
        self.trainer = trainer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item) -> Tuple[BatchEncoding, List[BatchEncoding]]:
        group = self.dataset[item]
        epoch = int(self.trainer.state.epoch)

        _hashed_seed = hash(item + self.trainer.args.seed)

        qry = group['query']

        passages = []
        group_positives = group['positives']
        group_negatives = group['negatives']

        if self.data_args.positive_passage_no_shuffle:
            pos_psg = group_positives[0]
        else:
            pos_psg = group_positives[(_hashed_seed + epoch) % len(group_positives)]
        passages.append(pos_psg)

        negative_size = self.data_args.train_n_passages - 1
        if len(group_negatives) < negative_size:
            negs = random.choices(group_negatives, k=negative_size)
        elif self.data_args.train_n_passages == 1:
            negs = []
        elif self.data_args.negative_passage_no_shuffle:
            negs = group_negatives[:negative_size]
        else:
            _offset = epoch * negative_size % len(group_negatives)
            negs = [x for x in group_negatives]
            random.Random(_hashed_seed).shuffle(negs)
            negs = negs * 2
            negs = negs[_offset: _offset + negative_size]

        for neg_psg in negs:
            passages.append(neg_psg)

        return qry, passages


class EncodeDataset(Dataset):
    input_keys = ['id', 'text']

    def __init__(self, data_args: DataArguments, cache_dir: str):
        data_files = data_args.encode_in_path
        if data_files:
            data_files = {'train': data_files}
        self.dataset = load_dataset(data_args.dataset_name,
                                    data_args.dataset_language,
                                    data_files=data_files, cache_dir=cache_dir,
                                    use_auth_token=True)['train']
        self.dataset = self.dataset.shard(data_args.encode_num_shard, data_args.encode_shard_index)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item) -> Tuple[str, BatchEncoding]:
        text_id, text = (self.dataset[item][f] for f in self.input_keys)
        return text_id, text
