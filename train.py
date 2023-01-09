import logging
import os
import sys

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
    EarlyStoppingCallback
)

from tevatron.arguments import ModelArguments, DataArguments, \
    TevatronTrainingArguments as TrainingArguments
from tevatron.collator import QPCollator
from tevatron.dataset import TrainDevDataset
from tevatron.modeling import DenseModel
from tevatron.trainer import TevatronTrainer as Trainer, GCTrainer

logger = logging.getLogger(__name__)


def compute_metrics(eval_prediction):
    indices = np.argmax(eval_prediction.predictions, 1)
    acc = (eval_prediction.label_ids == indices).sum()
    acc /= eval_prediction.predictions.shape[0]

    return {
        "acc": acc,
    }


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            f"Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)

    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir
    )
    model = DenseModel.build_for_train(
        model_args,
        training_args,
        cache_dir=model_args.cache_dir,
    )

    if training_args.local_rank > 0:
        print("Waiting for main process to perform the mapping")
        torch.distributed.barrier()
    train_dataset = TrainDevDataset(data_args=data_args, is_train=True,
                                    cache_dir=data_args.data_cache_dir or model_args.cache_dir,
                                    negative_num=data_args.train_negative_num)
    dev_dataset = TrainDevDataset(data_args=data_args, is_train=False,
                                  cache_dir=data_args.data_cache_dir or model_args.cache_dir,
                                  negative_num=data_args.dev_negative_num)
    if training_args.local_rank == 0:
        print("Loading results from main process")
        torch.distributed.barrier()

    trainer_cls = GCTrainer if training_args.grad_cache else Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=QPCollator(
            tokenizer=tokenizer,
            max_p_len=data_args.p_max_len,
            max_q_len=data_args.q_max_len
        ),
        compute_metrics=compute_metrics
    )
    if training_args.early_stop > 0:
        callback = EarlyStoppingCallback(early_stopping_patience=training_args.early_stop)
        trainer.add_callback(callback)
    train_dataset.trainer = trainer
    dev_dataset.trainer = trainer

    trainer.train()  # TODO: resume training
    # trainer.save_model()


if __name__ == "__main__":
    main()
