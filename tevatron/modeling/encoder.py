import copy
import json
import os
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn, Tensor
import torch.distributed as dist
from transformers import PreTrainedModel, AutoModel, DPRContextEncoder
from transformers.file_utils import ModelOutput

from tevatron.arguments import ModelArguments, \
    TevatronTrainingArguments as TrainingArguments

import logging

logger = logging.getLogger(__name__)


@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class EncoderPooler(nn.Module):
    def __init__(self, **kwargs):
        super(EncoderPooler, self).__init__()
        self._config = {}

    def forward(self, q_reps, p_reps):
        raise NotImplementedError('EncoderPooler is an abstract class')

    def load(self, model_dir: str):
        pooler_path = os.path.join(model_dir, 'pooler.pt')
        if pooler_path is not None:
            if os.path.exists(pooler_path):
                logger.info(f'Loading Pooler from {pooler_path}')
                state_dict = torch.load(pooler_path, map_location='cpu')
                self.load_state_dict(state_dict)
                return
        logger.info("Training Pooler from scratch")
        return

    def save_pooler(self, save_path):
        torch.save(self.state_dict(), os.path.join(save_path, 'pooler.pt'))
        with open(os.path.join(save_path, 'pooler_config.json'), 'w') as f:
            json.dump(self._config, f)


class EncoderModel(nn.Module):
    MODEL_CLS = AutoModel
    DPR_CLS = DPRContextEncoder

    def __init__(self,
                 lm_q: PreTrainedModel,
                 lm_p: PreTrainedModel,
                 pooler: nn.Module = None,
                 untie_encoder: bool = False,
                 negatives_x_device: bool = False
                 ):
        super().__init__()
        self.lm_q = lm_q
        self.lm_p = lm_p
        self.pooler = pooler
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.negatives_x_device = negatives_x_device
        self.untie_encoder = untie_encoder
        if self.negatives_x_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None):
        q_reps = self.encode_query(query)
        p_reps = self.encode_passage(passage)

        # for inference
        if q_reps is None or p_reps is None:
            return EncoderOutput(
                q_reps=q_reps,
                p_reps=p_reps
            )

        # for training
        if self.negatives_x_device:
            q_reps = self._dist_gather_tensor(q_reps)
            p_reps = self._dist_gather_tensor(p_reps)

        scores = self.compute_similarity(q_reps, p_reps)
        scores = scores.view(q_reps.size(0), -1)

        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (p_reps.size(0) // q_reps.size(0))

        loss = self.compute_loss(scores, target)
        if self.negatives_x_device:
            loss = loss * self.world_size  # counter average weight reduction
        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )

    @staticmethod
    def build_pooler(model_args):
        return None

    @staticmethod
    def load_pooler(weights, **config):
        return None

    def encode_passage(self, psg):
        raise NotImplementedError('EncoderModel is an abstract class')

    def encode_query(self, qry):
        raise NotImplementedError('EncoderModel is an abstract class')

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    @classmethod
    def build_for_train(
            cls,
            model_args: ModelArguments,
            train_args: TrainingArguments,
            **hf_kwargs,
    ):
        # load local
        if os.path.isdir(model_args.model_name_or_path):
            if model_args.untie_encoder:
                _qry_model_path = os.path.join(model_args.model_name_or_path, 'query_model')
                _psg_model_path = os.path.join(model_args.model_name_or_path, 'passage_model')
                logger.info(f'loading query model weight from {_qry_model_path}')
                lm_q = cls.MODEL_CLS.from_pretrained(_qry_model_path, **hf_kwargs)
                logger.info(f'loading passage model weight from {_psg_model_path}')
                if model_args.dpr:
                    lm_p = cls.DPR_CLS.from_pretrained(_psg_model_path, **hf_kwargs)
                else:
                    lm_p = cls.MODEL_CLS.from_pretrained(_psg_model_path, **hf_kwargs)
            else:
                lm_q = cls.MODEL_CLS.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
                lm_p = lm_q
        # load pre-trained
        else:
            if model_args.untie_encoder:
                lm_q = cls.MODEL_CLS.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
                if model_args.additional_model_name is None:
                    lm_p = copy.deepcopy(lm_q)
                else:
                    if model_args.dpr:
                        lm_p = cls.DPR_CLS.from_pretrained(model_args.additional_model_name, **hf_kwargs)
                    else:
                        lm_p = cls.MODEL_CLS.from_pretrained(model_args.additional_model_name, **hf_kwargs)
            else:
                lm_q = cls.MODEL_CLS.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
                lm_p = lm_q

        if model_args.add_pooler:
            pooler = cls.build_pooler(model_args)
        else:
            pooler = None

        model = cls(
            lm_q=lm_q,
            lm_p=lm_p,
            pooler=pooler,
            negatives_x_device=train_args.negatives_x_device,
            untie_encoder=model_args.untie_encoder
        )
        return model

    @classmethod
    def load_for_encode(
            cls,
            model_name_or_path,
            dpr,
            encode_is_qry,
            **hf_kwargs,
    ):
        lm_q = None
        lm_p = None
        _qry_model_path = os.path.join(model_name_or_path, 'query_model')
        _psg_model_path = os.path.join(model_name_or_path, 'passage_model')
        # from _qry_model_path or _psg_model_path
        if os.path.isdir(model_name_or_path) and os.path.exists(_qry_model_path):
            logger.info(f'found separate weight for query/passage encoders')
            if encode_is_qry:
                logger.info(f'loading query model weight from {_qry_model_path}')
                lm_q = cls.MODEL_CLS.from_pretrained(_qry_model_path, **hf_kwargs)
            else:
                logger.info(f'loading passage model weight from {_psg_model_path}')
                if dpr:
                    lm_p = cls.DPR_CLS.from_pretrained(_psg_model_path, **hf_kwargs)
                else:
                    lm_p = cls.MODEL_CLS.from_pretrained(_psg_model_path, **hf_kwargs)
        # from model_name_or_path
        else:
            logger.info(f'try loading tied weight')
            logger.info(f'loading model weight from {model_name_or_path}')
            if encode_is_qry:
                lm_q = cls.MODEL_CLS.from_pretrained(model_name_or_path, **hf_kwargs)
            else:
                if dpr:
                    lm_p = cls.DPR_CLS.from_pretrained(model_name_or_path, **hf_kwargs)
                else:
                    lm_p = cls.MODEL_CLS.from_pretrained(model_name_or_path, **hf_kwargs)

        pooler_weights = os.path.join(model_name_or_path, 'pooler.pt')
        pooler_config = os.path.join(model_name_or_path, 'pooler_config.json')
        if os.path.exists(pooler_weights) and os.path.exists(pooler_config):
            logger.info(f'found pooler weight and configuration')
            with open(pooler_config) as f:
                pooler_config_dict = json.load(f)
            pooler = cls.load_pooler(model_name_or_path, **pooler_config_dict)
        else:
            pooler = None

        # lm_q or lm_p
        model = cls(
            lm_q=lm_q,
            lm_p=lm_p,
            pooler=pooler,
            untie_encoder=False
        )
        return model

    def save(self, output_dir: str):
        if self.untie_encoder:
            os.makedirs(os.path.join(output_dir, 'query_model'))
            os.makedirs(os.path.join(output_dir, 'passage_model'))
            self.lm_q.save_pretrained(os.path.join(output_dir, 'query_model'))
            self.lm_p.save_pretrained(os.path.join(output_dir, 'passage_model'))
        else:
            self.lm_q.save_pretrained(output_dir)
        if self.pooler:
            self.pooler.save_pooler(output_dir)
