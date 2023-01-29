import torch
import torch.nn as nn
from torch import Tensor
import logging
from transformers import T5EncoderModel
from .encoder import EncoderModel, EncoderPooler

logger = logging.getLogger(__name__)


class DensePooler(EncoderPooler):
    def __init__(self, input_dim: int = 768, output_dim: int = 768, tied=True, normalize=False):
        super(DensePooler, self).__init__()
        self.normalize = normalize
        self.linear_q = nn.Linear(input_dim, output_dim)
        if tied:
            self.linear_p = self.linear_q
        else:
            self.linear_p = nn.Linear(input_dim, output_dim)
        self._config = {'input_dim': input_dim, 'output_dim': output_dim, 'tied': tied, 'normalize': normalize}


class T5Model(EncoderModel):
    TRANSFORMER_CLS = T5EncoderModel
    def encode_passage(self, psg):
        if psg is None:
            return None
        psg_out = self.lm_p(**psg, return_dict=True)
        p_hidden = psg_out.last_hidden_state
        p_hidden = p_hidden.masked_fill(~psg['attention_mask'][..., None].bool(), 0.0)
        if self.pooler is not None:
            p_reps = self.pooler(p=p_hidden)  # D * d
        else:
            p_reps = p_hidden.sum(dim=1) / psg['attention_mask'].sum(dim=1)[..., None]
        return p_reps

    def encode_query(self, qry):
        if qry is None:
            return None
        qry_out = self.lm_q(**qry, return_dict=True)
        q_hidden = qry_out.last_hidden_state
        q_hidden = q_hidden.masked_fill(~qry['attention_mask'][..., None].bool(), 0.0)
        if self.pooler is not None:
            q_reps = self.pooler(q=q_hidden)
        else:
            q_reps = q_hidden.sum(dim=1) / qry['attention_mask'].sum(dim=1)[..., None]
        return q_reps

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))

    @staticmethod
    def load_pooler(model_weights_file, **config):
        pooler = DensePooler(**config)
        pooler.load(model_weights_file)
        return pooler

    @staticmethod
    def build_pooler(model_args):
        pooler = DensePooler(
            model_args.projection_in_dim,
            model_args.projection_out_dim,
            tied=not model_args.untie_encoder,
            normalize=model_args.normalize
        )
        pooler.load(model_args.model_name_or_path)
        return pooler
