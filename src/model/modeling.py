import random
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from transformers import GPT2Model
from transformers.utils.generic import ModelOutput
import math
from .configuration import GPT2ClassifierConfig

class GPT2Classifier(GPT2Model):
    config_class = GPT2ClassifierConfig

    def __init__(self, config):
        super().__init__(config)
        self.class_head = nn.Linear(
            config.hidden_size, 
            config.n_classes, 
            bias=False
        )

    def forward(
        self,
        input_ids, 
        attention_mask=None, 
        labels=None
    ):
        hidden_states = super().forward(input_ids, attention_mask).last_hidden_state # (batch_size, max_length, hidden_size)
        d = attention_mask.sum(axis=1) - 1 # (batch_size)
        reps = torch.gather(
            input=hidden_states,
            dim=1, index=d[:, None, None].repeat(1, 1, hidden_states.shape[-1])
        ).squeeze(1) # (batch_size, hidden_dim)
        logits = self.class_head(reps)

        if self.training:
            loss = F.cross_entropy(logits, labels)
            return ModelOutput(loss=loss)
        
        return ModelOutput(logits=logits)
