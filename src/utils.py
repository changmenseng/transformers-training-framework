import torch
import torch.distributed as dist
from typing import *
from tqdm import tqdm

class Rank0ProcessBar(tqdm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if dist.is_initialized():
            self.disable = dist.get_rank() != 0
