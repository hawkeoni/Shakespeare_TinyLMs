import torch
from typing import NamedTuple


class LMOutput(NamedTuple):
    output: torch.Tensor
    mask: torch.Tensor