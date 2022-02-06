from typing import Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence




class LSTMLM(nn.Module):

    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty(vocab_size, hidden_dim)) 
        nn.Embedding
        nn.init.normal_(self.embedding)
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pad_token = 0
        self.dropout = nn.Dropout(0.1)
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # x - batch_size, seq_len
        mask = x != self.pad_token
        lengths = mask.sum(dim=1)
        x = nn.functional.embedding(x, self.embedding)
        x = self.dropout(x)
        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        outputs, (hidden, context) = self.rnn(x)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        outputs = self.dropout(outputs)
        # outputs - batch, seq_len, hidden_dim
        # outputs = self.out(outputs)
        outputs = nn.functional.linear(outputs, self.embedding)
        # outputs - batch, seq_len, vocab_size
        return outputs, (hidden, context)

