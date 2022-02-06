from typing import Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from src.modules import TransformerLayer, SwitchTransformerLayer, PositionalEmbedding


class LSTMLM(nn.Module):

    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty(vocab_size, hidden_dim)) 
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


class VanillaTransformer(nn.Module):

    def __init__(self, vocab_size: int, d_model: int, num_layers: int, n_heads: int):
        super().__init__()
        self.posemb = PositionalEmbedding(d_model, 0.1, 500)
        self.embedding = nn.Parameter(torch.empty(vocab_size, d_model)) 
        nn.init.normal_(self.embedding)
        self.transformer_layers = nn.ModuleList(TransformerLayer(d_model, n_heads) for _ in range(num_layers))
    
    def forward(self, x):
        mask = (x != 0).unsqueeze(1).unsqueeze(2)
        # mask - batch, 1, 1, seq_len
        # ones are for n_heads, seq_len
        x = nn.functional.embedding(x, self.embedding)
        x = self.posemb(x)
        for layer in self.transformer_layers:
            x = layer(x, mask)
        return nn.functional.linear(x, self.embedding)

class SwitchTransformer(nn.Module):

    def __init__(self, vocab_size: int, d_model: int, num_layers: int, n_heads: int, n_experts: int):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty(vocab_size, d_model)) 
        nn.init.normal_(self.embedding)
        self.transformer_layers = nn.ModuleList(
            SwitchTransformerLayer(d_model, n_heads, n_experts) 
            for _ in range(num_layers))

    def forward(self, x):
        mask = x != 0
        x = nn.functional.embedding(x, self.embedding)
        for layer in self.transformer_layers:
            x = layer(x, mask)
        return nn.functional.linear(x, self.embedding)
