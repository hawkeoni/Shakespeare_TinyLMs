from typing import Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class LSTMGen(nn.Module):

    def __init__(self, vocab_size: int, charemb_dim: int, hidden_dim: int, num_layers: int, pad_token: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.charemb_dim = charemb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pad_token = pad_token
        self.dropout = nn.Dropout(0.1)
        self.embedding = nn.Embedding(vocab_size, charemb_dim)
        self.rnn = nn.LSTM(charemb_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # x - batch_size, seq_len
        mask = x != self.pad_token
        lengths = mask.sum(dim=1)
        x = self.embedding(x)
        x = self.dropout(x)
        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        outputs, (hidden, context) = self.rnn(x)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        outputs = self.dropout(outputs)
        # outputs - batch, seq_len, hidden_dim
        outputs = self.out(outputs)
        # outputs - batch, seq_len, vocab_size
        return outputs, (hidden, context)

    def generate(self, x: torch.Tensor, maxlen: int) -> torch.Tensor:
        # x - 1, seq_len
        outputs, (hidden, context) = self(x)
        results = outputs.new_zeros((outputs.size(0), maxlen), dtype=torch.long)
        # [:, -1] - batch_size, vocab_size
        inputs = torch.argmax(outputs[:, -1], dim=1).unsqueeze(1) # 1, 1
        results[:, 0] = inputs[:, 0]
        for i in range(1, maxlen):
            outputs, (hidden, context) = self.rnn(self.embedding(inputs), (hidden, context))
            outputs = self.out(outputs)
            inputs = torch.argmax(outputs[:, -1], dim=1).unsqueeze(1)  # 1, 1
            results[:, i] = inputs[:, 0]
        return results

    def save_model(self, filename: str):
        hparams = {
            "vocab_size": self.vocab_size,
            "charemb_dim": self.charemb_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "pad_token": self.pad_token
        }
        state_dict = self.state_dict()
        torch.save({"hparams": hparams, "state_dict": state_dict}, filename)

    @classmethod
    def load_model(cls, filename: str) -> nn.Module:
        d = torch.load(filename, lambda storage, loc: storage)
        hparams = d["hparams"]
        state_dict = d["state_dict"]
        model = cls(**hparams)
        model.load_state_dict(state_dict)
        return model
