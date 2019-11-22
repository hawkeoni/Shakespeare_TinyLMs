import pickle
from typing import List, Iterator, Iterable
from collections import Counter
from random import randint, shuffle

import torch
from torch.nn.utils.rnn import pad_sequence


class Vocab:

    def __init__(self, min_freq: int):
        self.min_freq = min_freq
        self.pad_idx = 0
        self.pad_token = "<pad>"
        self.unk_idx = 1
        self.unk_token = "<unk>"
        self.stoi = {"<pad>": 0, "<unk>" : 1}
        self.itos = {0: "<pad>", 1: "<unk>"}
        self.count = Counter()

    def __len__(self):
        return len(self.stoi)

    def add_symbol(self, symbol: str):
        if symbol in self.stoi:
            return
        self.count[symbol] += 1
        if self.count[symbol] > self.min_freq:
            self.stoi[symbol] = len(self.stoi)
            self.itos[len(self.itos)] = symbol

    def fill(self, data: Iterable[str]):
        for c in data:
            self.add_symbol(c)

    def numericalize(self, inputs: List[str]) -> torch.Tensor:
        tensor_inputs = [
            torch.LongTensor([self.stoi.get(c, self.unk_idx) for c in line])
                         for line in inputs]
        return pad_sequence(tensor_inputs, batch_first=True, padding_value=self.pad_idx)

    def denumericalize(self, inputs: torch.Tensor) -> List[str]:
        inputs = inputs.tolist()
        return [''.join([self.itos[c] for c in sample]) for sample in inputs]

    def save_dict(self, filename: str):
        pickle.dump(self, open(filename, 'wb'))

    @classmethod
    def load_dict(cls, filename: str):
        return pickle.load(open(filename, 'rb'))


class GeneratorDataset:

    def __init__(self, filename: str, maxlength: int, min_freq: int, batch_size: int):
        self.filename = filename
        self.maxlength = maxlength
        self.min_freq = min_freq
        self.batch_size = batch_size
        self.data = open(filename, 'r', encoding='utf8').read()
        self.length = len(self.data)
        self.vocab = Vocab(min_freq)
        self.vocab.fill(self.data)

    def split_data(self) -> List[str]:
        splits = []
        start = 0
        while True:
            offset = randint(1, self.maxlength)
            sample = self.data[start: start + offset]
            if len(sample) > 0:
                splits.append(sample)
                start = min(start + offset, self.length)
            else:
                break
            if start >= self.length:
                break
        return splits

    def __iter__(self) -> Iterator[torch.Tensor]:
        splits = self.split_data()
        shuffle(splits)
        for batch_idx in range(len(splits) // self.batch_size + 1):
            batch = splits[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size]
            yield self.vocab.numericalize(batch)
