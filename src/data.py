import random
import pickle
from functools import partial
from collections import Counter
from typing import List, Iterator, Iterable, Union

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from pytorch_lightning.core.datamodule import LightningDataModule


class Vocab:

    def __init__(self, min_freq: int):
        self.min_freq = min_freq
        self.pad = 0
        self.unk = 1
        self.bos = 2
        self.eos = 3
        self.stoi = {"<pad>": 0, "<unk>" : 1, "<bos>": 2, "<eos>": 3}
        self.itos = {v: k for k, v in self.stoi.items()}
        self.count = Counter()

    def __len__(self):
        return len(self.stoi)

    def _add_symbol(self, symbol: str):
        if symbol in self.stoi:
            return
        self.count[symbol] += 1
        if self.count[symbol] > self.min_freq:
            self.stoi[symbol] = len(self.stoi)
            self.itos[len(self.itos)] = symbol

    def _fill(self, data: Iterable[str]):
        for c in data:
            self._add_symbol(c)
    
    def tokenize(self, text: str) -> List[str]:
        """
        N.B. Due to unks and other multichar tokens
        tokenization and detokenization does not
        work as one can expect
        """
        tokens = [self.itos[self.bos]]
        inner_tokens = list(text)
        for token in inner_tokens:
            tokens.append(token if token in self.stoi else self.itos[self.unk])
        tokens.append(self.itos[self.eos])
        return tokens
    
    def numericalize(self, tokens: List[str]) -> torch.Tensor:
        return torch.LongTensor([self.stoi[token] for token in tokens])

    def denumericalize(self, token_ids: Union[List[int], torch.Tensor]) -> List[str]:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().tolist()
        return [self.itos[tid] for tid in token_ids]

    def detokenize(self, tokens: List[str]) -> str:
        if tokens[0] == "<bos>":
            tokens = tokens[1:]
        if tokens[-1] == "<eos>":
            tokens = tokens[:-1]
        return "".join(tokens)
    
    def encode(self, text: str) -> torch.Tensor:
        return self.numericalize(self.tokenize(text))
    
    def decode(self, token_ids: torch.Tensor) -> str:
        return self.detokenize(self.denumericalize(token_ids))

    def to_file(self, filename: str):
        pickle.dump(self, open(filename, 'wb'))

    @classmethod
    def from_file(cls, filename: str):
        return pickle.load(open(filename, 'rb'))


class TinyShakespeareDataset(Dataset):

    def __init__(self, split: str, maxlength: int, only_maxlen: bool = False, seed: int = 42):
        assert split in {"train", "validation", "test"}
        self.split = split
        self.raw_data = load_dataset("tiny_shakespeare")[split]["text"][0]
        self.maxlength = maxlength
        self.only_maxlen = only_maxlen
        self.length = len(self.raw_data)
        self.samples = []
        random.seed(seed)
        self.fill_samples()
        self.vocab = None
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def fill_samples(self):
        start = 0
        data_len = len(self.raw_data)
        while start < data_len:
            end = min(start + random.randint(0, self.maxlength), data_len)
            self.samples.append(self.raw_data[start: end])
            start = end
    
    def build_vocab(self, min_freq: int = 100):
        assert self.split == "train"
        vocab = Vocab(min_freq)
        vocab._fill(self.raw_data)
        return vocab
    

    def set_vocab(self, vocab: Vocab):
        self.vocab = vocab


    def get_text(self, index: int) -> str:
        return self.samples[index]

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.vocab.encode(self.get_text(index))


def prepare_data(maxlength: int, only_maxlen: bool, min_freq: int):
    shakespeare_datasets = tuple(
        TinyShakespeareDataset(split, maxlength, only_maxlen) 
        for split in ["train", "validation", "test"]
        )
    vocab = shakespeare_datasets[0].build_vocab(min_freq)
    for ds in shakespeare_datasets:
        ds.set_vocab(vocab)
    return shakespeare_datasets, vocab



class TinyShakespeareDataModule(LightningDataModule):

    def __init__(self, batch_size: int, maxlength: int, only_maxlen: bool, min_freq: int):
        super().__init__()
        (tr, v, te), vocab = prepare_data(maxlength, only_maxlen, min_freq)
        self.tr = tr
        self.v = v
        self.te = te
        self.vocab = vocab
        self.collate_fn = partial(pad_sequence, batch_first=True, padding_value=vocab.pad)
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.tr, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn, num_workers=16)

    def val_dataloader(self):
        return DataLoader(self.v, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn, num_workers=16)

    def test_dataloader(self):
        return DataLoader(self.te, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn, num_workers=16)
