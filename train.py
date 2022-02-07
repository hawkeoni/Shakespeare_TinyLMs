import argparse

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer import Trainer

from src import (
    LMSystem, 
    TinyShakespeareDataModule, 
    LSTMLM,
    VanillaTransformer,
    SwitchTransformer
)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lstm", "transformer", "switch"])
    parser.add_argument("--batch-size", default=64)
    parser.add_argument("--max-length", default=100)
    parser.add_argument("--only-maxlen", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    dm = TinyShakespeareDataModule(
        args.batch_size, 
        args.max_length, 
        args.only_maxlen, 
        100)
    vocab_size = len(dm.vocab)
    if args.model == "lstm":
        net = LSTMLM(vocab_size, 256, 2)
    elif args.model == "transformer":
        net = VanillaTransformer(vocab_size, 256, 4, 8)
    elif args.model == "switch":
        net = SwitchTransformer(vocab_size, 256, 4, 8, 4)
    system = LMSystem(net)
    if args.wandb:
        trainer = Trainer(logger=WandbLogger(project="smol_lms"))
    else:
        trainer = Trainer()
    trainer.fit(system, dm)


if __name__ == "__main__":
    train()
