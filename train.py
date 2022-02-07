import argparse
import datetime

import torch
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
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
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--max-length", default=100, type=int)
    parser.add_argument("--only-maxlen", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    datamodule = TinyShakespeareDataModule(
        args.batch_size, 
        args.max_length, 
        args.only_maxlen, 
        100)
    vocab_size = len(datamodule.vocab)
    if args.model == "lstm":
        net = LSTMLM(vocab_size, 256, 2)
    elif args.model == "transformer":
        net = VanillaTransformer(vocab_size, 256, 4, 8)
    elif args.model == "switch":
        net = SwitchTransformer(vocab_size, 256, 4, 8, 4)
    system = LMSystem(net)
    # callback = EarlyStopping("validation_loss", min_delta=0.1, patience=2, mode="min")
    ckptname = f"{args.model}_{str(datetime.datetime.today())[:-7]}"
    logger = WandbLogger(project="smol_lms", name=ckptname) if args.wandb else None
    callback = ModelCheckpoint(
        f"./checkpoints/{args.model}/{str(datetime.datetime.today())[:-7]}",
        filename='{epoch}-{validation_loss:.2f}-{train_loss:.2f}',
        monitor="validation_loss",
        mode="min",
        save_last=True)
    trainer = Trainer.from_argparse_args(args, logger=logger, callbacks=callback)
    trainer.fit(system, datamodule)
    torch.save(system.state_dict(), f"./{args.model}")


if __name__ == "__main__":
    train()
