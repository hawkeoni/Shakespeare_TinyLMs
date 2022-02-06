import argparse

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer import Trainer

from src import LMSystem, TinyShakespeareDataModule, LSTMLM


def train():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model", choices=["lstm", "transformer", "switch"])
    batch_size = 64
    maxlength = 100
    dm = TinyShakespeareDataModule(batch_size, maxlength, False, 100)
    net = LSTMLM(len(dm.vocab), 256, 2)
    system = LMSystem(net)
    # trainer = Trainer(logger=WandbLogger(project="smol_lms"))
    trainer = Trainer()
    trainer.fit(system, dm)




if __name__ == "__main__":
    train()