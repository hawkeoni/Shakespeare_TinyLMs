import argparse

import torch
from pytorch_lightning.trainer import Trainer

from src import (
    LMSystem, 
    TinyShakespeareDataModule, 
    LSTMLM,
    VanillaTransformer,
    SwitchTransformer
)

def generate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lstm", "transformer", "switch"])
    parser.add_argument("--ckpt", required=True, type=str)
    args = parser.parse_args()
    datamodule = TinyShakespeareDataModule(
        64,
        100,
        False,
        100)
    vocab = datamodule.vocab
    vocab_size = len(datamodule.vocab)
    if args.model == "lstm":
        net = LSTMLM(vocab_size, 1024, 2)
    elif args.model == "transformer":
        net = VanillaTransformer(vocab_size, 256, 4, 8)
    elif args.model == "switch":
        net = SwitchTransformer(vocab_size, 256, 4, 8, 4)
    net.load_state_dict(torch.load(args.ckpt)["state_dict"])
    net = net.cuda()
    system = LMSystem(net)
    # callback = EarlyStopping("validation_loss", min_delta=0.1, patience=2, mode="min")
    try:
        while True:
            text = input()
            x = vocab.encode(text).unsqueeze(0).cuda()
            y = system.generate(x, 0.8, 100)
            # y - batch, seq_len2
            y = y.cpu()[0]
            print(vocab.decode(y))
    except KeyboardInterrupt:
        print("Finish.")

        



if __name__ == "__main__":
    with torch.no_grad():
        generate()