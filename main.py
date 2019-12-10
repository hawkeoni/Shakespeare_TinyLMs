import argparse

from src.train import train
from src.generate import generate


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    # Train subparser
    subparser = subparsers.add_parser("train")
    subparser.set_defaults(callback=train)
    subparser.add_argument("--model-name", type=str, default="model",
                           help="Name of the model for saving. Suffix `.pt` will be added, "
                                "as it will be a serialized pytorch weights file.")
    subparser.add_argument("--vocab-name", type=str, default="vocab",
                           help="Name of the vocab for saving. Suffix `.pickle` "
                                "will be added, as it will be a pickled object.")
    subparser.add_argument("--use-gru", dest="use_gru", action="store_true", default=False,
                           help="boolean, set it if you want to use GRU. By default the model uses LSTM.")
    subparser.add_argument("--max-length", type=int, default=100, help="Length of sample in batch.")
    subparser.add_argument("--batch-size", type=int, default=64, help="Batch size for training.")
    subparser.add_argument("--charemb-dim", type=int, default=300, help="Embedding dimension size for characters.")
    subparser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension size for RNN.")
    subparser.add_argument("--num-layers", type=int, default=2, help="Number of stacked layers in RNN.")
    subparser.add_argument("--epochs", type=int, default=50, help="Number of epochs for training.")
    # Generation subparser
    subparser = subparsers.add_parser("generate")
    subparser.set_defaults(callback=generate)
    subparser.add_argument("--model-name", type=str, default="model", help="Model name to load from.")
    subparser.add_argument("--vocab-name", type=str, default="vocab", help="Vocab name to load from.")
    subparser.add_argument("--use-gru", dest="use_gru", action="store_true", default=False,
                           help="Set this flag, if model uses GRU instead of LSTM.")
    subparser.add_argument("--length", type=int, default=1000, help="Length of text to be generated.")
    subparser.add_argument("--temperature", type=float, default=1.0,
                           help="Divisor before softmax. The closer to 0, the more confident and conservative"
                                "the model will be, the bigger the more random outputs will be produced.")
    args = parser.parse_args()
    args.callback(**vars(args))


if __name__ == "__main__":
    main()
