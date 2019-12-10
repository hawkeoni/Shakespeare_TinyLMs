import argparse

from src.train import train
from src.generate import generate


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    # Train subparser
    subparser = subparsers.add_parser("train")
    subparser.set_defaults(callback=train)
    subparser.add_argument("--model-name", type=str, default="model")
    subparser.add_argument("--vocab-name", type=str, default="vocab")
    subparser.add_argument("--use-gru", dest="use_gru", action="store_true", default=False)
    subparser.add_argument("--max-length", type=int, default=100)
    subparser.add_argument("--batch-size", type=int, default=64)
    subparser.add_argument("--charemb-dim", type=int, default=300)
    subparser.add_argument("--hidden-dim", type=int, default=256)
    subparser.add_argument("--num-layers", type=int, default=2)
    subparser.add_argument("--epochs", type=int, default=50)
    # Generation subparser
    subparser = subparsers.add_parser("generate")
    subparser.set_defaults(callback=generate)
    subparser.add_argument("--model-name", type=str, default="model")
    subparser.add_argument("--vocab-name", type=str, default="vocab")
    subparser.add_argument("--use-gru", dest="use_gru", action="store_true", default=False)
    subparser.add_argument("--length", type=int, default=100)
    subparser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()
    args.callback(**vars(args))


if __name__ == "__main__":
    main()
