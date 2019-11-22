from src.model import LSTMGen
from src.data import Vocab


def generate(model_name: str, vocab_name: str, length: int, **kwargs):
    model = LSTMGen.load_model(f"{model_name}.pt")
    vocab = Vocab.load_dict(f"{vocab_name}.pickle")
    while True:
        print("> ", end="")
        line = [input().strip()]
        x = vocab.numericalize(line)
        output = model.generate(x, length)
        print(line[0], vocab.denumericalize(output)[0], sep='')
