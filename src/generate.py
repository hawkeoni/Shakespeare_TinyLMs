from src.model import LSTMGen, GRUGen
from src.data import Vocab


def generate(model_name: str, vocab_name: str, length: int, **kwargs):
    if kwargs["use_gru"]:
        model = GRUGen.load_model(f"{model_name}.pt").eval()
    else:
        model = LSTMGen.load_model(f"{model_name}.pt").eval()
    vocab = Vocab.load_dict(f"{vocab_name}.pickle")
    while True:
        print("> ", end="")
        line = [input().strip()]
        x = vocab.numericalize(line)
        output = model.generate(x, length)
        print(line[0], vocab.denumericalize(output)[0], sep='')
