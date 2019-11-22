import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD = True
except BaseException as e:
    print("Failed to import tensorboard")
    print(e)
    TENSORBOARD = False

from src.model import LSTMGen
from src.data import GeneratorDataset


def train_epoch(model: nn.Module, optimizer: optim.Optimizer, dataset: GeneratorDataset) -> float:
    model.train()
    model_device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    for i, batch in tqdm(enumerate(dataset)):
        batch = batch.to(model_device)
        optimizer.zero_grad()
        outputs, (_, _) = model(batch)
        outputs = outputs[:, :-1]
        batch = batch[:, 1:]
        # outputs - batch_size, seq_len - 1, vocab_size
        # batch - batch_size, seq_len - 1
        active_loss_mask = batch != model.pad_token
        pred = outputs[active_loss_mask]
        corr = batch[active_loss_mask]
        loss = criterion(pred, corr)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / (i + 1)


def train(model_name: str, vocab_name: str, **kwargs):
    """
    Args:
        model_name - str, name of file to save model in.
        vocab_name - str, name of file to save vocab in.
    kwargs:
        charemb_dim - int
        hidden_dim - int
        num_layers - int
        batch_size - int
        epochs - int
        max_length - int
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    charemb_dim = kwargs["charemb_dim"]
    hidden_dim = kwargs["hidden_dim"]
    num_layers = kwargs["num_layers"]
    batch_size = kwargs["batch_size"]
    epochs = kwargs["epochs"]
    max_length = kwargs["max_length"]
    dataset = GeneratorDataset("data/input.txt", max_length, 5, batch_size)
    model = LSTMGen(len(dataset.vocab), charemb_dim, hidden_dim, num_layers, dataset.vocab.pad_idx).to(device)
    print(model)
    optimizer = optim.Adam(model.parameters())
    if TENSORBOARD:
        writer = SummaryWriter()
    dataset.vocab.save_dict(f"{vocab_name}.pickle")
    for epoch in range(epochs):
        print("=" * 50, f"Epoch {epoch}", "=" * 50)
        loss = train_epoch(model, optimizer, dataset)
        print(f"Train loss {loss}")
        if TENSORBOARD:
            writer.add_scalar('total_loss', loss, epoch)
        model.save_model(f"{model_name}.pt")
    print("Finished training")
