import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.model import LSTMGen
from src.data import GeneratorDataset


def train(model: nn.Module, optimizer: optim.Optimizer, dataset: GeneratorDataset):
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


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = GeneratorDataset("data/input.txt", 50, 5, 64)
    model = LSTMGen(len(dataset.vocab), 64, 128, 1, dataset.vocab.pad_idx).to(device)
    optimizer = optim.Adam(model.parameters())
    writer = SummaryWriter()
    for epoch in range(50):
        print("=" * 50, f"Epoch {epoch}", "=" * 50)
        loss = train(model, optimizer, dataset)
        print(f"Train loss {loss}")
        writer.add_scalar('total_loss', loss, epoch)
        torch.save(model.state_dict(), 'model.pt')
