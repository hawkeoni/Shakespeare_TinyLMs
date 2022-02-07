import torch
import torch.nn as nn
from pytorch_lightning.core.lightning import LightningModule

from src.models import SwitchOutput


class LMSystem(LightningModule):

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)
    
    def _universal_step(self, batch, batch_idx):
        preds = self(batch)
        loss = torch.nn.functional.cross_entropy(
            preds.output[:, :-1].reshape(-1, preds.size(-1)), 
            batch[:, 1:].reshape(-1), 
            ignore_index=0)
        if isinstance(preds, SwitchOutput):
            loss += preds.lbl
            self.log("lbl", preds.lbl, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("perplexity", torch.exp(loss), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        return self._universal_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._universal_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self._universal_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def generate(self, x: torch.Tensor, temperature: float = 1.0, length: int = 100) -> torch.Tensor:
        raise NotImplementedError("This is for the future")
        assert temperature > 0, f"Temperature set to {temperature}, must be positive."
        # x - 1, seq_len
        outputs, (hidden, context) = self(x)
        results = outputs.new_zeros((outputs.size(0), length), dtype=torch.long)
        # [:, -1] - batch_size, vocab_size
        # outputs - batch, seq_len, vocab_size
        outputs = outputs / temperature
        outputs = torch.softmax(outputs, dim=2)
        inputs = outputs[:, -1].multinomial(1) # 1, 1
        results[:, 0] = inputs[:, 0]
        for i in range(1, length):
            outputs, (hidden, context) = self.rnn(self.embedding(inputs), (hidden, context))
            outputs = self.out(outputs) / temperature
            outputs = torch.softmax(outputs, dim=2)
            inputs = outputs[:, -1].multinomial(1)
            # inputs = torch.argmax(outputs[:, -1], dim=1).unsqueeze(1)  # 1, 1
            results[:, i] = inputs[:, 0]
        return results