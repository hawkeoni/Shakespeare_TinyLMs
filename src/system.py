import torch
import torch.nn as nn
from pytorch_lightning.core.lightning import LightningModule


from src.models import SwitchTransformer, VanillaTransformer, LSTMLM


class LMSystem(LightningModule):

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)
    
    def _universal_step(self, batch, batch_idx):
        preds = self(batch)
        output = preds["output"]
        loss = torch.nn.functional.cross_entropy(
            output[:, :-1].reshape(-1, output.size(-1)), 
            batch[:, 1:].reshape(-1), 
            ignore_index=0)
        # switch load balancing loss
        loss += preds.get("lbl", 0)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._universal_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self._universal_step(batch, batch_idx)
        self.log("validation_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        loss = self._universal_step(batch, batch_idx)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def configure_optimizers(self):
        lr = 0.001
        if isinstance(self.net, (VanillaTransformer, SwitchTransformer)):
            lr /= 10
        return torch.optim.Adam(self.parameters(), lr=lr)

    def generate(self, x: torch.Tensor, temperature: float = 1.0, length: int = 100) -> torch.Tensor:
        batch_size = x.size(0)
        generated_ind = x.new_zeros((batch_size, length))
        input_tensor = x
        for generated_position in range(length):
            # input_tensor - batch, seq_len
            output = self(input_tensor)["output"]  # batch, seq_len, vocab_size
            output = output[:, -1]  # batch, vocab_size
            output = torch.softmax(output / temperature, dim=1)
            output = output.multinomial(1)  # batch, 1
            generated_ind[:, generated_position] = output[:, 0]
            input_tensor = torch.cat((input_tensor, output), dim=1)
        return generated_ind
