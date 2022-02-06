import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.Q = nn.Linear(d_model, d_model)
        self.K = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask):
        batch_size, seq_len, _ = x.shape
        query = self.Q(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        key = self.K(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        values = self.V(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        scores = torch.matmul(
            query.permute(0, 2, 1, 3),
            key.permute(0, 2, 3, 1)
        ) / (self.d_k ** 0.5)
        scores.masked_fill_(~mask, -1e9)
        # scores - batch, n_heads, seq_len, seq_len
        scores = torch.softmax(scores, 3)
        # attention - batch, n_heads, seq_len, d_k
        attention  = torch.matmul(
            scores, 
            values.permute(0, 2, 1, 3)
            )
        attention = attention\
            .permute(0, 2, 1, 3)\
            .reshape(batch_size, seq_len, self.d_model)
        return attention

class PositionalEmbedding(nn.Module):
    "Stolen from annotated transformer"
    def __init__(self, d_model, dropout, max_len=500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(torch.log(torch.Tensor([10000.0])) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x - batch, seq_len, d_model
        return self.dropout(x + self.pe[:, x.size(1)])


class TransformerLayer(nn.Module):
    """
    Not really vanilla, as we use PRE-LN,
    instead of POST-LN
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model))
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, x, mask):
        x = x + self.mha(self.ln1(x), mask)
        x = x + self.ffn(self.ln2(x))
        return x


class SwitchTransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_experts: int):
        super().__init__()
        pass
