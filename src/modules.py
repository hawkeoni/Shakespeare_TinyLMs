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


class SwitchFFN(nn.Module):
    """
    Capacity is actually something which would really matter in
    model parallel setup on a big model, but as this either
    a single gpu or not sharded DDP we can actually not care
    about it.
    """

    def __init__(self, d_model: int, n_experts: int, capacity_factor: float, drop_tokens: bool):
        super().__init__()
        self.d_model = d_model
        self.capacity_factor = capacity_factor
        self.drop_tokens = drop_tokens
        self.gate = nn.Linear(d_model, n_experts)
        self.experts = nn.ModuleList(nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model))
            for _ in range(n_experts)
        )
        self.n_experts = n_experts
    

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x = x.view(-1, x.size(2))
        tokens_per_batch = x.size(0)
        capacity = tokens_per_batch / self.n_experts * self.capacity_factor
        # actually ignore capacity and do not drop tokens
        # batch * seq_len, d_model or tokens, d_model
        route_probs = torch.softmax(self.gate(x), dim=1)
        # batch * seq_len, n_experts or tokens, n_experts
        # here we get best expert probs and best expert ind
        route_max_val, route_max_ind = torch.max(route_probs, 1)
        outputs = torch.zeros_like(x)
        expert_indices = []
        num_tokens = x.size(0)
        fi = outputs.new_zeros((num_tokens, ), dtype=torch.long)
        # route_probs - num_tokens, num_experts
        pi = route_probs.mean(dim=1)

        for expert_num in range(self.n_experts):
            expert_ind = (route_max_ind == expert_num).nonzero(as_tuple=True)[0]
            fi[expert_num] = len(expert_ind)
            expert_indices.append(expert_ind)
            expert_input = x[expert_ind]
            expert_output = self.experts[expert_num](expert_input)
            outputs[expert_ind] = expert_output
        
        fi = fi / num_tokens
        load_balancing_loss = 0.01 * self.n_experts * torch.dot(fi, pi)
        
        outputs = outputs * route_max_val.unsqueeze(1)
        outputs = outputs.view(batch_size, seq_len, self.d_model)

        return outputs, load_balancing_loss


class SwitchTransformerLayer(nn.Module):

    def __init__(self, d_model: int, n_heads: int, n_experts: int):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.switch_ffn = SwitchFFN(d_model, n_experts, 1.0, False)

    def forward(self, x, mask):
        x = x + self.mha(self.ln1(x), mask)
        y, lbl = self.switch_ffn(self.ln2(x), mask)
        x = x + y 
        return x, lbl
