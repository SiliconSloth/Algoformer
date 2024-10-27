import torch
from torch import nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, output_offset: int = 0, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        position[output_offset:] += output_offset
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)].unsqueeze(0)
        return self.dropout(x)


class Model(nn.Module):
    def __init__(self, vocab_size, output_offset=0):
        super().__init__()

        self.embedding = nn.Linear(vocab_size, 100)
        self.pos_encoder = PositionalEncoding(d_model=100, output_offset=output_offset)
        # Decoder-only transformers only have an encoder layer
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=100, nhead=5, batch_first=True),
            num_layers=2)
        self.decoder = nn.Linear(100, vocab_size)
    
    def forward(self, x, mask):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.encoder(x, mask)
        x = self.decoder(x)
        x = torch.log_softmax(x, dim=-1)

        return x