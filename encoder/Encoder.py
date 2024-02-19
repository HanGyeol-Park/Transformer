import torch.nn as nn
from encoder.MultiheadAttention import MultiheadAttention
from encoder.FeedForward import FeedForwardLayer
from encoder.PositionEncoding import PositionEncoding

class Encoder(nn.Module):
    def __init__(self, hidden_dim, num_head):
        super(Encoder, self).__init__()
        self.multihead_attention = MultiheadAttention(hidden_dim, num_head)
        self.feedforward_layer = FeedForwardLayer(hidden_dim)
        self.pos_Encoding = PositionEncoding(hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.pos_Encoding(x)
        x = self.norm1(x + self.multihead_attention(x))
        x = self.norm2(x + self.feedforward_layer(x))
        return x