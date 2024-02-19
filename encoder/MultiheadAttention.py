import torch.nn as nn
import torch.nn.functional as F
import torch

class MultiheadAttention(nn.Module):
    def __init__(self, hidden_dim, num_head):
        super(MultiheadAttention, self).__init__()
        self.num_head = num_head
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_head

        self.attfck = nn.Linear(hidden_dim, hidden_dim)
        self.attfcq = nn.Linear(hidden_dim, hidden_dim)
        self.attfcv = nn.Linear(hidden_dim, hidden_dim)
        self.attfco = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        Q = self.attfck(x)
        K = self.attfcq(x)
        V = self.attfcv(x)

        Q = Q.view(-1, self.num_head, self.head_dim)
        K = K.view(-1, self.head_dim, self.num_head)
        V = V.view(-1, self.num_head, self.head_dim)

        Scaled_dot_product = torch.matmul(Q, K) / np.sqrt(self.head_dim)
        Scaled_dot_product = F.softmax(Scaled_dot_product, dim=-1)
        Scaled_dot_product = torch.matmul(self.dropout(Scaled_dot_product), V)
        
        Scaled_dot_product = Scaled_dot_product.view(-1, self.num_head * self.head_dim)
        out = self.attfco(Scaled_dot_product)

        return out

