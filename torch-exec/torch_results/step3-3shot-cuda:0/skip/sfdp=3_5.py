import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import math
import torch as th
import torch.linalg as la
from torch.nn import Parameter
import torch.linalg as linalg

class Model(torch.nn.Module):

    def __init__(self, d_embed: int=128, seq_len: int=3, num_heads: int=4, dropout_p: float=0.2):
        super().__init__()
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.projection = Projection(num_heads, d_model=d_model)
        self.attention = MultiHeadAttention(d_model, num_heads, dropout_p)
        self.fc_projection = Projection(d_model, d_model=d_model)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor=None):
        projected_q = self.projection(q)
        projected_k = self.projection(k)
        projected_v = self.projection(v)
        output = self.attention(projected_q, projected_k, projected_v, mask)
        output = self.fc_projection(output)
        return output


func = Model().to('cuda:0')


q = torch.randn(1, 3, 64)

k = torch.randn(1, 4, 64)

v = torch.randn(1, 4, 64)

test_inputs = [q, k, v]
