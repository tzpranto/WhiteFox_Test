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

class Attention(torch.nn.Module):

    def __init__(self, hidden_dim, n_heads, n_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_layers = n_layers

    def init_multihead(self):
        self.heads = nn.ModuleList([nn.MultiheadAttention(self.hidden_dim, self.n_heads) for _ in range(self.n_layers)])

    def forward(self, x1, x2):
        for layer in self.heads:
            (x1, x2) = layer(x1, x2)
        return (x1, x2)


hidden_dim = 512
n_heads = 8
n_layers = 6
func = Attention(hidden_dim, n_heads, n_layers).to('cpu')


hidden_dim = 512
x1 = torch.rand(1, 100, hidden_dim)

hidden_dim = 512
x2 = torch.rand(1, 100, hidden_dim)

test_inputs = [x1, x2]

# JIT_FAIL
'''
direct:
'Attention' object has no attribute 'heads'

jit:
'Attention' object has no attribute 'heads'
'''