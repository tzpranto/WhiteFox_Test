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

    def __init__(self, dim, n_heads=16, n_layers=1):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.self_attentions = [torch.nn.MultiheadAttention(dim, n_heads) for i in range(n_layers)]
        self.pos_ffns = [torch.nn.Sequential(torch.nn.Linear(dim, 4 * dim), torch.nn.ReLU(), torch.nn.Linear(4 * dim, dim)) for i in range(n_layers)]
        self.norms_1 = [torch.nn.LayerNorm(dim) for i in range(n_layers)]

    def forward(self, x):
        v = x.view(1, 1, -1, self.dim)
        for ((self_attention, norm), pos_ffn) in zip(self.self_attentions, self.pos_ffns):
            v = self_attention(v, v, v, need_weights=False)[0]
            v = v + x
            v = pos_ffn(v)
            v = norm(v + x)
        return v


dim = 1
func = Model(dim).to('cpu')

x = 1

test_inputs = [x]
