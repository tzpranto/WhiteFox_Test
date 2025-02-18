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

    def __init__(self, num_layers):
        super().__init__()
        self.multihead_attn = torch.nn.MultiheadAttention(d_model, num_heads)

    def forward(self, x1, x2, x3, x4, x5):
        v1 = self.multihead_attn(x1, x2, x3)
        v2 = torch.sigmoid(v1)
        v3 = v4 + v5
        return v6


num_layers = 1

func = Model(num_layers).to('cuda:0')


x1 = torch.randint(10, (1, 4, 1024))
x1 = torch.randint(10, (1, 4, 1024))
x2 = x1.clone()
x3 = (x1 - x2) % 13


x4 = torch.triu(torch.tril(x3.transpose(0, 1).transpose(1, 2), diagonal=10).transpose(0, 1).transpose(1, 2), diagonal=-10)
x3 = 1
x5 = 1

test_inputs = [x1, x2, x4, x3, x5]
