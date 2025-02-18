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

    def __init__(self, query_channels, key_channels, n_units, n_heads, dropout):
        super().__init__()
        self.attn = AttentionCore(query_channels, key_channels, n_units, n_heads, dropout)

    def forward(self, query, key, value, scale_factor=1, dropout_p=0.2):
        v = self.attn(query, key, value, scale_factor, dropout_p)
        return v


query_channels = 1
key_channels = 1
n_units = 1
n_heads = 1
dropout = 1

func = Model(query_channels, key_channels, n_units, n_heads, dropout).to('cuda:0')


query = torch.randn(1, 32, 5, 32)

key = torch.randn(1, 32, 4, 32)

value = torch.randn(1, 32, 5, 32)

test_inputs = [query, key, value]
