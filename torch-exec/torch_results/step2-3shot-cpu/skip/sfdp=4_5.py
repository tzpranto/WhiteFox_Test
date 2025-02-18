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

    def __init__(self, config):
        super().__init__()
        self.q_weight = torch.nn.Parameter(torch.Tensor(config.hidden_size, config.hidden_size))
        self.k_weight = torch.nn.Parameter(torch.Tensor(config.hidden_size, config.hidden_size))

    def forward(self, query, key, value, attention_mask):
        n = torch.matmul(query, self.q_weight)
        n = torch.matmul(key, self.k_weight)
        n = n / math.sqrt(self.q_weight.size(0))
        n = n + attention_mask
        n = torch.softmax(n, dim=-1)
        n = torch.matmul(n, value)
        return n


config = TransformerConfig(32)
func = Model(config).to('cpu')


query = torch.randn(2, 32, 16)

key = torch.randn(2, 32, 16)

value = torch.randn(2, 32, 16)

attention_mask = torch.randn(2, 1, 1, 16)

test_inputs = [query, key, value, attention_mask]
