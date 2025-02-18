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

    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = self.model_dim // self.num_heads

    def forward(self, query, key, value, mask=None):
        scaled_dot_product = torch.matmul(query, key.transpose(-2, -1)) / self.head_dim ** 0.5
        if mask is not None:
            scaled_dot_product.masked_fill_(mask, -float('inf'))
        attention_weights = scaled_dot_product.softmax(dim=-1)
        output = attention_weights.matmul(value)
        return output


num_heads = 32
func = Model(num_heads).to('cpu')

query = 1
key = 1
value = 1

test_inputs = [query, key, value]
