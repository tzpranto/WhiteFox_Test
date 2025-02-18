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

    def __init__(self, hidden_size):
        super().__init__()
        self.dot_product_attention = DotProductAttention(dropout=0.3, is_cross_attention=False)
        self.out = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, query, key, value, training=False):
        x = self.dot_product_attention(query=query, key=key, value=value, training=training)
        output = self.out(x)
        return output


hidden_size = 1

func = Model(hidden_size).to('cuda:0')


query = torch.randn(1, 128, 576)

key = torch.randn(1, 128, 576)

value = torch.randn(1, 128, 576)

test_inputs = [query, key, value]
