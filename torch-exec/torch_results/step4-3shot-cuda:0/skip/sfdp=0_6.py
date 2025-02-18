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

    def __init__(self, num_heads=8, hidden_key_size=8, hidden_value_size=8):
        super(Model, self).__init__()
        self.attn = MultiHeadAttention(num_heads, hidden_key_size, hidden_value_size)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        return self.attn(query, key, value, attn_mask, key_padding_mask)


func = Model().to('cuda:0')


x1 = torch.rand(1, 64, 32)

x2 = torch.rand(1, 64, 48)

x3 = torch.rand(1, 64, 32)
query = 1
key = 1

test_inputs = [x1, x2, x3, query, key]
