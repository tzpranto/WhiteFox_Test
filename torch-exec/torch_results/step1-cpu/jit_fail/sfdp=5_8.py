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

    def __init__(self, dropout_p):
        super().__init__()
        self.query_net = ...
        self.key_net = ...
        self.attn_mask = ...
        self.dropout = torch.nn.Dropout(p=dropout_p, inplace=True)

    def forward(self, query_input, key_input, value_input):
        query = self.query_net(query_input)
        key = self.key_net(key_input)
        attn_scores = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1)) + self.attn_mask
        attn_weights = self.dropout(torch.nn.Softmax(dim=-1)(attn_scores))
        output = self.attn_mask @ value_input * attn_weights
        return output


dropout_p = 1
func = Model(0.2).to('cpu')


query_input = torch.randn(1, 2, 8)

key_input = torch.randn(1, 4, 8)

value_input = torch.randn(1, 4, 8)

test_inputs = [query_input, key_input, value_input]

# JIT_FAIL
'''
direct:
'ellipsis' object is not callable

jit:
'ellipsis' object is not callable
'''