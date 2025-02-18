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

class MultiHeadAttention(torch.nn.Module):

    def __init__(self, dropout_p):
        super().__init__()
        self.dropout_p = dropout_p

    def forward(self, query, key, value, attn_mask=None):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        if attn_mask is not None:
            qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, self.dropout, True)
        output = attn_weight @ value
        return output


dropout_p = 1
func = MultiHeadAttention(0.2).to('cpu')


x1 = torch.randn(1, 32, 1024, 1024)

x2 = torch.randn(1, 32, 1024, 1024)

x3 = torch.randn(1, 32, 1024, 1024)
query = 1

test_inputs = [x1, x2, x3, query]

# JIT_FAIL
'''
direct:
'MultiHeadAttention' object has no attribute 'dropout'

jit:
'MultiHeadAttention' object has no attribute 'dropout'
'''