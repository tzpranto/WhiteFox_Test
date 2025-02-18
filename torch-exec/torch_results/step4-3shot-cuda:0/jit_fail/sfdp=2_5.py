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

class MyModel(nn.Module):

    def __init__(self, query, key, value):
        super().__init__()
        self.scale_factor = math.sqrt(key.size(-1))
        self.dropout_p = 0.1
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(self.dropout_p)
        self.matmul = torch.matmul

    def forward(self, query):
        qk = self.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(self.scale_factor)
        softmax_qk = self.softmax(scaled_qk)
        dropout = self.dropout(self.softmax_qk)
        ouput = self.matmul(dropout, value)
        return output


query = torch.randn(3, 2, 5)
key = torch.randn(3, 5, 6)
value = torch.randn(3, 6, 7)
func = MyModel(query, key, value).to('cuda:0')


query = torch.randn(3, 2, 5)

key = torch.randn(3, 5, 6)

x1 = torch.rand(3, 2, 5)

test_inputs = [query, key, x1]

# JIT_FAIL
'''
direct:
forward() takes 2 positional arguments but 4 were given

jit:
forward() takes 2 positional arguments but 4 were given
'''