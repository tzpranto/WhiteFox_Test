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

    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = float(dropout)

    def forward(self, query, key, value):
        output = torch.softmax(query @ key.transpose(-2, -1) / math.sqrt(query.size(-1)), dim=-1)
        output = torch.dropout(output, self.dropout, training=self.training)
        output = output @ value
        return output


func = Model(dropout=0.1).to('cuda:0')


query = torch.randn(4, 16, 10)

key = torch.randn(4, 32, 10)

value = torch.randn(4, 32, 20)

test_inputs = [query, key, value]

# JIT_FAIL
'''
direct:
dropout() missing 1 required positional arguments: "train"

jit:
dropout() missing 1 required positional arguments: "train"
'''