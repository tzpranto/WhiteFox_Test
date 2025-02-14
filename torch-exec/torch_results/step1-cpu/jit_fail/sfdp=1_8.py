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

    def __init__(self, nheads, dim_head, dim, dropout_p=0.5):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout_p)
        self.m = torch.nn.Linear(dim, dim, bias=False)
        self.q = torch.nn.Linear(dim, dim, bias=False)
        self.k = torch.nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.m(x)
        attn = (q @ k.transpose(-2, -1)).div(self.dim ** (-0.5))
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        x = attn @ v
        return x


nheads = 1
dim_head = 1
dim = 1
func = Model(32, 16, 768, 0.5).to('cpu')


x = torch.randn(2, 768, requires_grad=True)

test_inputs = [x]

# JIT_FAIL
'''
direct:
'Model' object has no attribute 'dim'

jit:
'Model' object has no attribute 'dim'
'''