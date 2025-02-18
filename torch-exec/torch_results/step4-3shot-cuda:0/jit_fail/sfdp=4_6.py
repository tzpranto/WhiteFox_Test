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

    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=-2)

    def forward(self, q, k, v, a):
        d = q.size(-2)
        scale = torch.sqrt(k.size(-1))
        qk = q @ k.transpose(-2, -1) / scale
        qk = qk + a
        attn_w = self.softmax(qk)
        output = attn_w @ v
        return (output, attn_w.sum())


func = Model().to('cuda:0')


q = torch.randn(5, 3, 64)

k = torch.randn(6, 5, 64)

v = torch.randn(7, 6, 20)

a = torch.randn(6, 6)

test_inputs = [q, k, v, a]

# JIT_FAIL
'''
direct:
sqrt(): argument 'input' (position 1) must be Tensor, not int

jit:
sqrt(): argument 'input' (position 1) must be Tensor, not int
'''