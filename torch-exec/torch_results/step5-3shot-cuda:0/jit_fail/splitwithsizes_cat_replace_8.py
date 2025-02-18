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

    def forward(self, x1, split_sizes):
        (v1, v2, v3, v4, v5, v6) = torch.split(x1, split_sizes, dim=1)
        v7 = v1 * 0.5
        v8 = v2 * 0.7071067811865476
        v9 = torch.erf(v8)
        v10 = v9 + 1
        v11 = v3 * v10
        v12 = v4 * v10
        v13 = v5 * v10
        v14 = v6 * v10
        v15 = torch.cat([v7, v11, v12, v13, v14], dim=1)
        return v15


func = Model().to('cuda:0')


x1 = torch.randn(1, 15, 8, 8, 8)
split_sizes = 1

test_inputs = [x1, split_sizes]

# JIT_FAIL
'''
direct:
too many values to unpack (expected 6)

jit:
too many values to unpack (expected 6)
'''