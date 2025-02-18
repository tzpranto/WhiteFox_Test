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

    def forward(x1, x2, x3, x4):
        v1 = torch.matmul(x2, x4.transpose(-2, -1))
        v2 = v1 * x3
        v3 = torch.nn.functional.softmax(v2, dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=x1)
        v5 = torch.matmul(v4, x4)
        return v5


func = Model().to('cpu')


x1 = torch.rand([])

x2 = torch.randn(1, 20, 30)

x3 = torch.rand([])

x4 = torch.randn(1, 30, 20)

test_inputs = [x1, x2, x3, x4]

# JIT_FAIL
'''
direct:
forward() takes 4 positional arguments but 5 were given

jit:
forward() takes 4 positional arguments but 5 were given
'''