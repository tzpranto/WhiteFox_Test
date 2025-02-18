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
        self.query = torch.nn.Parameter(torch.randn(32, 32, 64))
        self.key = torch.nn.Parameter(torch.randn(32, 32, 64))
        self.value = torch.nn.Parameter(torch.randn(32, 32, 64))
        self.atten_mask = torch.randn(1, 1, 1, 32)

    def forward(self, x1):
        kq = self.query @ self.key.transpose(-2, -1)
        kq = kq / math.sqrt(self.query.size(-1))
        kq = kq + self.atten_mask
        atten_w = torch.softmax(kq, dim=-1)
        output = atten_w @ self.value
        return output


func = Model().to('cuda:0')


x1 = torch.randn(1, 1, 32)

x2 = torch.randn(1, 1, 32)

x3 = torch.randn(1, 1, 32)

test_inputs = [x1, x2, x3]

# JIT_FAIL
'''
direct:
forward() takes 2 positional arguments but 4 were given

jit:
forward() takes 2 positional arguments but 4 were given
'''