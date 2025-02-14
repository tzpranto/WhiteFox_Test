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
        self.linear1 = torch.nn.Linear(2, 3)

    def forward(self, x):
        q = self.linear1(x1)
        k = self.linear1(x2)
        v = self.linear1(x3)
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)
        scale_factor = torch.matmul(q, k.transpose(-2, -1))
        scale_factor = torch.softmax(scale_factor / math.sqrt(min(len(q[0]), len(k[0]))), dim=-1)
        drop_res = scale_factor * v
        return drop_res


func = Model().to('cpu')


x1 = torch.randn(1, 2)

x2 = torch.randn(1, 2)

x3 = torch.randn(1, 3)

test_inputs = [x1, x2, x3]

# JIT_FAIL
'''
direct:
forward() takes 2 positional arguments but 4 were given

jit:
forward() takes 2 positional arguments but 4 were given
'''