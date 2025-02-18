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
        self.linear = torch.nn.Linear(5, 5)

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + torch.tensor([1, 2, 3, 4, 5], dtype=torch.float)
        return v2


func = Model().to('cuda:0')


x1 = torch.randn(1, 5)


a = torch.tensor([10, 100, 1000, 10000, 100000], dtype=torch.float)

test_inputs = [x1, a]

# JIT_FAIL
'''
direct:
forward() takes 2 positional arguments but 3 were given

jit:
forward() takes 2 positional arguments but 3 were given
'''