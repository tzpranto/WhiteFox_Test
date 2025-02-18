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
        self.m0 = torch.nn.Linear(10, 20)
        self.m1 = torch.nn.Linear(20, 10)
        self.b1 = torch.nn.Linear(10, 1)

    def forward(self, x1):
        v0 = torch.relu(self.m0(x1))
        v1 = self.m1(v0)
        v1 = v1 + self.b1(x1)
        return v1


func = Model().to('cpu')

x1 = 1

test_inputs = [x1]

# JIT_FAIL
'''
direct:
linear(): argument 'input' (position 1) must be Tensor, not int

jit:
linear(): argument 'input' (position 1) must be Tensor, not int
'''