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

class M(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(6, 8)
        self.linear2 = torch.nn.Linear(8, 10)

    def forward(self, x):
        h = torch.tanh(self.linear1(x))
        y = torch.flatten(self.linear2(h), 1)
        return y



func = M().to('cuda:0')

x = 1

test_inputs = [x]

# JIT_FAIL
'''
direct:
linear(): argument 'input' (position 1) must be Tensor, not int

jit:
linear(): argument 'input' (position 1) must be Tensor, not int
'''