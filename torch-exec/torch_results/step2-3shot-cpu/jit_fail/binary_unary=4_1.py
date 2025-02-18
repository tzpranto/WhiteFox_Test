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

class Model(nn.Module):

    def __init(self, tensor):
        super().__init__()
        self.linear = nn.Linear(16, 32, bias=True)
        self.tensor = tensor

    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 + self.tensor
        v3 = F.relu(v2)
        return v3



func = Model().to('cpu')


t = torch.zeros(32)

x = torch.randn(1, 16)

test_inputs = [t, x]

# JIT_FAIL
'''
direct:
forward() takes 2 positional arguments but 3 were given

jit:
forward() takes 2 positional arguments but 3 were given
'''