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
        super(Model, self).__init__()
        self.mlp = torch.nn.Sequential(torch.nn.Linear(3, 10), torch.nn.ReLU(), torch.nn.Linear(10, 10), torch.nn.ReLU(), torch.nn.Linear(10, 10))

    def forward(self, x1):
        v1 = self.mlp(x1)
        return v1


func = Model().to('cpu')


x1 = torch.randn(1, 3)

x2 = torch.randn(1, 10)

x3 = torch.randn(6, 1)

test_inputs = [x1, x2, x3]

# JIT_FAIL
'''
direct:
forward() takes 2 positional arguments but 4 were given

jit:
forward() takes 2 positional arguments but 4 were given
'''