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
        self.linear = torch.nn.Linear(7, 3)

    def forward(self, x1):
        h1 = self.linear(x1)
        v1 = h1 + h2
        return v1


func = Model().to('cuda:0')


h2 = torch.randn(1, 3)

x1 = torch.randn(1, 7)

test_inputs = [h2, x1]

# JIT_FAIL
'''
direct:
forward() takes 2 positional arguments but 3 were given

jit:
forward() takes 2 positional arguments but 3 were given
'''