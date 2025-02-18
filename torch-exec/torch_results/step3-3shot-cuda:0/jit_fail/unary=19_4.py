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
        self.linear = torch.nn.Linear(2, 3)

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.sigmoid(v1)
        return v2


func = Model().to('cuda:0')


x1_1 = torch.randn(1, 2)

x1_2 = torch.randn(1, 2)
x1_2 = torch.randn(1, 2)
x1_1 = torch.randn(1, 2)

x1 = torch.cat((x1_1, x1_2), dim=1)

test_inputs = [x1_1, x1_2, x1]

# JIT_FAIL
'''
direct:
forward() takes 2 positional arguments but 4 were given

jit:
forward() takes 2 positional arguments but 4 were given
'''