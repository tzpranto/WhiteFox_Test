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

    def __init__(self, other):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 16, False)
        self.linear2 = torch.nn.Linear(16, other, False)

    def forward(self, input):
        return self.linear2(self.linear1(input)) + self.linear1(input)


other = 1
func = Model(150).to('cuda:0')


x1 = torch.randn(1, 10)

x2 = torch.randn(1, 150)

test_inputs = [x1, x2]

# JIT_FAIL
'''
direct:
forward() takes 2 positional arguments but 3 were given

jit:
forward() takes 2 positional arguments but 3 were given
'''