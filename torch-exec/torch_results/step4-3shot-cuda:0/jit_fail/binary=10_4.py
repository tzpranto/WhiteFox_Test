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
        self.linear = torch.nn.Linear(3, 5)

    def forward(self, x2):
        v2 = self.linear(x2)
        return v2 + x2


func = Model().to('cuda:0')


x2 = torch.randn(2, 3)

other = torch.ones(3)

test_inputs = [x2, other]

# JIT_FAIL
'''
direct:
forward() takes 2 positional arguments but 3 were given

jit:
forward() takes 2 positional arguments but 3 were given
'''