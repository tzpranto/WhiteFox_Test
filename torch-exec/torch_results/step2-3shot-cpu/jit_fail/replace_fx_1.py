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
        self.x1 = torch.nn.Parameter(torch.randn(10))

    def forward(self, x2):
        x1 = self.x1
        x1 = x1 + x2
        x2 = torch.nn.functional.dropout(x1, 0.8)
        x1 = torch.nn.functional.dropout(x2, 0.7)
        return (x1, x2)



func = Model().to('cpu')


x1 = torch.randn(1, 10)

x2 = torch.randn(1, 10)

test_inputs = [x1, x2]

# JIT_FAIL
'''
direct:
forward() takes 2 positional arguments but 3 were given

jit:
forward() takes 2 positional arguments but 3 were given
'''