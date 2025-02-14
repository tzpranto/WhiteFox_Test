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
        self.linear = torch.nn.Linear(3, 8)

    def forward(self, x):
        return torch.relu(x + self.linear.default(x))


func = Model().to('cpu')


x = torch.randn(1, 3)

test_inputs = [x]

# JIT_FAIL
'''
direct:
'Linear' object has no attribute 'default'

jit:
'Linear' object has no attribute 'default'
'''