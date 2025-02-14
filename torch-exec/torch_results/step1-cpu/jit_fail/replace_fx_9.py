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
        self.linear = torch.nn.Linear(2, 2)
        self.dropout = torch.nn.Dropout(p=0.4, inplace=False)

    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = self.dropout(v1)
        return v2


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