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
        self.f1 = torch.nn.Linear(15, 8)
        self.f2 = torch.nn.Linear(10, 12)

    def forward(self, input0, input1):
        v1 = torch.mul(self.f1(input0), torch.clamp_max(torch.clamp_min(self.f2(input1) + 3, min=0), max=6))
        v2 = v1 / 6
        return v2


func = Model().to('cpu')

input0 = 1
input1 = 1

test_inputs = [input0, input1]

# JIT_FAIL
'''
direct:
linear(): argument 'input' (position 1) must be Tensor, not int

jit:
linear(): argument 'input' (position 1) must be Tensor, not int
'''