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

    def forward(self, x1, x2):
        v1 = x1.permute(0, 2, 1)
        v2 = x2.permute(0, 2, 1)
        v3 = torch.bmm(v1, v3)
        return v3



func = Model().to('cuda:0')


x1 = torch.randn(1, 2, 2)

x2 = torch.randn(1, 2, 2)

test_inputs = [x1, x2]

# JIT_FAIL
'''
direct:
local variable 'v3' referenced before assignment

jit:
local variable 'v3' referenced before assignment
'''