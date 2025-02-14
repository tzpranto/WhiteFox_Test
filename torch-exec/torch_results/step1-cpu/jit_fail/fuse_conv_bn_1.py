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
        self.conv = torch.nn.Conv2d(3, 3, 3, bias=True)
        self.batch_norm = torch.nn.BatchNorm2d(3, affine=False)

    def forward(self, x1):
        v1 = F.relu(self.conv(x1))
        v2 = F.batch_norm(x1, None, None, None, None, False, 0, self.conv.weight, self.conv.bias)
        return v1


func = Model().to('cpu')


x1 = torch.randn(1, 3, 32, 32)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
batch_norm() takes from 3 to 8 positional arguments but 9 were given

jit:
batch_norm() takes from 3 to 8 positional arguments but 9 were given
'''