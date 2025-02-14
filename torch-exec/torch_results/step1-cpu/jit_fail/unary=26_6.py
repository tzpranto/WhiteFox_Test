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
        self.conv = torch.nn.ConvTranspose2d(3, 8, 3, stride=1, padding=1)

    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.gt(v1, 0)
        v3 = v2
        v4 = torch.mul(v1, negative_slope=0.1)
        v5 = torch.where(v3, v1, v4)
        return v5


func = Model().to('cpu')


x = torch.randn(1, 3, 64, 64)

test_inputs = [x]

# JIT_FAIL
'''
direct:
mul() missing 1 required positional arguments: "other"

jit:
mul() missing 1 required positional arguments: "other"
'''