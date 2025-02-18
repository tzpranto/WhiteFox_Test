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
        self.layer = torch.nn.Linear(64, 100)

    def forward(self, x1):
        y = self.layer(x1)
        z = y > 0
        w = y * self.negative_slope
        x = torch.where(z, y, w)
        return x


func = Model().to('cuda:0')


x1 = torch.randn(1, 64)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
'Model' object has no attribute 'negative_slope'

jit:
'Model' object has no attribute 'negative_slope'
'''