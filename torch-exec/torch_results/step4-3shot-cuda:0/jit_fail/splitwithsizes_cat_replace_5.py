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

    def forward(self, x1):
        (s1, s2) = torch.split(x1, 8, dim=2)
        c = torch.cat([s1, s2], dim=2)
        return True if c.equal(x1) else False


func = Model().to('cuda:0')


x1 = torch.randn(1, 8, 128, 32)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
too many values to unpack (expected 2)

jit:
too many values to unpack (expected 2)
'''