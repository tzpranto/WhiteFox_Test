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
        (t1, t2, t3) = torch.split(x1, 1, 2)
        x2 = torch.cat([t1, t2, t3], 2)
        return x2


func = Model().to('cuda:0')


x1 = torch.randn(1, 1000, 2)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
not enough values to unpack (expected 3, got 2)

jit:
not enough values to unpack (expected 3, got 2)
'''