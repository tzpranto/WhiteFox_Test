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

    def __init__(self, conv_op):
        super().__init__()
        self.conv = conv_op

    def forward(self, x1, x2):
        v1 = self.conv(x1)
        v2 = v1 + x2
        return v2


conv_op = 1

func = Model(conv_op).to('cpu')


x1 = torch.randn(1, 3, 64, 64)

x2 = torch.randn(1, 8, 64, 64)

test_inputs = [x1, x2]

# JIT_FAIL
'''
direct:
'int' object is not callable

jit:
'int' object is not callable
'''