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
        self.linear = torch.nn.Linear(20, 30)

    def forward(self, x):
        v1 = self.linear(x)
        y = v1 + y
        return y


func = Model().to('cuda:0')


__input__ = torch.randn(1, 20)

test_inputs = [__input__]

# JIT_FAIL
'''
direct:
local variable 'y' referenced before assignment

jit:
local variable 'y' referenced before assignment
'''