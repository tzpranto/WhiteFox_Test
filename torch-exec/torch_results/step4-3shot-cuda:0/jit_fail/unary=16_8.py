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
        self.l1 = torch.nn.Linear(64, 10)

    def forward(self, x):
        y = self.l1(x)
        y = torch.sigmoid(y)
        return y


func = Model().to('cuda:0')


x = torch.randn(2, 64)

label = torch.randint(10, [2])

test_inputs = [x, label]

# JIT_FAIL
'''
direct:
forward() takes 2 positional arguments but 3 were given

jit:
forward() takes 2 positional arguments but 3 were given
'''