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
        self.conv = torch.nn.Conv2d(8, 1, 3, stride=1, padding=1)

    def forward(self, x):
        v1 = self.conv(x)
        return v1.relu(other=0.25)


func = Model().to('cpu')


x = torch.randn(1, 8, 122, 122)

test_inputs = [x]

# JIT_FAIL
'''
direct:
Tensor.relu() takes no keyword arguments

jit:
Tensor.relu() takes no keyword arguments
'''