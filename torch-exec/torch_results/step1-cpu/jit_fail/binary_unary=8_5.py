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
        self.conv1 = torch.nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, stride=1, padding=1)

    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.relu(v1, inplace=True)
        v3 = self.conv2(v2)
        v4 = torch.relu(v3, inplace=True)
        return v4


func = Model().to('cpu')


x = torch.randn(1, 3, 64, 64)

test_inputs = [x]

# JIT_FAIL
'''
direct:
relu() got an unexpected keyword argument 'inplace'

jit:
relu() got an unexpected keyword argument 'inplace'
'''