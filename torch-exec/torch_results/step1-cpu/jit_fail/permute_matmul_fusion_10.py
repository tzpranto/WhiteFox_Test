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

    def forward(self, x1, x2, x3):
        v1 = x1.permute(0, 2, 1)
        v1 = torch.matmul(v1, x2)
        v1 = x1.permute(0, 2, 1)
        v1 = torch.matmul(v1, x3)
        v3 = x1.permute(0, 2, 1)
        v4 = x2.permute(0, 2, 1)
        v5 = x3.permute(0, 2, 1)
        v2 = torch.matmul(v3, v4)
        v2 = torch.matmul(v2, v5)
        return (v1, v2)

class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(3, 3, 1, 1, 0, 1, 1)
        self.relu6 = torch.nn.ReLU6(True)

    def forward(self, x):
        v1 = self.conv2d(x)
        v2 = self.relu6(v1)
        return v2

class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(3, 3, 1, 1, 0, 1, 1)
        self.relu6 = torch.nn.ReLU6(True)

    def forward(self, x):
        v1 = self.conv2d(x)
        v2 = self.conv2d(v1)
        v2 = v2.clamp_(0, 6)
        return v2

class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(3, 3, 1, 1, 0, 1, 1)
        self.relu6 = torch.nn.ReLU6(True)

    def forward(self, x):
        v1 = self.conv2d(x)
        v2 = self.conv2d(v1)
        v2 = v2.clamp_(0, 6)
        return v2

class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(3, 3, 1, 1, 0, 1, 1)
        self.relu6 = torch.nn.ReLU6(True)

    def forward(self, x):
        v1 = self.conv2d(x)
        v2 = v1.clamp_(0, 6)
        return v2


func = Model().to('cpu')


x1 = torch.randn(1, 2, 2)

x2 = torch.randn(1, 1, 2)

x3 = torch.randn(1, 2, 2)

x = torch.randn(2, 3, 2, 2)

input_x = torch.randn(2, 3, 2, 2, requires_grad=True)

test_inputs = [x1, x2, x3, x, input_x]

# JIT_FAIL
'''
direct:
forward() takes 2 positional arguments but 6 were given

jit:
forward() takes 2 positional arguments but 6 were given
'''