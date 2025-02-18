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

    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
        self.linear.weight.data = torch.eye(3)[::-1].view(8, 3)
        self.linear.bias = torch.nn.Parameter(torch.zeros(3))
        self.other = torch.tensor(other, dtype)

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - self.other
        v3 = F.relu(v2)
        return v3


other = 1
func = Model(-1.5).to('cpu')


x1 = torch.randn(1, 3, 64, 64)
x_min = torch.min(x1, dim=3)[0]
x_min = x_min.flatten(start_dim=1)
x_max = torch.max(x1, dim=3)[0]
x_max = x_max.flatten(start_dim=1)

x1 = torch.randn(1, 3, 64, 64)
x_mean = torch.mean(x1, dim=[3, 2]).flatten(start_dim=1)

x1 = torch.randn(1, 3, 64, 64)
x_std = torch.std(x1, dim=[3, 2]).flatten(start_dim=1)

test_inputs = [x1, x_min, x_max, x_mean, x_std]
