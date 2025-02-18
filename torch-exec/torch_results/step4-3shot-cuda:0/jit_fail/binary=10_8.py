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

    def __init__(self, other_tensor_a, other_tensor_b):
        super().__init__()
        self.linear = torch.nn.Linear(5, 10)

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + other_tensor_a
        v3 = v1 + other_tensor_b
        return (v2, v3)


other_tensor_a = torch.randn(10, 5)
other_tensor_b = torch.randn(10, 5)
func = Model(other_tensor_a, other_tensor_b).to('cuda:0')


other_tensor_a = torch.randn(10, 5)

other_tensor_b = torch.randn(10, 5)

x1 = torch.randn(1, 5)

test_inputs = [other_tensor_a, other_tensor_b, x1]

# JIT_FAIL
'''
direct:
forward() takes 2 positional arguments but 4 were given

jit:
forward() takes 2 positional arguments but 4 were given
'''