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

    def forward(self, input, weights, bias):
        return torch.add(torch.matmul(input, weights), bias)


func = Model().to('cpu')


input = torch.randn(4, 3)

weights = torch.randn(5, 3)

bias = torch.randn(5)

test_inputs = [input, weights, bias]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (4x3 and 5x3)

jit:
Failed running call_function <built-in method matmul of type object at 0x7fce11a5f1c0>(*(FakeTensor(..., size=(4, 3)), FakeTensor(..., size=(5, 3))), **{}):
a and b must have same reduction dim, but got [4, 3] X [5, 3].

from user code:
   File "<string>", line 19, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''