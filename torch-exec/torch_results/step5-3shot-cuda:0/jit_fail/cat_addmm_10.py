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

    def forward(self, x, mat1, mat2):
        return torch.cat([torch.addmm(input, mat1, mat2)], dim=1)


func = Model().to('cuda:0')


x = torch.randn(1, 16)

mat1 = torch.randn(4, 16)

mat2 = torch.randn(4, 16)

test_inputs = [x, mat1, mat2]

# JIT_FAIL
'''
direct:
addmm(): argument 'input' (position 1) must be Tensor, not builtin_function_or_method

jit:
addmm(): argument 'input' (position 1) must be Tensor, not builtin_function_or_method
'''