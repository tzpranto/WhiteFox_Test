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

    def forward(self, x, y):
        x = torch.Tensor.addmm(b=x, mat1=y, mat2=y, alpha=1, beta=1)
        return x


func = Model().to('cpu')


x = torch.randn(8, 8)

y = torch.randn(8, 8)

test_inputs = [x, y]

# JIT_FAIL
'''
direct:
unbound method TensorBase.addmm() needs an argument

jit:
unbound method TensorBase.addmm() needs an argument
'''