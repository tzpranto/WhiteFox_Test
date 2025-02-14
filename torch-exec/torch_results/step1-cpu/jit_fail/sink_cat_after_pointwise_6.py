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
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, x1):
        x = torch.cat(x1, x1, 1)
        x = x.view(2, 4)
        x += 1
        return x.relu()


func = Model().to('cpu')


x1 = torch.randn(2, 2)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
cat() received an invalid combination of arguments - got (Tensor, Tensor, int), but expected one of:
 * (tuple of Tensors tensors, int dim = 0, *, Tensor out = None)
 * (tuple of Tensors tensors, name dim, *, Tensor out = None)


jit:
cat() received an invalid combination of arguments - got (Tensor, Tensor, int), but expected one of:
 * (tuple of Tensors tensors, int dim = 0, *, Tensor out = None)
 * (tuple of Tensors tensors, name dim, *, Tensor out = None)

'''