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

    def forward(self, x):
        y = x.view(x.shape[0], -1)
        y = y.view(y.shape[0], -1)
        x = y.view(y.shape[0], -1)
        y = x.view(y.shape[0], -1).relu()
        x = y.view(-1, y.shape[1]).relu()
        y = x.view(y.type(torch.BoolTensor))
        return x



func = Model().to('cuda:0')


x = torch.randn(2, 3, 4)

test_inputs = [x]

# JIT_FAIL
'''
direct:
view() received an invalid combination of arguments - got (Tensor), but expected one of:
 * (torch.dtype dtype)
      didn't match because some of the arguments have invalid types: (!Tensor!)
 * (tuple of ints size)
      didn't match because some of the arguments have invalid types: (!Tensor!)


jit:
view() received an invalid combination of arguments - got (Tensor), but expected one of:
 * (torch.dtype dtype)
      didn't match because some of the arguments have invalid types: (!Tensor!)
 * (tuple of ints size)
      didn't match because some of the arguments have invalid types: (!Tensor!)

'''