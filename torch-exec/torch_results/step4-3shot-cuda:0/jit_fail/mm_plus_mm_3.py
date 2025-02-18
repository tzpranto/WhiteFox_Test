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

    def forward(self, x1, x2, x3, x4):
        out = torch.mm(x1, x4) + torch.mm(x1, x4)
        out.transpose()
        return out



func = Model().to('cuda:0')


x1 = torch.randn(3, 2)

x2 = torch.randn(2, 5)

x3 = torch.randn(3, 2)

x4 = torch.randn(2, 5)

test_inputs = [x1, x2, x3, x4]

# JIT_FAIL
'''
direct:
transpose() received an invalid combination of arguments - got (), but expected one of:
 * (int dim0, int dim1)
 * (name dim0, name dim1)


jit:
transpose() received an invalid combination of arguments - got (), but expected one of:
 * (int dim0, int dim1)
 * (name dim0, name dim1)

'''