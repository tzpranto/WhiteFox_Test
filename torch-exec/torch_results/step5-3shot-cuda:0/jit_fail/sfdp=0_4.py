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
        y1 = torch.matmul(x1, x2)
        y2 = torch.linalg.inv(torch.eye(x3.shape[-1]) * x4 ** 0.5)
        y3 = torch.matmul(y1, torch.transpose(y2))
        y4 = torch.softmax(y3, 1)
        y5 = torch.matmul(x3, y4)
        return y5


func = Model().to('cuda:0')


x1 = torch.randn(2, 10, 5)

x2 = torch.randn(2, 5, 10)

x3 = torch.randn(2, 10, 16)
x4 = 1

test_inputs = [x1, x2, x3, x4]

# JIT_FAIL
'''
direct:
transpose() received an invalid combination of arguments - got (Tensor), but expected one of:
 * (Tensor input, int dim0, int dim1)
 * (Tensor input, name dim0, name dim1)


jit:
transpose() received an invalid combination of arguments - got (Tensor), but expected one of:
 * (Tensor input, int dim0, int dim1)
 * (Tensor input, name dim0, name dim1)

'''