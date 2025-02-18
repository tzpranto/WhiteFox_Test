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

    def forward(self, param0):
        var2 = torch.Tensor.permute(param0, 1, 0)
        var1 = var2 - param0
        var0 = torch.bmm(param0, var1)
        return var0



func = Model().to('cuda:0')


param0 = torch.randn(2, 2, 2)

test_inputs = [param0]

# JIT_FAIL
'''
direct:
permute(sparse_coo): number of dimensions in the tensor input does not match the length of the desired ordering of dimensions i.e. input.dim() = 3 is not equal to len(dims) = 2

jit:
Failed running call_function <method 'permute' of 'torch._C.TensorBase' objects>(*(FakeTensor(..., device='cuda:0', size=(2, 2, 2)), 1, 0), **{}):
Attempting to permute a tensor of rank 3, but received a permutation of length 2!

from user code:
   File "<string>", line 19, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''