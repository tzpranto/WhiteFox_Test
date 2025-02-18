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
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 4, 2, 3, 1)
        return v2



func = Model().to('cuda:0')


x1 = torch.randn(1, 2, 2, 1, 2, 2)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
permute(sparse_coo): number of dimensions in the tensor input does not match the length of the desired ordering of dimensions i.e. input.dim() = 6 is not equal to len(dims) = 5

jit:
Failed running call_method permute(*(FakeTensor(..., device='cuda:0', size=(1, 2, 2, 1, 2, 2)), 0, 4, 2, 3, 1), **{}):
Attempting to permute a tensor of rank 6, but received a permutation of length 5!

from user code:
   File "<string>", line 21, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''