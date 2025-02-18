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
        y = np.zeros([239, 3])
        if x.dim() == 3:
            x = torch.cat((x, x), dim=1)
        y = torch.cat((x, y), dim=1)
        if y.dim() == 3:
            y = y.tanh()
        y = y.view(y.shape[0], y.shape[1], -1).tanh()
        y = torch.cat((y, y), dim=1)
        x = y.view(y.shape[0], -1).tanh()
        return x



func = Model().to('cuda:0')


x = torch.randn(2, 3, 4)

test_inputs = [x]

# JIT_FAIL
'''
direct:
expected Tensor as element 1 in argument 0, but got numpy.ndarray

jit:
Failed running call_function <built-in method cat of type object at 0x7f74f2a5f1c0>(*((FakeTensor(..., device='cuda:0', size=(2, 6, 4)), FakeTensor(..., size=(239, 3), dtype=torch.float64)),), **{'dim': 1}):
Number of dimensions of tensors must match.  Expected 3-D tensors, but got 2-D for tensor number 1 in the list

from user code:
   File "<string>", line 22, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''