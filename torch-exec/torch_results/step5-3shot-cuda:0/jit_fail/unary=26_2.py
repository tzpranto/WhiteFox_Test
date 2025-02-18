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

    def __init__(self, negative_slope):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 1, stride=1, padding=1)
        self.relu = torch.nn.ReLU6(inplace=True)
        self.negative_slope = negative_slope

    def forward(self, x1):
        x2 = self.conv_transpose(x1)
        x3 = self.relu(x2)
        x4 = x3 * self.negative_slope
        x5 = torch.where(x3, x2, x4)
        return x5


negative_slope = 1

func = Model(negative_slope).to('cuda:0')


x1 = torch.randn(1, 3, 64, 64)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
where expected condition to be a boolean tensor, but got a tensor with dtype Float

jit:
Failed running call_function <built-in method where of type object at 0x7f6aca85f1c0>(*(FakeTensor(..., device='cuda:0', size=(1, 8, 62, 62)), FakeTensor(..., device='cuda:0', size=(1, 8, 62, 62)), FakeTensor(..., device='cuda:0', size=(1, 8, 62, 62))), **{}):
expected predicate to be bool, got torch.float32

from user code:
   File "<string>", line 25, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''