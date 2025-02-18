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
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)

    def forward(self, x1, other=np.random.randn(1, 3, 64, 64)):
        v1 = self.conv(x1)
        if np.array_equal(other.shape, v1.shape):
            other = 0
        v2 = v1 + other
        return v2



func = Model().to('cuda:0')


x1 = torch.randn(1, 3, 64, 64)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.

jit:
Failed running call_function <built-in function add>(*(FakeTensor(..., device='cuda:0', size=(1, 8, 66, 66)), FakeTensor(..., size=(1, 3, 64, 64), dtype=torch.float64)), **{}):
Attempting to broadcast a dimension of length 64 at -1! Mismatching argument at index 1 had torch.Size([1, 3, 64, 64]); but expected shape should be broadcastable to [1, 8, 66, 66]

from user code:
   File "<string>", line 23, in torch_dynamo_resume_in_forward_at_21


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''