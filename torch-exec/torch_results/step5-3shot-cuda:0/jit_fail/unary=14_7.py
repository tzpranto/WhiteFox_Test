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
        self.convtranspose = torch.nn.ConvTranspose2d(1, 1, kernel_size=3, stride=1, padding=1, output_padding=1, bias=False)
        self.fc1 = torch.nn.Linear(8, 16)

    def forward(self, x1):
        v1 = self.convtranspose(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.fc1(v3)
        v5 = torch.sigmoid(v3)
        v6 = v4 * v5
        return x1



func = Model().to('cuda:0')


x1 = torch.randn(1, 1, 8, 8)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
output padding must be smaller than either stride or dilation, but got output_padding_height: 1 output_padding_width: 1 stride_height: 1 stride_width: 1 dilation_height: 1 dilation_width: 1

jit:
Failed running call_function <built-in function linear>(*(FakeTensor(..., device='cuda:0', size=(1, 1, 9, 9)), Parameter(FakeTensor(..., device='cuda:0', size=(16, 8), requires_grad=True)), Parameter(FakeTensor(..., device='cuda:0', size=(16,), requires_grad=True))), **{}):
a and b must have same reduction dim, but got [9, 9] X [8, 16].

from user code:
   File "<string>", line 24, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''