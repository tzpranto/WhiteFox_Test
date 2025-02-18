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
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(1, 1, kernel_size=(1, 1))
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(2, 1, kernel_size=(1, 1))
        self.conv_transpose_3 = torch.nn.ConvTranspose2d(5, 7, kernel_size=(1, 1))
        self.conv_transpose_4 = torch.nn.ConvTranspose2d(6, 7, kernel_size=(1, 1))
        self.conv_transpose_5 = torch.nn.ConvTranspose2d(1, 4, kernel_size=(1, 1))
        self.conv_transpose_6 = torch.nn.ConvTranspose2d(8, 5, kernel_size=(1, 1))
        self.conv_transpose_7 = torch.nn.ConvTranspose2d(9, 3, kernel_size=(1, 1))
        self.conv_transpose_8 = torch.nn.ConvTranspose2d(10, 6, kernel_size=(1, 1))

    def forward(self, x):
        x = self.conv_transpose_1(x)
        x = torch.tanh(x)
        x = self.conv_transpose_2(x)
        x = torch.tanh(x)
        x = self.conv_transpose_3(x)
        x = torch.tanh(x)
        x = self.conv_transpose_4(x)
        x = torch.tanh(x)
        x = self.conv_transpose_5(x)
        x = torch.tanh(x)
        x = self.conv_transpose_6(x)
        x = torch.tanh(x)
        x = self.conv_transpose_7(x)
        x = torch.tanh(x)
        x = self.conv_transpose_8(x)
        x = torch.tanh(x)
        return x



func = Model().to('cuda:0')


x = torch.randn(1, 10, 1, 1)

test_inputs = [x]

# JIT_FAIL
'''
direct:
Given transposed=1, weight of size [1, 1, 1, 1], expected input[1, 10, 1, 1] to have 1 channels, but got 10 channels instead

jit:
Failed running call_function <built-in method conv_transpose2d of type object at 0x7f6a94a5f1c0>(*(FakeTensor(..., device='cuda:0', size=(1, 10, 1, 1)), Parameter(FakeTensor(..., device='cuda:0', size=(1, 1, 1, 1), requires_grad=True)), Parameter(FakeTensor(..., device='cuda:0', size=(1,), requires_grad=True)), (1, 1), (0, 0), (0, 0), 1, (1, 1)), **{}):
Given transposed=1, weight of size [1, 1, 1, 1], expected input[1, 10, 1, 1] to have 1 channels, but got 10 channels instead

from user code:
   File "<string>", line 27, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 1162, in forward
    return F.conv_transpose2d(


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''