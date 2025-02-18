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
        self.conv_1 = torch.nn.Conv2d(2, 7, 5, stride=1, padding=2)
        self.conv_2 = torch.nn.Conv2d(7, 10, 1, stride=1, padding=1)
        self.conv_3 = torch.nn.Conv2d(10, 15, 3, stride=1, padding=2)

    def forward(self, x1):
        v1 = self.conv_1(x1)
        v2 = v1 * 0.5
        v3 = v1 * v1
        v4 = v3 * v1
        v5 = v4 * 0.044715
        v6 = v1 + v5
        v7 = v6 * 0.7978845608028654
        v8 = torch.tanh(v7)
        v9 = v8 + 1
        v10 = v2 * v9
        v11 = self.conv_2(v10)
        v13 = v1 * v10
        v14 = v13 * v11
        v15 = v14 * 0.044715
        v16 = v11 + v15
        v17 = v16 * 0.7978845608028654
        v18 = torch.tanh(v17)
        v19 = v18 + 1
        v20 = v1 * v19
        v21 = v10 + v20
        v23 = v1 + v19
        v26 = v11 * v8
        v28 = v19 * torch.tanh(v11)
        v24 = v11 * v8
        v27 = v11 * v9
        v25 = v28 + v19
        v22 = v24 + v27
        return (v21, v22, v23)



func = Model().to('cuda:0')


x1 = torch.randn(1, 2, 32, 32)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
The size of tensor a (32) must match the size of tensor b (34) at non-singleton dimension 3

jit:
Failed running call_function <built-in function mul>(*(FakeTensor(..., device='cuda:0', size=(1, 7, 32, 32)), FakeTensor(..., device='cuda:0', size=(1, 10, 34, 34))), **{}):
Attempting to broadcast a dimension of length 34 at -1! Mismatching argument at index 1 had torch.Size([1, 10, 34, 34]); but expected shape should be broadcastable to [1, 7, 32, 32]

from user code:
   File "<string>", line 34, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''