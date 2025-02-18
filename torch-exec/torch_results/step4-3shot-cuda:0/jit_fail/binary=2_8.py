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

    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - x1
        return v2



func = Model().to('cuda:0')


x1 = torch.randn(1, 3, 224, 224)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
The size of tensor a (226) must match the size of tensor b (224) at non-singleton dimension 3

jit:
Failed running call_function <built-in function sub>(*(FakeTensor(..., device='cuda:0', size=(1, 8, 226, 226)), FakeTensor(..., device='cuda:0', size=(1, 3, 224, 224))), **{}):
Attempting to broadcast a dimension of length 224 at -1! Mismatching argument at index 1 had torch.Size([1, 3, 224, 224]); but expected shape should be broadcastable to [1, 8, 226, 226]

from user code:
   File "<string>", line 21, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''