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

    def forward(self, x):
        input228 = torch.split_with_sizes(x, [1, 1, 225, 225], 1)
        v0 = []
        v1 = 0
        for i in range(len(input228)):
            v1 = v1 + 1
            v2 = input228[v1].squeeze(1)
            v3 = torch.unsqueeze(v2, 0)
            v4 = [v0.append(v3) for _ in range(v2.size(0))]
        v5 = torch.cat(v0, 0)
        return v5


func = Model().to('cpu')


x = torch.randn(25, 6)

test_inputs = [x]

# JIT_FAIL
'''
direct:
split_with_sizes expects split_sizes to sum exactly to 6 (input tensor's size at dimension 1), but got split_sizes=[1, 1, 225, 225]

jit:
Failed running call_function <built-in method split_with_sizes of type object at 0x7f42a2a5f1c0>(*(FakeTensor(..., size=(25, 6)), [1, 1, 225, 225], 1), **{}):
Split sizes add up to 452 but got the tensor's size of 6

from user code:
   File "<string>", line 16, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''