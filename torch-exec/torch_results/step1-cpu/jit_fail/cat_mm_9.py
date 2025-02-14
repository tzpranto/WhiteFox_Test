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
        v1 = torch.randn(4, 4, requires_grad=True)
        v2 = torch.randn(3, 4, requires_grad=True)
        v3 = torch.randn(4, 3, requires_grad=True)
        v4 = torch.randn(3, 4, requires_grad=True)
        v5 = torch.randn(4, 4, requires_grad=True)
        v6 = torch.randn(3, 3, requires_grad=True)
        v7 = torch.mm(v1, v2)
        v8 = torch.mm(v3, v4)
        v9 = torch.mm(v5, v6)
        output = torch.cat([v7, v8], 1)
        output = torch.cat([output, v9], 1)
        return output


func = Model().to('cpu')


x = torch.randn(1, 8, 8)

test_inputs = [x]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (4x4 and 3x4)

jit:
Failed running call_function <built-in method mm of type object at 0x7f1ed905f1c0>(*(FakeTensor(..., size=(4, 4), requires_grad=True), FakeTensor(..., size=(3, 4), requires_grad=True)), **{}):
a and b must have same reduction dim, but got [4, 4] X [3, 4].

from user code:
   File "<string>", line 25, in torch_dynamo_resume_in_forward_at_24


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''