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
        for module in self.modules():
            if type(module) == torch.nn.Linear:
                module.bias.data = module.bias.data / 6
                module.weight.data = module.weight.data / 6

    def forward(self, x1):
        l1 = torch.nn.functional.linear(x1, torch.nn.Linear(3, 8, bias=False).weight)
        l2 = l1 + 3
        l3 = torch.nn.functional.relu6(l2)
        return l3


func = Model().to('cpu')


x1 = torch.randn(1, 3, 64, 64)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (192x64 and 3x8)

jit:
Failed running call_function <built-in function linear>(*(FakeTensor(..., size=(1, 3, 64, 64)), Parameter(FakeTensor(..., size=(8, 3), requires_grad=True))), **{}):
a and b must have same reduction dim, but got [192, 64] X [3, 8].

from user code:
   File "<string>", line 23, in torch_dynamo_resume_in_forward_at_23


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''