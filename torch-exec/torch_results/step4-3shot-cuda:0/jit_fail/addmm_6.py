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

    def forward(self, x1, x2, inp):
        v1 = torch.einsum('ab,ac->bc', inp, x1)
        v2 = v1 + x2
        return v2



func = Model().to('cuda:0')


x1 = torch.randn(6, 12)

x2 = torch.randn(6, 6)

inp = torch.randn(12, 6)

test_inputs = [x1, x2, inp]

# JIT_FAIL
'''
direct:
einsum(): subscript a has size 6 for operand 1 which does not broadcast with previously seen size 12

jit:
Failed running call_function <function einsum at 0x7fd72b776d30>(*('ab,ac->bc', FakeTensor(..., device='cuda:0', size=(12, 6)), FakeTensor(..., device='cuda:0', size=(6, 12))), **{}):
einsum(): subscript a has size 6 for operand 1 which does not broadcast with previously seen size 12

from user code:
   File "<string>", line 19, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''