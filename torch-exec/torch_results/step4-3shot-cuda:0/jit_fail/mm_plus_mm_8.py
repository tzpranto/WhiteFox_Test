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

    def forward(self, input1, input2, input3, input4):
        t1 = torch.mm(input1, input2)
        t2 = torch.mm(input3, input4)
        x1 = t1 + t2
        return x1



func = Model().to('cuda:0')


input1 = torch.randn(4, 3)

input2 = torch.randn(4, 4)

input3 = torch.randn(3, 4)

input4 = torch.randn(3, 3)

test_inputs = [input1, input2, input3, input4]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (4x3 and 4x4)

jit:
Failed running call_function <built-in method mm of type object at 0x7f2fe2c5f1c0>(*(FakeTensor(..., device='cuda:0', size=(4, 3)), FakeTensor(..., device='cuda:0', size=(4, 4))), **{}):
a and b must have same reduction dim, but got [4, 3] X [4, 4].

from user code:
   File "<string>", line 19, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''