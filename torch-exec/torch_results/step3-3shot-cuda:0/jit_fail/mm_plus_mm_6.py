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

    def forward(self, input1, input2, input3, input4):
        t1 = torch.mm(input1, input2)
        t2 = torch.mm(input3, input4)
        t3 = t1 + t2
        return t3



func = Model().to('cuda:0')


input1 = torch.tensor(1)

input2 = torch.tensor(1)

input3 = torch.tensor(1)

input4 = torch.tensor([[]])

test_inputs = [input1, input2, input3, input4]

# JIT_FAIL
'''
direct:
self must be a matrix

jit:
Failed running call_function <built-in method mm of type object at 0x7efd99a5f1c0>(*(FakeTensor(..., device='cuda:0', size=(), dtype=torch.int64), FakeTensor(..., device='cuda:0', size=(), dtype=torch.int64)), **{}):
a must be 2D

from user code:
   File "<string>", line 16, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''