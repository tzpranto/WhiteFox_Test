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

    def forward(self, x1, x2, x3):
        v4 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = __ // __
        v3 = v4.__ // __
        v5 = v3.softmax(dim=-1)
        v6 = v5.__ // __
        v7 = torch.matmul(v6, x3)
        return v7


func = Model().to('cuda:0')


x1 = torch.randn(3, 10)

x2 = torch.randn(6, 10)

x3 = torch.randn(6, 20)

test_inputs = [x1, x2, x3]

# JIT_FAIL
'''
direct:
name '__' is not defined

jit:
NameError: name '__' is not defined

from user code:
   File "<string>", line 20, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''