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
        (t0, t1, t2, t3) = torch.split(x, [4, 2, 2, 1], 1)
        t4 = torch.cat((t0, t1), 1)
        t5 = torch.cat((t2, t3), 1)
        t6 = torch.cat((t4, t5), 1)
        return t6


func = Model().to('cuda:0')


x = torch.randn(1, 6, 4, 4)

test_inputs = [x]

# JIT_FAIL
'''
direct:
split_with_sizes expects split_sizes to sum exactly to 6 (input tensor's size at dimension 1), but got split_sizes=[4, 2, 2, 1]

jit:
Failed running call_function <function split at 0x7f321ae78ca0>(*(FakeTensor(..., device='cuda:0', size=(1, 6, 4, 4)), [4, 2, 2, 1], 1), **{}):
Split sizes add up to 9 but got the tensor's size of 6

from user code:
   File "<string>", line 19, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''