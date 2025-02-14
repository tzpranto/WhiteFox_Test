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

    def forward(self, x, y):
        v1 = torch.cat((x, y), 1)
        v2 = torch.cat((v1, v1[0:None]), dim=0)
        return v2


func = Model().to('cpu')


x = torch.tensor([1.0, 2.0, 3.0])

y = torch.tensor([4.0, 5.0, 6.0])

test_inputs = [x, y]

# JIT_FAIL
'''
direct:
Dimension out of range (expected to be in range of [-1, 0], but got 1)

jit:
Failed running call_function <built-in method cat of type object at 0x7fdd2365f1c0>(*((FakeTensor(..., size=(3,)), FakeTensor(..., size=(3,))), 1), **{}):
Dimension out of range (expected to be in range of [-1, 0], but got 1)

from user code:
   File "<string>", line 16, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''