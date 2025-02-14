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
        pass

    def forward(self, img, fushion):
        out1 = torch.matmul(img, fushion)
        out2 = torch.matmul(img, fushion)
        out3 = torch.matmul(img, fushion)
        fc = torch.cat([out1, out2, out3], dim=1)
        return fc


func = Model().to('cpu')


img = torch.randn(1, 104, 400)

fushion = torch.randn(104, 256)

test_inputs = [img, fushion]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (104x400 and 104x256)

jit:
Failed running call_function <built-in method matmul of type object at 0x7f1ed905f1c0>(*(FakeTensor(..., size=(1, 104, 400)), FakeTensor(..., size=(104, 256))), **{}):
a and b must have same reduction dim, but got [104, 400] X [104, 256].

from user code:
   File "<string>", line 20, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''