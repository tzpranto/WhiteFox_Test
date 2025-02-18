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
        y = x.view(3, 80).permute(1, 0)
        y = y.tanh()
        y = y.t()
        y = y.tanh()
        y = y.permute(1, 0)
        y = y.view(3, 80)
        z = torch.cat((y, y), dim=0)
        if z.shape[0] == 2:
            z = z.tanh()
        else:
            z = z.reshape(2, 10)
        z = torch.relu(z)
        x = torch.cat((z, z), dim=2)
        y = x.tanh()
        z = y.view(-1).tanh()
        return z



func = Model().to('cuda:0')


x = torch.randn(2, 3, 10)

test_inputs = [x]

# JIT_FAIL
'''
direct:
shape '[3, 80]' is invalid for input of size 60

jit:
Failed running call_method view(*(FakeTensor(..., device='cuda:0', size=(2, 3, 10)), 3, 80), **{}):
shape '[3, 80]' is invalid for input of size 60

from user code:
   File "<string>", line 19, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''