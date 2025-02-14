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
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)

    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 * 0.5049359050664642
        v3 = v1 * 3.857285613210263
        v4 = torch.elu(v3, alpha=1, inplace=False)
        v5 = v3 * 3.00707337171274
        v6 = v4 + 0.005188588426746512
        v7 = v2 * v6
        return v7


func = Model().to('cpu')


x = torch.randn(1, 3, 64, 64)

test_inputs = [x]

# JIT_FAIL
'''
direct:
module 'torch' has no attribute 'elu'

jit:
AttributeError: module 'torch' has no attribute 'elu'

from user code:
   File "<string>", line 23, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''