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
        v1 = x1 @ x2.transpose(-2, -1)
        v2 = v1 / math.sqrt(v1.size(-1))
        v3 = v2 + x3.float().masked_fill(x3 == float('-inf'), -1000000000.0)
        v4 = torch.softmax(v3, dim=-1)



func = Model().to('cpu')

x1 = 1
x2 = 1
x3 = 1

test_inputs = [x1, x2, x3]

# JIT_FAIL
'''
direct:
'int' object has no attribute 'transpose'

jit:
AttributeError: 'int' object has no attribute 'transpose'

from user code:
   File "<string>", line 19, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''