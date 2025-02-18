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

    def forward(self, x, y):
        z = x.cat()
        x = y + z
        return x



func = Model().to('cuda:0')


x = torch.randn(1)

y = torch.randn(1)

test_inputs = [x, y]

# JIT_FAIL
'''
direct:
'Tensor' object has no attribute 'cat'

jit:
AttributeError: 'Tensor' object has no attribute 'cat'

from user code:
   File "<string>", line 19, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''