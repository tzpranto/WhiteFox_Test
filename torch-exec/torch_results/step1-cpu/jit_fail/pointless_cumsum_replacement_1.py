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
        v1 = torch.full((64, 64), 1, dtype=torch.float32)
        v2 = torch.convert_element_type(v1, dtype=torch.float64)
        return torch.cumsum(v2, 1)


func = Model().to('cpu')


x = torch.randn(1, 3, 64, 64)

test_inputs = [x]

# JIT_FAIL
'''
direct:
module 'torch' has no attribute 'convert_element_type'

jit:
AttributeError: module 'torch' has no attribute 'convert_element_type'

from user code:
   File "<string>", line 20, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''