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

    def forward(self, x):
        result = [torch.ops.aten.addmm(x, x, x)] * 5
        x = torch.aten.cat(result, 0)
        return x


func = Model().to('cpu')


x = torch.randn(3, 3)

test_inputs = [x]

# JIT_FAIL
'''
direct:
module 'torch' has no attribute 'aten'

jit:
AttributeError: module 'torch' has no attribute 'aten'

from user code:
   File "<string>", line 17, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''