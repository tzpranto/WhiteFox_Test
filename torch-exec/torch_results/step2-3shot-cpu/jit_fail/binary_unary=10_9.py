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
        weight_init = torch.zeros(384, 256)
        self.linear = torch.nn.Linear(384, 256, bias=False)
        self.linear.weight.data = weight_init

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + x1
        v3 = torch.nn.functional.relu(v2)
        return v3


func = Model().to('cpu')


x1 = torch.randn(1, 384)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (1x384 and 256x384)

jit:
Failed running call_function <built-in function linear>(*(FakeTensor(..., size=(1, 384)), Parameter(FakeTensor(..., size=(384, 256), requires_grad=True)), None), **{}):
a and b must have same reduction dim, but got [1, 384] X [256, 384].

from user code:
   File "<string>", line 22, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''