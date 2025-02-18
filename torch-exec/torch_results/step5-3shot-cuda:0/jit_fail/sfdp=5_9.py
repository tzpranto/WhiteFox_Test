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
        model = torch.nn.Transformer(d_model=512)
        self.model = model
        self.linear = torch.nn.Linear(512, 128)

    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = self.model(v1, x2)
        v3 = self.linear(v2)
        return v3


func = Model().to('cuda:0')


x1 = torch.randn(10, 128)

x2 = torch.randn(10, 10, 512)

test_inputs = [x1, x2]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (10x128 and 512x128)

jit:
Failed running call_function <built-in function linear>(*(FakeTensor(..., device='cuda:0', size=(10, 128)), Parameter(FakeTensor(..., device='cuda:0', size=(128, 512), requires_grad=True)), Parameter(FakeTensor(..., device='cuda:0', size=(128,), requires_grad=True))), **{}):
a and b must have same reduction dim, but got [10, 128] X [512, 128].

from user code:
   File "<string>", line 22, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''