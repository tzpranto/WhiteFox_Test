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
        self.conv1 = torch.nn.Conv2d(1, 1, 3, stride=1, padding=0)

    def forward(self, x1):
        b = {}
        a = {}
        b['dtype'] = torch.bool
        b['layout'] = torch.strided
        b['device'] = torch.device('cuda:0')
        a['dtype'] = torch.float32
        a['layout'] = torch.strided
        a['device'] = torch.device('cuda:0')
        a['dtype_to'] = torch.float16
        a['dtype_from'] = torch.float32
        b['dtype_to'] = torch.float32
        b['dtype_from'] = torch.float16
        t1 = torch.full([25, 1024, 10], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = t1.to(dtype=a['dtype'])
        t3 = torch.cumsum(t2, 1)
        t4 = self.conv1(t3)
        t5 = t4.type(torch.float16)
        t5 = t5.div(1024)
        return t5



func = Model().to('cuda:0')


x1 = torch.rand(25, 1, 1024, 1024, device='cuda:0')

test_inputs = [x1]

# JIT_FAIL
'''
direct:
Given groups=1, weight of size [1, 1, 3, 3], expected input[1, 25, 1024, 10] to have 1 channels, but got 25 channels instead

jit:
Failed running call_function <built-in method conv2d of type object at 0x7f051dc5f1c0>(*(FakeTensor(..., device='cuda:0', size=(25, 1024, 10)), Parameter(FakeTensor(..., device='cuda:0', size=(1, 1, 3, 3), requires_grad=True)), Parameter(FakeTensor(..., device='cuda:0', size=(1,), requires_grad=True)), (1, 1), (0, 0), (1, 1), 1), **{}):
Given groups=1, weight of size [1, 1, 3, 3], expected input[1, 25, 1024, 10] to have 1 channels, but got 25 channels instead

from user code:
   File "<string>", line 35, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 554, in forward
    return self._conv_forward(input, self.weight, self.bias)
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 549, in _conv_forward
    return F.conv2d(


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''