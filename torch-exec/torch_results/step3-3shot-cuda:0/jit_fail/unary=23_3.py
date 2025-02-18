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
        self.conv2d_1 = torch.nn.Conv2d(1, 1, 1, bias=False, padding=(0, 0), stride=(1, 1))
        self.conv_transpose_1 = torch.nn.ConvTranspose1d(1, 1, 1, bias=True, padding=(0, 0), stride=(1, 1))

    def forward(self, x1):
        v1 = self.conv2d_1(x1)
        v2 = self.conv_transpose_1(v1)
        return v2



func = Model().to('cuda:0')


x1 = torch.randn(2, 1, 10, 10)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
Expected 2D (unbatched) or 3D (batched) input to conv_transpose1d, but got input of size: [2, 1, 10, 10]

jit:
Failed running call_function <built-in method conv_transpose1d of type object at 0x7f6a94a5f1c0>(*(FakeTensor(..., device='cuda:0', size=(2, 1, 10, 10)), Parameter(FakeTensor(..., device='cuda:0', size=(1, 1, 1), requires_grad=True)), Parameter(FakeTensor(..., device='cuda:0', size=(1,), requires_grad=True)), (1, 1), (0, 0), (0,), 1, (1,)), **{}):
Expected 2D (unbatched) or 3D (batched) input to conv_transpose1d, but got input of size: [2, 1, 10, 10]

from user code:
   File "<string>", line 22, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 974, in forward
    return F.conv_transpose1d(


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''