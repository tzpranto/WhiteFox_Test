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
        self.norm = torch.nn.LayerNorm([100, 10])

    def forward(self, input):
        v1 = self.norm(input)
        v2 = torch.nn.functional.softmax(v1, dim=-1)
        v3 = torch.nn.functional.dropout(v2, p=0.5, training=self.training)
        v4 = v3.matmul(input)
        return v4


func = Model().to('cuda:0')


x1 = torch.randn(1, 100, 10)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
Expected size for first two dimensions of batch2 tensor to be: [1, 10] but got: [1, 100].

jit:
Failed running call_method matmul(*(FakeTensor(..., device='cuda:0', size=(1, 100, 10)), FakeTensor(..., device='cuda:0', size=(1, 100, 10))), **{}):
Expected size for first two dimensions of batch2 tensor to be: [1, 10] but got: [1, 100].

from user code:
   File "<string>", line 23, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''