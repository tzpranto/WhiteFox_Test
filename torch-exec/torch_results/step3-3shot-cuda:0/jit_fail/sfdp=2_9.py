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

    def forward(self, __input1__, __input2__):
        w1 = __input1__.transpose(-2, -1).matmul(__input2__)
        w2 = w1.sum(dim=2)
        w3 = w2.div(8)
        w4 = torch.nn.functional.softmax(w3, dim=1)
        w5 = torch.nn.functional.dropout(w4, p=0.3)
        w6 = w5.matmul(__input2__)
        return w6


func = Model().to('cuda:0')


x1 = torch.randn(1, 8, 5)

x2 = torch.randn(1, 8, 8)

test_inputs = [x1, x2]

# JIT_FAIL
'''
direct:
Expected size for first two dimensions of batch2 tensor to be: [1, 5] but got: [1, 8].

jit:
Failed running call_method matmul(*(FakeTensor(..., device='cuda:0', size=(1, 5)), FakeTensor(..., device='cuda:0', size=(1, 8, 8))), **{}):
Expected size for first two dimensions of batch2 tensor to be: [1, 5] but got: [1, 8].

from user code:
   File "<string>", line 24, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''