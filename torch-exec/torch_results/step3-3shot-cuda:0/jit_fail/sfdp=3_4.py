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

    def forward(self, x1, x2):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1 * 0.125
        v3 = torch.nn.functional.softmax(v2, -1)
        v4 = torch.nn.functional.dropout(v3)
        output = v4.matmul(x2)
        return output


func = Model().to('cuda:0')


x1 = torch.randn(5, 768, 8)

x2 = torch.randn(5, 8, 768)

test_inputs = [x1, x2]

# JIT_FAIL
'''
direct:
Expected size for first two dimensions of batch2 tensor to be: [5, 8] but got: [5, 768].

jit:
Failed running call_function <built-in method matmul of type object at 0x7f9d4aa5f1c0>(*(FakeTensor(..., device='cuda:0', size=(5, 768, 8)), FakeTensor(..., device='cuda:0', size=(5, 768, 8))), **{}):
Expected size for first two dimensions of batch2 tensor to be: [5, 8] but got: [5, 768].

from user code:
   File "<string>", line 19, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''