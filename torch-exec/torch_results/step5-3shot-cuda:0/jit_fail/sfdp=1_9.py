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

    def forward(self, t5, t6, t7):
        qk = torch.matmul(t5, t6.transpose(-2, -1))
        scaled_qk = qk.div(2.0 ** (-1.0))
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.3)
        output = dropout_qk.matmul(t7)
        return output


func = Model().to('cuda:0')


t5 = torch.randn(4, 20, 40, 50)

t6 = torch.randn(4, 20, 30, 25)

t7 = torch.randn(4, 20, 30, 25)

test_inputs = [t5, t6, t7]

# JIT_FAIL
'''
direct:
Expected size for first two dimensions of batch2 tensor to be: [80, 50] but got: [80, 25].

jit:
Failed running call_function <built-in method matmul of type object at 0x7efff6a5f1c0>(*(FakeTensor(..., device='cuda:0', size=(4, 20, 40, 50)), FakeTensor(..., device='cuda:0', size=(4, 20, 25, 30))), **{}):
Expected size for first two dimensions of batch2 tensor to be: [80, 50] but got: [80, 25].

from user code:
   File "<string>", line 19, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''