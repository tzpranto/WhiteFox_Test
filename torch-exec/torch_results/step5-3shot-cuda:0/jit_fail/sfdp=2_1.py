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

    def forward(self, x1, x2):
        qk = torch.matmul(x1, x2.transpose(-2, -1))
        scaled_qk = qk.div(0.1)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.05)
        output = dropout_qk.matmul(x2)
        return output



func = Model().to('cuda:0')


x1 = torch.randn(1, 128).requires_grad_()

x2 = torch.randn(1, 128, 256).requires_grad_()

test_inputs = [x1, x2]

# JIT_FAIL
'''
direct:
Expected size for first two dimensions of batch2 tensor to be: [1, 128] but got: [1, 256].

jit:
Failed running call_function <built-in method matmul of type object at 0x7f7e6e85f1c0>(*(FakeTensor(..., device='cuda:0', size=(1, 128)), FakeTensor(..., device='cuda:0', size=(1, 256, 128))), **{}):
Expected size for first two dimensions of batch2 tensor to be: [1, 128] but got: [1, 256].

from user code:
   File "<string>", line 16, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''