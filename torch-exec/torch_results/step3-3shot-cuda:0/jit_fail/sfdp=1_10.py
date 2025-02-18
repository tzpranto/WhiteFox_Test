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
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(0.16805113117317413)
        self.matmul = torch.nn.Linear(300, 128)

    def forward(self, input, weight):
        qk = torch.matmul(input, weight.transpose(-2, -1))
        scaled_qk = qk.div(1)
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = torch.matmul(dropout_qk, value)
        return output


func = Model().to('cuda:0')


x1 = torch.randn(1, 300)

x2 = torch.randn(300, 128)

test_inputs = [x1, x2]

# JIT_FAIL
'''
direct:
mat1 and mat2 shapes cannot be multiplied (1x300 and 128x300)

jit:
Failed running call_function <built-in method matmul of type object at 0x7f6d6205f1c0>(*(FakeTensor(..., device='cuda:0', size=(1, 300)), FakeTensor(..., device='cuda:0', size=(128, 300))), **{}):
a and b must have same reduction dim, but got [1, 300] X [128, 300].

from user code:
   File "<string>", line 22, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''