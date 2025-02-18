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

    def forward(self, v1, v2, q1, q2, k1, k2):
        qk = (q1 @ k1.transpose(-1, -2) + q2 @ k2.transpose(-1, -2)) / math.sqrt(q1.size(-1))
        qk = qk + torch.autograd.Variable(torch.zeros(qk.shape), requires_grad=False)
        attn_weight = torch.softmax(qk, dim=-1)
        dropout_p = 0.5
        attn_weight = torch.dropout(attn_weight, dropout_p)
        output = attn_weight @ v1 + attn_weight @ v2
        return (v1, v2, q1, q2, k1, k2, output)


func = Model().to('cuda:0')


v1 = torch.randn(2, 8, 4, 8)

v2 = torch.randn(2, 8, 4, 16)

q1 = torch.randn(2, 4, 8)

q2 = torch.randn(2, 4, 8)

k1 = torch.randn(2, 4, 16)

k2 = torch.randn(2, 4, 16)

test_inputs = [v1, v2, q1, q2, k1, k2]

# JIT_FAIL
'''
direct:
Expected size for first two dimensions of batch2 tensor to be: [2, 8] but got: [2, 16].

jit:
Failed running call_function <built-in function matmul>(*(FakeTensor(..., device='cuda:0', size=(2, 4, 8)), FakeTensor(..., device='cuda:0', size=(2, 16, 4))), **{}):
Expected size for first two dimensions of batch2 tensor to be: [2, 8] but got: [2, 16].

from user code:
   File "<string>", line 19, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''