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

class Model(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, attn_mask=None):
        QK = torch.matmul(Q, K.permute(0, 1, 3, 2))
        QK = QK / math.sqrt(K.size(-1))
        if attn_mask is not None:
            QK += attn_mask
        attn_weight = torch.softmax(QK, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0.1, True)
        attn_output = torch.matmul(attn_weight, V)
        return attn_output


func = Model().to('cpu')


Q = torch.randn(2, 8, 64, 16)

K = torch.randn(2, 8, 16, 64)

V = torch.randn(2, 8, 64, 64)

test_inputs = [Q, K, V]

# JIT_FAIL
'''
direct:
Expected size for first two dimensions of batch2 tensor to be: [16, 16] but got: [16, 64].

jit:
Failed running call_function <built-in method matmul of type object at 0x7f22ae45f1c0>(*(FakeTensor(..., size=(2, 8, 64, 16)), FakeTensor(..., size=(2, 8, 64, 16))), **{}):
Expected size for first two dimensions of batch2 tensor to be: [16, 16] but got: [16, 64].

from user code:
   File "<string>", line 19, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''