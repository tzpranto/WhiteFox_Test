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
        self.scale = math.sqrt(8)

    def forward(self, q1, k1, v1, attn_mask):
        q2 = q1 @ k1.transpose(-2, -1)
        q2 = q2 / self.scale
        q2 = q2 + attn_mask
        attn_weight = torch.softmax(q2, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, True)
        output = attn_weight @ v1
        return output


func = Model().to('cuda:0')


q1 = torch.randn(1, 3, 64, 64)

k1 = torch.randn(1, 3, 8, 8)

v1 = torch.randn(1, 3, 8, 8)


attn_mask = torch.ones(1, 64, 64, dtype=torch.long)

test_inputs = [q1, k1, v1, attn_mask]

# JIT_FAIL
'''
direct:
Expected size for first two dimensions of batch2 tensor to be: [3, 64] but got: [3, 8].

jit:
Failed running call_function <built-in function matmul>(*(FakeTensor(..., device='cuda:0', size=(1, 3, 64, 64)), FakeTensor(..., device='cuda:0', size=(1, 3, 8, 8))), **{}):
Expected size for first two dimensions of batch2 tensor to be: [3, 64] but got: [3, 8].

from user code:
   File "<string>", line 20, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''