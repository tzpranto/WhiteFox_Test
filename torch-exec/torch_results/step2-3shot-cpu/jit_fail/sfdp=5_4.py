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
        self.nhead = 4
        self.dk = 64
        self.dropout_p = 0.1

    def forward(self, q, k, v, attn_mask):
        q = q / math.sqrt(self.dk)
        dots = torch.matmul(q, k.transpose(-2, -1))
        attn_mask = attn_mask.float().masked_fill(attn_mask == 0.0, float('-inf')).masked_fill(attn_mask == 1.0, float(0.0))
        dots = dots + attn_mask
        attn = torch.softmax(dots, dim=-1)
        attn = torch.dropout(attn, self.dropout_p, True)
        out = torch.matmul(attn, v)
        return out


func = Model().to('cpu')


q = torch.randn(16, 32, 64)

k = torch.randn(16, 32, 64)

v = torch.randn(16, 32, 64)
attn_mask = 1

test_inputs = [q, k, v, attn_mask]

# JIT_FAIL
'''
direct:
'int' object has no attribute 'float'

jit:
AttributeError: 'int' object has no attribute 'float'

from user code:
   File "<string>", line 24, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''