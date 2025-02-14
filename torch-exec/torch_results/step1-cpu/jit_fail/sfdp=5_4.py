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

    def forward(self, x):
        v1 = x @ x.transpose(-2, -1)
        v2 = v1 / np.sqrt(x.shape[-1])
        v3 = v2 + attn_mask
        v4 = torch.softmax(v3, dim=-1)
        v5 = torch.nn.functional.dropout(v4, p=dropout_p, train=True)
        v6 = v5 @ value
        return v6


func = Model().to('cpu')


x = torch.randn(1, 2, 64)

test_inputs = [x]

# JIT_FAIL
'''
direct:
name 'attn_mask' is not defined

jit:
NameError: name 'attn_mask' is not defined

from user code:
   File "<string>", line 18, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''