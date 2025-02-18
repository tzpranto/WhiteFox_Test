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

    def forward(self, query, key, value, attn_mask):
        qk = torch.matmul(query, key.permute(0, 1, 3, 2))
        qk = qk / math.sqrt(64)
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = torch.matmul(attn_weight, value)
        return (output, attn_weight)


func = Model().to('cuda:0')


query = torch.randn([1, 8, 64])

key = torch.randn([1, 8, 64])

value = torch.randn([1, 8, 64])
attn_mask = 1

test_inputs = [query, key, value, attn_mask]

# JIT_FAIL
'''
direct:
permute(sparse_coo): number of dimensions in the tensor input does not match the length of the desired ordering of dimensions i.e. input.dim() = 3 is not equal to len(dims) = 4

jit:
Failed running call_method permute(*(FakeTensor(..., device='cuda:0', size=(1, 8, 64)), 0, 1, 3, 2), **{}):
Dimension out of range (expected to be in range of [-3, 2], but got 3)

from user code:
   File "<string>", line 19, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''