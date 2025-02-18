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

    def forward(query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ value
        return output


func = Model().to('cuda:0')


__query__ = torch.rand(10, 3, 20)

__key__ = torch.rand(10, 4, 20)

__value__ = torch.randn(10, 4, 20)


__attn_mask__ = torch.tensor([[0, 10, 0], [0, 0, 0], [0, 0, 0]], dtype=torch.float32)

test_inputs = [__query__, __key__, __value__, __attn_mask__]

# JIT_FAIL
'''
direct:
forward() takes 4 positional arguments but 5 were given

jit:
forward() takes 4 positional arguments but 5 were given
'''