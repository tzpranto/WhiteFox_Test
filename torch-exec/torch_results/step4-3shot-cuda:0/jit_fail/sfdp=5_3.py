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

    def _forward(self, q, k, v):
        a = torch.matmul(q, k.transpose(2, 3))
        s = a / float(q.shape[-1]) ** 0.5
        m = torch.nn.Softmax(dim=-1)(s + attn_mask)
        m = torch.nn.Dropout(dropout_p)(m)
        x = torch.matmul(m, v)
        return x


func = Model().to('cuda:0')


q = torch.randn(1, 12, 256, 512)

k = torch.randn(1, 12, 256, 512)

v = torch.randn(1, 16, 256, 512)

test_inputs = [q, k, v]

# JIT_FAIL
'''
direct:
Module [Model] is missing the required "forward" function

jit:
Module [Model] is missing the required "forward" function
'''