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

    def forward(self, x1):
        b = {}
        a = {}
        b['dtype'] = torch.uint8
        b['layout'] = torch.strided
        b['device'] = torch.device('cuda:0')
        a['dtype'] = torch.float32
        a['layout'] = torch.strided
        a['device'] = torch.device('cuda:0')
        a['dtype_to'] = torch.float64
        a['dtype_from'] = torch.float32
        b['dtype_to'] = torch.float64
        b['dtype_from'] = torch.float32
        t1 = torch.randint(low=-2147483648, high=2147483647, size=[256, 1024], dtype=b['dtype'], layout=b['layout'], device=b['device'])
        t2 = t1.to(dtype=a['dtype'])
        t3 = torch.cumsum(t2, 1)
        return t3



func = Model().to('cuda:0')


x1 = torch.randn(256, 1024, device='cuda:0')

test_inputs = [x1]

# JIT_STATUS
'''
direct:
from is out of bounds for unsigned char

jit:

'''