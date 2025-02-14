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

    def __init__(self, device, dtype):
        super().__init__()

    def forward(self, x):
        v1 = torch.full([x.size()[0]], 1, dtype=dtype, layout=torch.strided, device=device, pin_memory=False)
        v2 = v1.to(dtype)
        v3 = v2.to('cpu').to(dtype)
        v4 = torch.cumsum(v3, 1)
        return v4


device = 1
dtype = 1
func = Model('cpu', 'float16').to('cpu')


x = torch.randn(10, 20)

test_inputs = [x]

# JIT_FAIL
'''
direct:
full() received an invalid combination of arguments - got (list, int, pin_memory=bool, device=int, layout=torch.layout, dtype=int), but expected one of:
 * (tuple of ints size, Number fill_value, *, tuple of names names, torch.dtype dtype = None, torch.layout layout = None, torch.device device = None, bool pin_memory = False, bool requires_grad = False)
 * (tuple of ints size, Number fill_value, *, Tensor out = None, torch.dtype dtype = None, torch.layout layout = None, torch.device device = None, bool pin_memory = False, bool requires_grad = False)


jit:
full() received an invalid combination of arguments - got (list, int, pin_memory=bool, device=int, layout=torch.layout, dtype=int), but expected one of:
 * (tuple of ints size, Number fill_value, *, tuple of names names, torch.dtype dtype = None, torch.layout layout = None, torch.device device = None, bool pin_memory = False, bool requires_grad = False)
 * (tuple of ints size, Number fill_value, *, Tensor out = None, torch.dtype dtype = None, torch.layout layout = None, torch.device device = None, bool pin_memory = False, bool requires_grad = False)

'''