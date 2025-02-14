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

    def __init__(self, **kwargs):
        super().__init__()
        self.convt = torch.nn.ConvTranspose2d(3, 8, 4, stride=2, **kwargs)

    def forward(self, x):
        return torch.clamp_max(torch.clamp_min(self.convt(x), min_value=-0.5), max_value=0.5)


func = Model(padding=1, output_padding=1).to('cpu')


x = torch.randn(1, 3, 64, 64)

test_inputs = [x]

# JIT_FAIL
'''
direct:
clamp_min() received an invalid combination of arguments - got (Tensor, min_value=float), but expected one of:
 * (Tensor input, Tensor min, *, Tensor out = None)
 * (Tensor input, Number min, *, Tensor out = None)


jit:
clamp_min() received an invalid combination of arguments - got (Tensor, min_value=float), but expected one of:
 * (Tensor input, Tensor min, *, Tensor out = None)
 * (Tensor input, Number min, *, Tensor out = None)

'''