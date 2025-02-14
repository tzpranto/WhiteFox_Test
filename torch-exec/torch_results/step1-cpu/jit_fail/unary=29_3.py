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
        self.conv = torch.nn.ConvTranspose2d(3, 8, 3, stride=1, padding=1)

    def forward(self, x):
        v1 = self.conv(x)
        v2 = x.clamp_min(min_value=0.2)
        v3 = v2.clamp_max(max_value=0.25)
        return v3


func = Model().to('cpu')


x = torch.randn(1, 3, 64, 64)

test_inputs = [x]

# JIT_FAIL
'''
direct:
clamp_min() received an invalid combination of arguments - got (min_value=float, ), but expected one of:
 * (Tensor min)
      didn't match because some of the keywords were incorrect: min_value
 * (Number min)
      didn't match because some of the keywords were incorrect: min_value


jit:
clamp_min() received an invalid combination of arguments - got (min_value=float, ), but expected one of:
 * (Tensor min)
      didn't match because some of the keywords were incorrect: min_value
 * (Number min)
      didn't match because some of the keywords were incorrect: min_value

'''