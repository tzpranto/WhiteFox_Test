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
        self.conv = torch.nn.Conv2d(1, 8, 3, stride=1, padding=1)
        self.conv2 = torch.nn.ConvTranspose2d(8, 4, 5, stride=1, padding=1)

    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = nn.Activation('sigmoid')(v1)
        v3 = self.conv2(v2)
        v4 = nn.Sigmoid()(v3)
        return v4



func = Model().to('cuda:0')


x1 = torch.randn(1, 1, 128, 128)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
module 'torch.nn' has no attribute 'Activation'

jit:
AttributeError: module 'torch.nn' has no attribute 'Activation'

from user code:
   File "<string>", line 22, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''