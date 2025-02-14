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
        super(Model, self).__init__()
        self.batch_split_size = 1
        self.channels_split_size = [2, 4]
        self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, x):
        (x, _) = torch.split(x, [self.batch_split_size, int(x.size(1) * sum(self.channels_split_size) / x.size(1))])
        (x, _) = torch.split(x, self.channels_split_size, dim=2)
        x = self.softmax(x / 2)
        x = torch.cat((x, x, x), dim=2)
        x = x + x
        return x



func = Model().to('cpu')

x = 1

test_inputs = [x]

# JIT_FAIL
'''
direct:
'int' object has no attribute 'size'

jit:
AttributeError: 'int' object has no attribute 'size'

from user code:
   File "<string>", line 22, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''