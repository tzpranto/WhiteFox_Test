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

    def forward(self, __input1__, __input2__):
        inv_scale = math.sqrt(__input2__.shape[-1])



func = Model().to('cuda:0')

__input1__ = 1
__input2__ = 1

test_inputs = [__input1__, __input2__]

# JIT_FAIL
'''
direct:
'int' object has no attribute 'shape'

jit:
AttributeError: 'int' object has no attribute 'shape'

from user code:
   File "<string>", line 19, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''