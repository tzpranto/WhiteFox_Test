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
        b['dtype'] = torch.bool
        b['layout'] = torch.strided
        b['device'] = torch.device('cuda:0')
        a['dtype'] = torch.float64
        a['layout'] = torch.strided
        a['device'] = torch.device('cuda:0')
        a['dtype_to'] = torch.int32
        a['dtype_from'] = torch.int32
        b['dtype_to'] = torch.int32
        b['dtype_from'] = torch.int32
        t1 = torch.full([1024, 512], 1, dtype=b['dtype'], layout=b['layout'], device=b['device'], pin_memory=False)
        t2 = t1.to(memory_format=1, memory_format=4, dtype=a['dtype'], non_blocking=False, copy=False, memory_format=None)
        t3 = torch.cumsum(t2, 1)
        return t3



func = Model().to('cuda:0')


x1 = torch.randn(1024, 512, device='cuda:0')

test_inputs = [x1]
