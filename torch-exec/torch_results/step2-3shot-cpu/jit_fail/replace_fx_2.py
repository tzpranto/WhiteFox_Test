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
        dropout_layer = torch.nn.Dropout(p=0.25)
        x2 = dropout_layer(x1)
        x3 = torch.dropout(x1, p=2)
        return x2 + x3



func = Model().to('cpu')


x1 = torch.randn(10)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
dropout() missing 1 required positional arguments: "train"

jit:
dropout() missing 1 required positional arguments: "train"
'''