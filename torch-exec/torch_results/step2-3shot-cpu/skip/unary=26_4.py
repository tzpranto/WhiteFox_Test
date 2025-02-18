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
        self.gelu = torch.nn.GELU(1.4901161193847656e-07, 2.44140625e-07)

    def forward(self, x):
        v1 = self.gelu(x)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4



func = Model().to('cpu')


x = torch.randn(1, 3, 64, 64)

test_inputs = [x]
