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
        self.conv = torch.nn.Conv2d(64, 10, 3, padding=1, groups=32)

    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + 3.0
        v3 = v2 / 100
        v4 = torch.clamp(v3, min=-5, max=6)
        return v4



func = Model().to('cuda:0')


x1 = torch.randn(1, 64, 64, 64)

test_inputs = [x1]
