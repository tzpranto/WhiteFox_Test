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

    def __init__(self, negative_slope):
        super(Model).__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.negative_slope = negative_slope

    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 > 0
        v3 = self.negative_slope * v1
        v4 = torch.where(v2, v1, v3)
        return v4


negative_slope = 1

func = Model(negative_slope).to('cuda:0')


x1 = torch.randn(1, 3, 64, 64)

test_inputs = [x1]
