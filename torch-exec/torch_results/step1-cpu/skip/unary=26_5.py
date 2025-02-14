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
        self.conv = M
        self.relu = torch.nn.ReLU()

    def forward(self, x, negative_slope=0.3):
        v1 = self.conv(x)
        v2 = v1 > 0
        relu = self.relu(v1)
        v3 = torch.where(v2, v1, relu * negative_slope)
        return v3


func = Model().to('cpu')


x = torch.randn(1, 99, 197, 189)

test_inputs = [x]
