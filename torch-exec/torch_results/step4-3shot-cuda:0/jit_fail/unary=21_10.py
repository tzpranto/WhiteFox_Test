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

class ModelTanh(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=2, padding=1, dilation=2)

    def forward(self, x):
        v1 = self.conv(input)
        v2 = torch.nn.Tanh()(v1)
        return v2



func = ModelTanh().to('cuda:0')


input = torch.randn(1, 3, 256, 256)

test_inputs = [input]

# JIT_FAIL
'''
direct:
Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same or input should be a MKLDNN tensor and weight is a dense tensor

jit:
Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same or input should be a MKLDNN tensor and weight is a dense tensor
'''