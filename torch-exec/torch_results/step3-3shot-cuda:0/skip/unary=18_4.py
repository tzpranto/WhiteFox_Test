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
        self.conv = torch.conv2d(1, 1, 1, stride=1, padding=1)

    def forward(self, x1):
        v1 = nn.Sigmoid()(self.conv(x1))
        return v1



func = Model().to('cuda:0')


x1 = torch.randn(1, 1, 64, 64)

test_inputs = [x1]
