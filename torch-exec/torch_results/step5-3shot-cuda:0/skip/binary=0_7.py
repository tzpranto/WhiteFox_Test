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

class Model(torch.nn.Sequential):

    def __init__(self):
        super().__init__([torch.nn.Conv2d(3, 8, 1, stride=1, padding=1), torch.nn.Conv2d(8, 4, 1, stride=1, padding=1)])

    def forward(self, x1, other=1):
        v1 = self.conv(x1)
        if other == 1:
            other = torch.randn(v1.shape)
        v2 = v1 + other
        return v2



func = Model().to('cuda:0')


x1 = torch.randn(1, 3, 64, 64)

test_inputs = [x1]
