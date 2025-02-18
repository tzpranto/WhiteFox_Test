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
        self.conv_transpose = torch.nn.ConvTranspose2d(19, 64, 1, stride=1, padding=0, dilation=3, groups=2, output_padding=1)

    def forward(self, x2):
        v1 = self.conv_transpose(x2)
        v2 = v1 > 0
        v3 = v1 * -0.1
        v4 = torch.where(v2, v1, v3)
        return v4



func = Model().to('cuda:0')


x2 = torch.randn(8, 19, 4, 4)

test_inputs = [x2]
