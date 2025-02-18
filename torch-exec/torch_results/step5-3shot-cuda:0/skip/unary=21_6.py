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
        self.conv = torch.nn.Conv2d(1, 1, (2, 2), (1, 1), (1, 1), (0, 0), 1, 1, True)

    def forward(self, x2755):
        x2757 = self.conv(x2755)
        x2761 = torch.tanh(x2757)
        return x2761



func = Model().to('cuda:0')


x2755 = torch.randn(4, 1, 8, 8)

test_inputs = [x2755]
