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

    def __init__(self, other):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.other = torch.nn.Parameter(other, requires_grad=False)

    def forward(self, x):
        return F.relu(self.conv(x) - self.other)


other = 1

func = Model(other).to('cpu')


x = torch.randn(1, 3, 64, 64)

test_inputs = [x]
