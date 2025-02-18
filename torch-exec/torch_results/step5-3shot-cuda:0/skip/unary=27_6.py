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

class clamp_model(nn.Module):

    def __init__(self, min_value=0.0, max_value=0.01):
        super(clamp_model, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(32, 84, kernel_size=5, stride=2, padding=1, groups=32)
        self.relu = nn.ReLU()
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = self.conv3(v3)
        output = self.relu(v4)
        return output



func = clamp_model().to('cuda:0')


x1 = torch.randn(1, 3, 64, 64)

test_inputs = [x1]
