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
        super(Model, self).__init__()
        torch.manual_seed(12)
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3)
        torch.manual_seed(12)
        self.conv2 = torch.nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(4, 3, kernel_size=(7, 7), groups=4, bias=False)
        self.conv2 = torch.nn.Conv2d(3, 3, kernel_size=(1, 1), groups=3, bias=False)
        self.conv3 = torch.nn.Conv2d(3, 3, kernel_size=(1, 1), groups=3, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x



func = Model().to('cuda:0')


x = torch.randn(2, 3, 4, 4)

test_inputs = [x]
