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
        self.conv1 = torch.nn.Conv2d(80, 80, (1, 1), (2, 2), (3, 3), (4, 4))
        self.conv2 = torch.nn.Conv2d(80, 80, (3,), (3, 3))
        self.conv3 = torch.nn.Conv2d(80, 80, (2, 2), (1, 2))
        self.conv4 = torch.nn.Conv2d(80, 80, 1)
        self.conv5 = torch.nn.Conv2d(13, 13, 1, (2, 2), (3, 3), (4, 4), 3)
        self.conv6 = torch.nn.Conv2d(13, 13, (2, 2), (2, 2), (3, 3))

    def forward(self, x):
        x = self.conv1(x)
        x = torch.tanh(x)
        x = self.conv2(x)
        x = torch.tanh(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = torch.tanh(x)
        x = self.conv6(x)
        return x



func = ModelTanh().to('cuda:0')


x = torch.randn(1, 13, 26, 40)

test_inputs = [x]
