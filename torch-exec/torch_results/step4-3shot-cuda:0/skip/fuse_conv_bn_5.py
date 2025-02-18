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
        self.conv = torch.nn.ConvTransfom(3, 1, 1)
        self.bn = torch.nn.BatchNorm1d(3)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x



func = Model().to('cuda:0')


x = torch.randn(1, 3, 1, 1)

test_inputs = [x]
