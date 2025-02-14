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
        self.conv = torch.nn.Conv(1, 1, 2)
        self.bn = torch.nn.BatchNorm2d(1)

    def init_parameters(self, conv, bn):
        self.conv.weight = conv.weight
        self.conv.bias = conv.bias
        self.bn.weight = bn.weight
        self.bn.bias = bn.bias
        self.bn.running_mean = bn.running_mean
        self.bn.running_var = bn.running_var

    def forward(self, x):
        x1 = self.conv(x)
        x2 = torch.nn.functional.batch_norm(x1, self.bn.running_mean, self.bn.running_var, self.bn.weight, self.bn.bias, training=False, eps=self.bn.eps)
        return x2


func = Model().to('cpu')


x = torch.randn(1, 1, 2, 2)

test_inputs = [x]
