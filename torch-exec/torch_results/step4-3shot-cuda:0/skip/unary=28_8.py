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
        self.linear = torch.nn.Linear(in_features, out_features)
        self.min_value = torch.randn(1, in_features)
        self.max_value = torch.randn(1, in_features)

    def forward(self, x):
        y = self.linear(x)
        z = torch.clamp_min(y, self.min_value)
        return torch.clamp_max(z, self.max_value)


func = Model().to('cuda:0')

x = 1

test_inputs = [x]
