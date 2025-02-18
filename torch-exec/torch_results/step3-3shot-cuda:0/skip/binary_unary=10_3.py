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
        self.linear = torch.nn.Linear(64, 256, 16, 16)

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + v1
        v3 = v2.relu()
        return v3


func = Model().to('cuda:0')


x1 = torch.randn(1, 64)

x2 = torch.randn(1, 64)

test_inputs = [x1, x2]
