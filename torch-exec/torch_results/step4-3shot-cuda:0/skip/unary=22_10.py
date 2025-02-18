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
        super.__init__()
        self.w = torch.rand(16, 32)
        self.linear = torch.nn.Linear(16, 32)

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.tanh(v1)
        return v2


func = Model().to('cuda:0')


x1 = torch.randn(1, 16)

test_inputs = [x1]
