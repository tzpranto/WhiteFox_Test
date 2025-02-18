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
        self.linear = torch.nn.Linear(in_channels=3, out_channels=257, bias=True)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        v1 = self.linear(x)
        v2 = torch.tanh(v1)
        return v2



func = ModelTanh().to('cuda:0')


x = torch.randn(1, 3)

test_inputs = [x]
