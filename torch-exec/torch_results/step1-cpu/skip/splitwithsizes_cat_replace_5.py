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
        self.relu = torch.nn.ReLU()
        self.layer1 = torch.nn.Linear(8, 128, 1, bias=True)

    def forward(self, x):
        v1 = self.relu(x)
        v2 = self.layer1(v1)
        (v3, v4, v5) = torch.split(v2, split_size_or_sections=[64, 32, 16], dim=1)
        v6 = torch.cat((v4, v3, v5), 1)
        return v6


func = Model().to('cpu')


x = torch.randn(1, 8)

test_inputs = [x]
