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
        self.linear = torch.nn.Linear(3, 10, 1, stride=1, padding=1)

    def forward(self, x2):
        v1 = self.linear(x2)
        v2 = torch.relu(v1)
        return v2


func = Model().to('cuda:0')


x2 = torch.randn(1, 3, 64, 64)

test_inputs = [x2]
