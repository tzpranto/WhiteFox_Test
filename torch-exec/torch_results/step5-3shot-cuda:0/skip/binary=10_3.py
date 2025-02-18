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

    def __init__(self, w, b):
        super().__init__()
        self.linear = torch.nn.Linear(3, 1)
        self.linear.weight.data = w
        self.linear.bias.data = b

    def forward(self, x1):
        v1 = self.linear(x1)
        return v1


w = 1
b = 1
func = Model(weights, bias).to('cuda:0')


weights = torch.tensor([[-0.1222, 0.2531, -1.6051]], requires_grad=True)

bias = torch.tensor([-0.5976], requires_grad=True)

x1 = torch.randn(1, 3)

test_inputs = [weights, bias, x1]
