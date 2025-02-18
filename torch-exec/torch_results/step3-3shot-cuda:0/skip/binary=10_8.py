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

    def __init__(self, weight):
        super().__init__()
        self.linear = torch.nn.Linear(weight.size(1), weight.size(0))
        self.linear.weight = torch.nn.Parameter(weight)
        self.linear.bias = torch.nn.Parameter(torch.zeros((weight.size(0),)))

    def forward(self, x1):
        v1 = self.linear(x1)
        __output__
        return v1


weight = 1

func = Model(weight).to('cuda:0')


w = torch.randn(5, 3)

x1 = torch.randn(2, 3)

test_inputs = [w, x1]
