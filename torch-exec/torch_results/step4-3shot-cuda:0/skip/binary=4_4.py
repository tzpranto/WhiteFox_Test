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

    def __init__(self, x2, x3):
        super().__init__()
        self.fc1 = torch.nn.Linear(2, 2)

    def forward(self, input_tensor, x2=x2, x3=x3):
        v1 = self.fc1(input_tensor)
        v2 = x2 + x3
        v3 = v1 + v2
        return v3


x2 = torch.randn(1, 1, 1, 256)
x3 = torch.randn(1, 1, 1, 256)
func = Model(x2, x3).to('cuda:0')


x2 = torch.randn(1, 1, 1, 256)

x3 = torch.randn(1, 1, 1, 256)
input_tensor = 1

test_inputs = [x2, x3, input_tensor]
