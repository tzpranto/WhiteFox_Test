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
        self.conv = torch.nn.Conv1d(14, 14, 1, 1, 0, 1, 1, torch.nn.Conv1d, False, False, False)

    def forward(self, x2):
        v1 = self.conv(x2)
        v2 = torch.tanh(v1)
        return v2



func = Model().to('cuda:0')


x2 = torch.randn(1, 14, 64)

test_inputs = [x2]
