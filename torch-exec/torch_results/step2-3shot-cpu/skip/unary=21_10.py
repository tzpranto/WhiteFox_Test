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
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = torch.tanh(x)
        return x

class ModelTanh(torch.jit.ScriptModule):

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = torch.tanh(x)
        return x



func = ModelTanh().to('cpu')


x = torch.randn(1, 3, 64, 64)

test_inputs = [x]
