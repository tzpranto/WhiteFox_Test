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
        __output_dim__ = __input_dim__ * __factor__
        self.linear = torch.nn.Linear(__input_dim__, __output_dim__)

    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 + __other__
        v3 = torch.relu(v2)
        return v3


func = Model().to('cuda:0')

x = 1

test_inputs = [x]
