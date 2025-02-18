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

    def test_layer(self):
        t1 = torch.randn(1, 64, 10, 5)
        t2 = torch.randn(64, 64)
        t3 = torch.matmul(t1, t2)
        t4 = self.fc(t3)
        return torch.cat([t1], dim=1)


func = Model().to('cuda:0')

input_tensor = torch.randn(1, 1, 1)

test_inputs = [input_tensor]

# JIT_FAIL
'''
direct:
Module [Model] is missing the required "forward" function

jit:
Module [Model] is missing the required "forward" function
'''